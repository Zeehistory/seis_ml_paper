#!/usr/bin/env python3
import os, json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# -------------------------------
# Config
# -------------------------------
RESULTS_DIR = "seis-paper/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------
# Helpers
# -------------------------------
def log(msg):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}", flush=True)

def evaluate_and_save(y_true, y_pred, y_proba, model_name, fold):
    """Evaluate metrics and save per-fold arrays"""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=2, output_dict=True)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else np.nan

    log(f"\n[RESULT] {model_name} Fold {fold} Confusion Matrix:\n{cm}")
    log(f"[RESULT] {model_name} Fold {fold} Balanced Acc: {bal_acc:.3f}, Macro F1: {macro_f1:.3f}, AUC: {auc:.3f}")

    # Save per-fold arrays
    np.save(os.path.join(RESULTS_DIR, f"y_true_{model_name}_fold{fold}.npy"), y_true)
    np.save(os.path.join(RESULTS_DIR, f"y_pred_{model_name}_fold{fold}.npy"), y_pred)
    if y_proba is not None:
        np.save(os.path.join(RESULTS_DIR, f"y_proba_{model_name}_fold{fold}.npy"), y_proba)

    return bal_acc, macro_f1, auc

# -------------------------------
# Main
# -------------------------------
def main():
    log("Starting ML training")

    # Load features
    csv_path = "/home/zsyed/orcd/pool/seis-data/processed/features.csv"
    df = pd.read_csv(csv_path)
    log(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {csv_path}")

    # Label = large quake (>=6.0) vs small (<6.0)
    df["label"] = (df["Magnitude"] >= 6.0).astype(int)
    log("Created label column: 0=small quake (<6.0), 1=large quake (>=6.0)")

    # Features: drop identifiers + trivial leakage
    drop_cols = ["EventID", "WaveformFile", "Magnitude"]
    X = df.drop(columns=drop_cols + ["label"])
    y = df["label"].values
    log(f"Feature matrix shape: {X.shape}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        log(f"\n[CV] Fold {fold}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply SMOTE only on training split
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        log(f"[CV] Applied SMOTE: {X_train.shape[0]} → {X_res.shape[0]} samples")

        # --- RandomForest ---
        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_res, y_res)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)
        rf_bal, rf_f1, rf_auc = evaluate_and_save(y_test, y_pred, y_proba, "rf", fold)
        results.append({"model": "rf", "fold": fold, "bal_acc": rf_bal, "macro_f1": rf_f1, "auc": rf_auc})

        # --- XGBoost ---
        xgb_clf = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=len(y_res[y_res == 0]) / len(y_res[y_res == 1])
        )
        xgb_clf.fit(X_res, y_res)
        y_pred = xgb_clf.predict(X_test)
        y_proba = xgb_clf.predict_proba(X_test)
        xgb_bal, xgb_f1, xgb_auc = evaluate_and_save(y_test, y_pred, y_proba, "xgb", fold)
        results.append({"model": "xgb", "fold": fold, "bal_acc": xgb_bal, "macro_f1": xgb_f1, "auc": xgb_auc})

        fold += 1

    # Save CV results as CSV
    cv_path = os.path.join(RESULTS_DIR, "cv_ablation_results.csv")
    pd.DataFrame(results).to_csv(cv_path, index=False)
    log(f"Saved CV results to {cv_path}")

    # Save statistical summary (means/stds)
    summary = {}
    for model in ["rf", "xgb"]:
        model_df = pd.DataFrame([r for r in results if r["model"] == model])
        summary[model] = {
            "bal_acc_mean": model_df["bal_acc"].mean(),
            "bal_acc_std": model_df["bal_acc"].std(),
            "macro_f1_mean": model_df["macro_f1"].mean(),
            "macro_f1_std": model_df["macro_f1"].std(),
            "auc_mean": model_df["auc"].mean(),
            "auc_std": model_df["auc"].std(),
        }
    with open(os.path.join(RESULTS_DIR, "statistical_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved statistical summary to {RESULTS_DIR}/statistical_summary.json")

    # Train final model on all data with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    final_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_res, y_res)
    outpath = os.path.join(RESULTS_DIR, "ml_model.pkl")
    joblib.dump(final_model, outpath)
    log(f"Saved final RandomForest model to {outpath}")

    log("Finished ML training")

if __name__ == "__main__":
    log(f"Starting ML training at {datetime.now().strftime('%c')}")
    main()
    log(f"Finished ML training at {datetime.now().strftime('%c')}")
