#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# -------------------------------
# Helpers
# -------------------------------
def log(msg):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}", flush=True)

def evaluate(y_true, y_pred, label=""):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=2)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    log(f"\n[RESULT] {label} Confusion Matrix:\n{cm}")
    log(f"\n[RESULT] {label} Classification Report:\n{report}")
    log(f"[RESULT] {label} Balanced Accuracy: {bal_acc:.3f}")
    log(f"[RESULT] {label} Macro F1: {macro_f1:.3f}")
    return bal_acc, macro_f1

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
    rf_scores, xgb_scores = [], []

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        log(f"\n[CV] Fold {fold}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply SMOTE only on training split
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        log(f"[CV] Applied SMOTE: {X_train.shape[0]} → {X_res.shape[0]} samples")

        # RandomForest
        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_res, y_res)
        y_pred = rf.predict(X_test)
        rf_bal, rf_f1 = evaluate(y_test, y_pred, label=f"RF Fold {fold}")
        rf_scores.append((rf_bal, rf_f1))

        # XGBoost
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
        xgb_bal, xgb_f1 = evaluate(y_test, y_pred, label=f"XGB Fold {fold}")
        xgb_scores.append((xgb_bal, xgb_f1))

        fold += 1

    # Aggregate results
    def summarize(scores, name):
        scores = np.array(scores)
        log(f"\n[SUMMARY] {name} Mean Balanced Acc: {scores[:,0].mean():.3f} ± {scores[:,0].std():.3f}")
        log(f"[SUMMARY] {name} Mean Macro F1: {scores[:,1].mean():.3f} ± {scores[:,1].std():.3f}")

    summarize(rf_scores, "RandomForest")
    summarize(xgb_scores, "XGBoost")

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
    outpath = "/home/zsyed/orcd/pool/seis-data/processed/ml_model.pkl"
    joblib.dump(final_model, outpath)
    log(f"Saved final RandomForest model to {outpath}")

    log("Finished ML training")

if __name__ == "__main__":
    log(f"Starting ML training at {datetime.now().strftime('%c')}")
    main()
    log(f"Finished ML training at {datetime.now().strftime('%c')}")
