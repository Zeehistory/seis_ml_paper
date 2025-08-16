import os, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -------------------
# Reproducibility
# -------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------
# Paths
# -------------------
FEATURES_CSV = Path("/home/zsyed/orcd/pool/seis-data/processed/features.csv")
RESULTS_DIR  = Path("/orcd/home/002/zsyed/seis-paper/results")
MODELS_DIR   = Path("/orcd/home/002/zsyed/seis-paper/models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------
# Load Data
# -------------------
print("[INFO] Loading features CSV")
df = pd.read_csv(FEATURES_CSV)
df["label"] = (df["Magnitude"] >= 6.0).astype(int)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_cols].drop("label", axis=1).values.astype(np.float32)
y = df["label"].values.astype(np.int64)

scaler = StandardScaler()
X = scaler.fit_transform(X)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Deep Network
# -------------------
class DeepNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes)
        )
    def forward(self, x):
        return self.layers(x)

# -------------------
# Training / Eval
# -------------------
def run_fold(train_idx, val_idx, fold, hidden_dim=256, dropout=0.3):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Weighted sampler for class balance
    class_counts = np.bincount(y_train)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = weights[y_train]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=64)

       # Model, loss, optim
    model = DeepNet(input_dim=X.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 🔧 FIX: OneCycleLR aligned with 50 epochs
    num_epochs = 50
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )

    # Training loop
    best_loss, patience, patience_ctr = float("inf"), 5, 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        y_true, y_pred, y_prob, val_loss = [], [], [], 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                val_loss += criterion(out, yb).item()
                preds = out.argmax(1)
                probs = torch.softmax(out, 1)[:,1]
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
        val_loss /= len(val_loader)

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        aucpr = average_precision_score(y_true, y_prob)

        print(f"[Fold {fold} | Epoch {epoch}] "
              f"ValLoss={val_loss:.4f} Acc={acc:.3f} BalAcc={bal_acc:.3f} "
              f"MCC={mcc:.3f} AUC={auc:.3f} AUCPR={aucpr:.3f}")

        if val_loss < best_loss:
            best_loss, patience_ctr = val_loss, 0
            torch.save(model.state_dict(), MODELS_DIR / f"best_fold{fold}_hd{hidden_dim}_do{dropout}.pth")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    return {
        "val_acc": acc, "bal_acc": bal_acc, "mcc": mcc,
        "auc": auc, "aucpr": aucpr, "report": classification_report(y_true, y_pred, digits=4)
    }

# -------------------
# Cross-Validation + Ablation
# -------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
hidden_dims = [128, 256, 512]
dropouts = [0.3, 0.5]

all_results = []
for hd in hidden_dims:
    for do in dropouts:
        print(f"\n[INFO] Running ablation: hidden_dim={hd}, dropout={do}")
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            metrics = run_fold(train_idx, val_idx, fold, hidden_dim=hd, dropout=do)
            fold_results.append(metrics)
        df_res = pd.DataFrame(fold_results)
        df_res["hidden_dim"] = hd
        df_res["dropout"] = do
        all_results.append(df_res)

cv_results = pd.concat(all_results, ignore_index=True)
cv_results.to_csv(RESULTS_DIR / "cv_ablation_results.csv", index=False)

# -------------------
# Baseline Models
# -------------------
print("[INFO] Training baselines")
baselines = {}
for name, clf in {
    "LogReg": LogisticRegression(max_iter=1000),
    "RF": RandomForestClassifier(n_estimators=200),
    "XGB": xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss"
    )
}.items():
    clf.fit(X, y)
    preds = clf.predict(X)
    baselines[name] = {
        "acc": accuracy_score(y, preds),
        "bal_acc": balanced_accuracy_score(y, preds),
        "mcc": matthews_corrcoef(y, preds)
    }

with open(RESULTS_DIR / "baselines.json", "w") as f:
    json.dump(baselines, f, indent=4)

print("[INFO] Baseline results saved:", baselines)

# -------------------
# Statistical Significance
# -------------------
metrics_to_test = ["val_acc","bal_acc","mcc","auc","aucpr"]
summary = {}
for m in metrics_to_test:
    vals = cv_results[m].values
    mean, ci95 = np.mean(vals), stats.t.interval(
        0.95, len(vals)-1, loc=np.mean(vals), scale=stats.sem(vals)
    )
    summary[m] = {"mean": float(mean), "ci95_low": float(ci95[0]), "ci95_high": float(ci95[1])}

with open(RESULTS_DIR / "statistical_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("[INFO] Statistical significance computed. Done.")
