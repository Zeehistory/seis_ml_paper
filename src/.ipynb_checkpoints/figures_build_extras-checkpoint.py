#!/usr/bin/env python3
"""
figures_build_extras.py

Adds the last IEEE-Access-ready figures:
  1) Calibration curves + Brier scores for DL, ML-RF, ML-XGB
  2) DL ablation grid (hidden_dim × dropout) with best cell highlighted
  3) Model comparison table (CSV + PNG + LaTeX)

Inputs (must exist if the corresponding figure is to be produced):
DL dir (deep learning): 
  - y_true.npy, y_proba.npy
  - ml_cv_ablation_results.csv   (cols: hidden_dim, dropout, val_acc, bal_acc, mcc, auc, aucpr)
  - metrics.json                 (optional: test_* keys)

ML dir (traditional ML):
  - cv_ablation_results.csv      (cols: model, fold, bal_acc, macro_f1, auc)
  - y_true_{rf|xgb}_fold*.npy
  - y_proba_{rf|xgb}_fold*.npy   (N,2) or (N,) = prob of class 1

Outputs into /orcd/home/002/zsyed/seis-paper/figures:
  - dl_calibration.{png,pdf}
  - ml_rf_calibration.{png,pdf}
  - ml_xgb_calibration.{png,pdf}
  - dl_ablation_grid_auc.{png,pdf}
  - summary_table.{csv,png,tex}
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                # headless safe
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# ---------- paths ----------
BASE_DIR = Path("/orcd/home/002/zsyed/seis-paper")
DL_DIR   = BASE_DIR / "results"
ML_DIR   = BASE_DIR / "src" / "seis-paper" / "ml_results"
FIG_DIR  = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# IEEE-ish defaults (single-column friendly)
sns.set_theme(context="paper", style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (3.5, 2.8),   # inches
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def savefig(name: str):
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close()

def _exists(p: Path) -> bool:
    return p.exists()

def _concat_folds(dirpath: Path, pattern: str):
    files = sorted(dirpath.glob(pattern))
    if not files:
        return None
    arrs = []
    for f in files:
        try:
            arrs.append(np.load(f, allow_pickle=False))
        except Exception:
            return None
    try:
        return np.concatenate(arrs, axis=0)
    except Exception:
        return None

# ---------- 1) Calibration & Brier ----------
def _calib_plot(y_true, y_prob, title, out_name):
    """Reliability curve + histogram; returns Brier score."""
    bs = brier_score_loss(y_true, y_prob)

    fig = plt.figure(figsize=(3.5, 2.8))
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    # reliability curve
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2, label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")

    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(f"{title}\nBrier={bs:.3f}")
    plt.legend(frameon=True)
    savefig(out_name)
    return bs

def build_calibration_all():
    # DL
    ytrue_dl = DL_DIR / "y_true.npy"
    yprob_dl = DL_DIR / "y_proba.npy"
    if _exists(ytrue_dl) and _exists(yprob_dl):
        yt = np.load(ytrue_dl)
        yp = np.load(yprob_dl)
        if yp.ndim == 2:  # multiclass → take class-1 if binary-like, else skip
            if yp.shape[1] >= 2:
                yp = yp[:, -1]
            else:
                yp = None
        if yp is not None:
            print("[ok] DL calibration")
            _calib_plot(yt, yp, "DL Calibration", "dl_calibration")
        else:
            print("[skip] DL calibration (probas not 1-D)")

    # ML per model
    for mdl in ["rf", "xgb"]:
        yt = _concat_folds(ML_DIR, f"y_true_{mdl}_fold*.npy")
        yp = _concat_folds(ML_DIR, f"y_proba_{mdl}_fold*.npy")
        if yt is None or yp is None:
            print(f"[skip] ML {mdl} calibration (missing folds)")
            continue
        # reduce to prob of class 1
        if yp.ndim == 2 and yp.shape[1] >= 2:
            yp = yp[:, -1]
        _calib_plot(yt, yp, f"ML {mdl.upper()} Calibration", f"ml_{mdl}_calibration")

# ---------- 2) DL ablation grid with best cell ----------
def build_ablation_grid(metric="auc"):
    csv = DL_DIR / "ml_cv_ablation_results.csv"
    if not _exists(csv):
        print("[skip] DL ablation grid (ml_cv_ablation_results.csv not found)")
        return
    df = pd.read_csv(csv)
    needed = {"hidden_dim", "dropout", metric}
    if not needed.issubset(df.columns):
        print(f"[skip] DL ablation grid (missing columns: {needed - set(df.columns)})")
        return

    piv = df.pivot_table(index="dropout", columns="hidden_dim", values=metric, aggfunc="mean")

    # find best
    best_idx = np.unravel_index(np.nanargmax(piv.values), piv.shape)
    best_drop = piv.index[best_idx[0]]
    best_dim  = piv.columns[best_idx[1]]
    best_val  = piv.values[best_idx]

    plt.figure(figsize=(3.5, 3.0))
    ax = sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", cbar=True)
    ax.set_title(f"DL Ablation — {metric.upper()} (mean)")
    ax.set_xlabel("hidden_dim")
    ax.set_ylabel("dropout")

    # highlight best with a rectangle
    ax.add_patch(plt.Rectangle((best_idx[1], best_idx[0]), 1, 1,
                               fill=False, edgecolor="red", linewidth=2.0))
    # annotate best cell
    ax.text(best_idx[1]+0.5, best_idx[0]+0.5, "★", ha="center", va="center",
            color="red", fontsize=14, fontweight="bold")

    savefig(f"dl_ablation_grid_{metric}")
    print(f"[ok] DL ablation grid: best {metric}={best_val:.3f} at dropout={best_drop}, hidden_dim={best_dim}")

# ---------- 3) Model comparison table (CSV + PNG + LaTeX) ----------
def build_summary_table():
    rows = []

    # DL test metrics (optional but nice)
    metrics_json = DL_DIR / "metrics.json"
    if _exists(metrics_json):
        mj = json.loads(metrics_json.read_text())
        rows.append({
            "Model": "DL",
            "Metric Source": "test",
            "ACC": mj.get("test_acc"),
            "AUC": mj.get("test_auc"),
            "PREC": mj.get("test_prec"),
            "REC": mj.get("test_rec"),
            "F1": mj.get("test_f1"),
        })

    # DL CV means from ml_statistical_summary.json (already in your results/)
    dl_stat = DL_DIR / "ml_statistical_summary.json"
    if _exists(dl_stat):
        s = json.loads(dl_stat.read_text())
        rows.append({
            "Model": "DL (CV mean)",
            "Metric Source": "cv_mean",
            "ACC": s.get("val_acc", {}).get("mean"),
            "AUC": s.get("auc", {}).get("mean"),
            "BAL_ACC": s.get("bal_acc", {}).get("mean"),
            "MCC": s.get("mcc", {}).get("mean"),
            "AUPRC": s.get("aucpr", {}).get("mean"),
        })

    # ML CV means from ml_results/cv_ablation_results.csv
    ml_csv = ML_DIR / "cv_ablation_results.csv"
    if _exists(ml_csv):
        mldf = pd.read_csv(ml_csv)
        if "model" in mldf.columns:
            g = mldf.groupby("model").agg({
                "bal_acc": "mean", "macro_f1": "mean", "auc": "mean"
            })
            for mdl in g.index:
                rows.append({
                    "Model": f"ML {mdl.upper()} (CV mean)",
                    "Metric Source": "cv_mean",
                    "BAL_ACC": g.loc[mdl, "bal_acc"],
                    "MACRO_F1": g.loc[mdl, "macro_f1"],
                    "AUC": g.loc[mdl, "auc"],
                })

    if not rows:
        print("[skip] summary table (no sources)")
        return

    df = pd.DataFrame(rows)
    # Order sensible columns if present
    preferred = ["Model","Metric Source","ACC","BAL_ACC","AUC","AUPRC","MCC","PREC","REC","F1","MACRO_F1"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    # CSV
    out_csv = FIG_DIR / "summary_table.csv"
    df.to_csv(out_csv, index=False)

    # LaTeX
    try:
        (FIG_DIR / "summary_table.tex").write_text(df.to_latex(index=False, float_format="%.3f"))
    except Exception:
        pass

    # PNG (matplotlib table)
    fig, ax = plt.subplots(figsize=(6.8, 2 + 0.28*len(df)))
    ax.axis("off")
    tbl = ax.table(cellText=df.round(3).values,
                   colLabels=df.columns,
                   loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.2)
    ax.set_title("Model Comparison Summary", pad=10)
    savefig("summary_table")
    print(f"[ok] summary table → {out_csv.name}, summary_table.png/.tex")

# ---------- main ----------
def main():
    print("[info] building calibration curves…")
    build_calibration_all()

    print("[info] building DL ablation grid…")
    build_ablation_grid(metric="auc")   # change to "bal_acc" if that’s your headline

    print("[info] building model comparison table…")
    build_summary_table()

    print("✅ Extras done. Files in:", FIG_DIR)

if __name__ == "__main__":
    main()
