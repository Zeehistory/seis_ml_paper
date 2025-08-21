#!/usr/bin/env python3
"""
figures_build_all.py

Generate the full IEEE Access figure set from BOTH deep-learning (DL) and classic ML outputs.

Looks here:
- DL   : /orcd/home/002/zsyed/seis-paper/results
- ML   : /orcd/home/002/zsyed/seis-paper/src/seis-paper/ml_results
Saves here:
- FIGS : /orcd/home/002/zsyed/seis-paper/figures

Outputs (when data exists)
DL:
  - dl_training_loss, dl_val_accuracy, dl_lr_schedule
  - dl_confusion_matrix, dl_roc_curve, dl_pr_curve
  - boxplot_*, heatmap_*, bar_dropout_* (from DL ablations)
  - radar_metrics (from DL statistical summary)
  - umap_embeddings, tsne_embeddings
  - baseline_*  (from baselines.json, if present)
  - rf_feature_importance / xgb_feature_importance (if *_feature_importance.npy exist)

ML (per model: rf, xgb):
  - ml_<model>_confusion_matrix, ml_<model>_roc_curve, ml_<model>_pr_curve
  - ml_cv_bal_acc, ml_cv_macro_f1, ml_cv_auc  (mean ± 95% CI)
  - ml_radar_<model>  (from ml_results/statistical_summary.json)
  - ml_<model>_feature_importance  (if *_feature_importance.npy exist in DL dir)

Safe to run repeatedly. Missing inputs are skipped with a message.
"""

import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                 # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.manifold import TSNE
import umap

# -------------------
# Paths
# -------------------
BASE_DIR = Path("/orcd/home/002/zsyed/seis-paper")
DL_DIR   = BASE_DIR / "results"
ML_DIR   = BASE_DIR / "src" / "seis-paper" / "ml_results"
FIG_DIR  = BASE_DIR / "figures"

for d in (DL_DIR, ML_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -------------------
# Figure aesthetics (IEEE-ish)
# -------------------
# Single-column ~3.5", double-column ~7.2"
INCH = 1.0
W_SINGLE = 3.5 * INCH
H_SINGLE = 2.6 * INCH
W_DOUBLE = 7.2 * INCH
H_DOUBLE = 4.2 * INCH

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
})

warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# -------------------
# Utilities
# -------------------
def exists(p: Path) -> bool:
    return p.exists()

def savefig(name: str, w=W_SINGLE, h=H_SINGLE):
    fig = plt.gcf()
    fig.set_size_inches(w, h)
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close()

def nice_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)

def log_skip(msg: str):
    print(f"[skip] {msg}")

def print_plan():
    print("[plan] Scanning artifacts...")
    have = lambda p: "✓" if p.exists() else "–"
    print(" DL dir:", DL_DIR)
    for f in [
        "train_loss.npy","val_acc.npy","lr_schedule.npy",
        "y_true.npy","y_pred.npy","y_proba.npy",
        "ml_cv_ablation_results.csv",
        "statistical_summary.json","ml_statistical_summary.json",
        "embeddings_val.npy","embeddings.npy",
        "baselines.json","rf_feature_importance.npy","xgb_feature_importance.npy",
    ]:
        print(f"   {have(DL_DIR/f):2} {f}")
    print(" ML dir:", ML_DIR)
    for f in ["cv_ablation_results.csv","statistical_summary.json"]:
        print(f"   {have(ML_DIR/f):2} {f}")
    for mdl in ["rf","xgb"]:
        yt = list(ML_DIR.glob(f"y_true_{mdl}_fold*.npy"))
        yp = list(ML_DIR.glob(f"y_pred_{mdl}_fold*.npy"))
        pr = list(ML_DIR.glob(f"y_proba_{mdl}_fold*.npy"))
        print(f"   folds {mdl}: y_true={len(yt)}  y_pred={len(yp)}  y_proba={len(pr)}")

def concat_folds(pattern: str, base: Path) -> np.ndarray | None:
    files = sorted(base.glob(pattern))
    if not files:
        return None
    stacks = []
    for f in files:
        try:
            stacks.append(np.load(f, allow_pickle=False))
        except Exception:
            return None
    try:
        return np.concatenate(stacks, axis=0)
    except Exception:
        return None

def safe_json_load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

# -------------------
# DL: Training curves
# -------------------
def plot_training_curves_dl():
    loss_p = DL_DIR / "train_loss.npy"
    acc_p  = DL_DIR / "val_acc.npy"
    lr_p   = DL_DIR / "lr_schedule.npy"
    if not (exists(loss_p) and exists(acc_p)):
        return log_skip("DL training curves (need train_loss.npy & val_acc.npy)")

    train_loss = np.load(loss_p)
    val_acc    = np.load(acc_p)

    # If per-fold arrays exist, show fold separators (improves readability for concatenated curves)
    folds_loss = sorted(DL_DIR.glob("train_loss_fold*.npy"))
    fold_cuts = []
    if folds_loss:
        cum = 0
        for f in folds_loss[:-1]:
            cum += len(np.load(f))
            fold_cuts.append(cum)

    # Training loss
    plt.figure()
    ax = plt.gca(); nice_ax(ax)
    ax.plot(train_loss, label="Train Loss")
    for cut in fold_cuts:
        ax.axvline(cut, ls="--", lw=0.8, color="gray", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Loss")
    ax.legend()
    savefig("dl_training_loss", w=W_DOUBLE, h=H_SINGLE)

    # Validation accuracy
    plt.figure()
    ax = plt.gca(); nice_ax(ax)
    ax.plot(val_acc, label="Val Acc")
    for cut in fold_cuts:
        ax.axvline(cut, ls="--", lw=0.8, color="gray", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Validation Accuracy")
    ax.legend()
    savefig("dl_val_accuracy", w=W_DOUBLE, h=H_SINGLE)

    # LR schedule (optional)
    if exists(lr_p):
        lr = np.load(lr_p)
        plt.figure()
        ax = plt.gca(); nice_ax(ax)
        ax.plot(lr)
        ax.set_xlabel("Step"); ax.set_ylabel("Learning Rate"); ax.set_title("OneCycleLR Schedule")
        savefig("dl_lr_schedule", w=W_DOUBLE, h=H_SINGLE)

# -------------------
# DL: Confusion + ROC/PR
# -------------------
def plot_confusion_and_curves_dl():
    yt_p = DL_DIR / "y_true.npy"
    yp_p = DL_DIR / "y_pred.npy"
    prob_p = DL_DIR / "y_proba.npy"
    if not (exists(yt_p) and exists(yp_p)):
        return log_skip("DL confusion/curves (need y_true.npy & y_pred.npy)")

    y_true = np.load(yt_p)
    y_pred = np.load(yp_p)
    y_hat  = y_pred.argmax(1) if y_pred.ndim == 2 else y_pred

    # color by normalized matrix; annotate counts
    cm_norm = confusion_matrix(y_true, y_hat, normalize="true")
    cm_cnt  = confusion_matrix(y_true, y_hat)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm_norm)
    disp.plot(cmap="Blues", values_format=".2f", colorbar=False, ax=ax)
    ax.set_title("DL Normalized Confusion Matrix")
    # annotate counts
    for (i, j), v in np.ndenumerate(cm_cnt):
        ax.text(j, i, f"{v}", ha="center", va="center", color="navy", fontsize=9, fontweight="bold")
    savefig("dl_confusion_matrix", w=W_SINGLE, h=W_SINGLE)

    # ROC / PR (prefer probabilities if available)
    if exists(prob_p):
        p = np.load(prob_p)
        if p.ndim == 1:  # binary
            fpr, tpr, _ = roc_curve(y_true, p); roc_auc = auc(fpr, tpr)
            plt.figure(); ax = plt.gca(); nice_ax(ax)
            ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
            ax.plot([0,1],[0,1],"--",color="gray")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("DL ROC Curve")
            ax.legend()
            savefig("dl_roc_curve", w=W_SINGLE, h=W_SINGLE)

            prec, rec, _ = precision_recall_curve(y_true, p)
            ap = average_precision_score(y_true, p)
            plt.figure(); ax = plt.gca(); nice_ax(ax)
            ax.plot(rec, prec, label=f"AP={ap:.2f}")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("DL PR Curve")
            ax.legend()
            savefig("dl_pr_curve", w=W_SINGLE, h=W_SINGLE)
        else:           # multiclass one-vs-rest
            plt.figure(); ax = plt.gca(); nice_ax(ax)
            for i in range(p.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i, p[:, i])
                ax.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr,tpr):.2f})")
            ax.plot([0,1],[0,1],"--",color="gray")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("DL ROC Curves")
            ax.legend()
            savefig("dl_roc_curve", w=W_DOUBLE, h=H_SINGLE)

            plt.figure(); ax = plt.gca(); nice_ax(ax)
            for i in range(p.shape[1]):
                prec, rec, _ = precision_recall_curve(y_true == i, p[:, i])
                ap = average_precision_score(y_true == i, p[:, i])
                ax.plot(rec, prec, label=f"Class {i} (AP={ap:.2f})")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("DL PR Curves")
            ax.legend()
            savefig("dl_pr_curve", w=W_DOUBLE, h=H_SINGLE)
    else:
        log_skip("DL ROC/PR (missing y_proba.npy)")

# -------------------
# DL: CV ablations (box/heat/bar)
# -------------------
def plot_cv_dl():
    csv = DL_DIR / "ml_cv_ablation_results.csv"
    if not exists(csv):
        return log_skip("DL CV ablations (need ml_cv_ablation_results.csv)")
    df = pd.read_csv(csv)
    metrics = [m for m in ["val_acc","bal_acc","mcc","auc","aucpr"] if m in df.columns]
    if not metrics:
        return log_skip("DL CV ablations (no metric columns)")

    for m in metrics:
        # boxplot (hidden_dim x dropout)
        plt.figure()
        ax = plt.gca(); nice_ax(ax)
        sns.boxplot(x="hidden_dim", y=m, hue="dropout", data=df, palette="Set2", ax=ax)
        sns.stripplot(x="hidden_dim", y=m, data=df, color="black", alpha=0.45, ax=ax)
        ax.set_title(f"DL Ablation — {m.upper()} by hidden_dim / dropout")
        ax.legend(title="dropout")
        savefig(f"boxplot_{m}", w=W_DOUBLE, h=H_DOUBLE)

        # heatmap (mean over folds)
        plt.figure()
        ax = plt.gca()
        piv = df.pivot_table(index="dropout", columns="hidden_dim", values=m, aggfunc="mean")
        sns.heatmap(piv, annot=True, cmap="viridis", ax=ax)
        ax.set_title(f"DL Ablation Heatmap — {m.upper()}")
        savefig(f"heatmap_{m}", w=W_SINGLE, h=W_SINGLE)

        # bar vs dropout (mean ± sd)
        plt.figure()
        ax = plt.gca(); nice_ax(ax)
        sns.barplot(x="dropout", y=m, data=df, errorbar="sd", ax=ax)
        ax.set_title(f"DL Ablation — {m.upper()} vs Dropout")
        savefig(f"bar_dropout_{m}", w=W_SINGLE, h=H_SINGLE)

# -------------------
# DL: Radar
# -------------------
def plot_radar_dl():
    # support either results/statistical_summary.json or results/ml_statistical_summary.json
    jsA = DL_DIR / "statistical_summary.json"
    jsB = DL_DIR / "ml_statistical_summary.json"
    src = jsA if exists(jsA) else (jsB if exists(jsB) else None)
    if not src:
        return log_skip("DL radar (no statistical summary)")

    stats_dict = safe_json_load(src)
    labels, means = [], []
    for k, v in stats_dict.items():
        if isinstance(v, dict) and "mean" in v and v["mean"] is not None:
            labels.append(k.upper()); means.append(float(v["mean"]))
    if not labels:
        return log_skip("DL radar (summary has no 'mean' fields)")

    # radar
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    vals = means + means[:1]
    angs = angles + angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angs, vals, "o-", linewidth=2, label="DL")
    ax.fill(angs, vals, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title("DL Statistical Summary (means)")
    savefig("radar_metrics", w=W_SINGLE, h=W_SINGLE)

# -------------------
# DL: Embeddings (UMAP / t-SNE)
# -------------------
def plot_embeddings_dl():
    feats = DL_DIR / "embeddings_val.npy"
    if not exists(feats):
        feats = DL_DIR / "embeddings.npy"
    yfile = DL_DIR / "y_true.npy"
    if not (exists(feats) and exists(yfile)):
        return log_skip("DL embeddings (need embeddings*_ and y_true.npy)")

    X = np.load(feats, allow_pickle=False)
    y = np.load(yfile, allow_pickle=False)
    palette = sns.color_palette("Set2", len(np.unique(y)))

    # UMAP
    try:
        um = umap.UMAP(random_state=42).fit_transform(X)
        plt.figure()
        ax = plt.gca(); nice_ax(ax)
        sns.scatterplot(x=um[:,0], y=um[:,1], hue=y, palette=palette, s=14, alpha=0.8, edgecolor=None, ax=ax)
        ax.set_title("DL UMAP Embeddings"); ax.legend(title="Class", bbox_to_anchor=(1.02,1), loc="upper left")
        savefig("umap_embeddings", w=W_DOUBLE, h=H_DOUBLE)
    except Exception as e:
        log_skip(f"DL UMAP ({e})")

    # t-SNE
    try:
        ts = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)
        plt.figure()
        ax = plt.gca(); nice_ax(ax)
        sns.scatterplot(x=ts[:,0], y=ts[:,1], hue=y, palette=palette, s=14, alpha=0.8, edgecolor=None, ax=ax)
        ax.set_title("DL t-SNE Embeddings"); ax.legend(title="Class", bbox_to_anchor=(1.02,1), loc="upper left")
        savefig("tsne_embeddings", w=W_DOUBLE, h=H_DOUBLE)
    except Exception as e:
        log_skip(f"DL t-SNE ({e})")

# -------------------
# DL: Baselines + Feature importance
# -------------------
def plot_baselines_dl():
    bj = DL_DIR / "baselines.json"
    if not exists(bj):
        return
    data = safe_json_load(bj)
    if not data:
        return
    names = list(data.keys())
    metric_keys = [k for k in {"acc","bal_acc","mcc","auc","aucpr"} if any(k in data[m] for m in names)]
    if not metric_keys:
        return

    for mk in metric_keys:
        vals = [data[m].get(mk, np.nan) for m in names]
        plt.figure()
        ax = plt.gca(); nice_ax(ax)
        sns.barplot(x=names, y=vals, ax=ax)
        ax.set_ylabel(mk.upper()); ax.set_title(f"DL Baselines — {mk.upper()}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        savefig(f"baseline_{mk}", w=W_SINGLE, h=H_SINGLE)

def plot_feature_importance_dl():
    for tag in ["rf","xgb"]:
        f = DL_DIR / f"{tag}_feature_importance.npy"
        if not exists(f):
            continue
        imp = np.load(f)
        plt.figure()
        ax = plt.gca(); nice_ax(ax)
        ax.bar(np.arange(len(imp)), imp)
        ax.set_xlabel("Feature Index"); ax.set_ylabel("Importance")
        ax.set_title(f"{tag.upper()} Feature Importance")
        savefig(f"{tag}_feature_importance", w=W_SINGLE, h=H_SINGLE)

# -------------------
# ML: helpers
# -------------------
def _load_ml_folds(model_key: str):
    """Concat y_true, y_pred, y_proba across folds for a model key (rf / xgb)."""
    y_true = concat_folds(f"y_true_{model_key}_fold*.npy", ML_DIR)
    y_pred = concat_folds(f"y_pred_{model_key}_fold*.npy", ML_DIR)
    y_prob = concat_folds(f"y_proba_{model_key}_fold*.npy", ML_DIR)
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        y_prob = y_prob[:, -1]
    return y_true, y_pred, y_prob

# -------------------
# ML: Confusion + ROC/PR per model
# -------------------
def plot_confusion_and_curves_ml(model_key: str):
    y_true, y_pred, y_prob = _load_ml_folds(model_key)
    if y_true is None or y_pred is None:
        return log_skip(f"ML {model_key}: confusion/curves (no fold files)")

    # Confusion (normalized colors + count annotations)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    cm_cnt  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm_norm)
    disp.plot(cmap="Blues", values_format=".2f", colorbar=False, ax=ax)
    ax.set_title(f"ML {model_key.upper()} Normalized Confusion Matrix")
    for (i, j), v in np.ndenumerate(cm_cnt):
        ax.text(j, i, f"{v}", ha="center", va="center", color="navy", fontsize=9, fontweight="bold")
    savefig(f"ml_{model_key}_confusion_matrix", w=W_SINGLE, h=W_SINGLE)

    # ROC/PR
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob); roc_auc = auc(fpr, tpr)
        plt.figure(); ax = plt.gca(); nice_ax(ax)
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax.plot([0,1],[0,1],"--",color="gray")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ML {model_key.upper()} ROC Curve")
        ax.legend()
        savefig(f"ml_{model_key}_roc_curve", w=W_SINGLE, h=W_SINGLE)

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.figure(); ax = plt.gca(); nice_ax(ax)
        ax.plot(rec, prec, label=f"AP={ap:.2f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"ML {model_key.upper()} PR Curve")
        ax.legend()
        savefig(f"ml_{model_key}_pr_curve", w=W_SINGLE, h=W_SINGLE)
    else:
        log_skip(f"ML {model_key} ROC/PR (no probabilities)")

# -------------------
# ML: CV bars (mean ± 95% CI)
# -------------------
def plot_cv_ml():
    csv = ML_DIR / "cv_ablation_results.csv"
    if not exists(csv):
        return log_skip("ML CV barplots (missing cv_ablation_results.csv)")
    df = pd.read_csv(csv)
    if "model" not in df.columns:
        return log_skip("ML CV barplots (no 'model' column)")
    metrics = [c for c in ["bal_acc","macro_f1","auc"] if c in df.columns]
    if not metrics:
        return log_skip("ML CV barplots (no metric columns)")

    g = df.groupby("model")
    for m in metrics:
        means = g[m].mean()
        sems  = g[m].sem()
        ci95  = 1.96 * sems
        plt.figure()
        ax = plt.gca(); nice_ax(ax)
        ax.bar(means.index, means.values, yerr=ci95.values, capsize=4)
        ax.set_ylabel(m.replace("_"," ").upper())
        ax.set_title(f"ML CV — {m.replace('_',' ').title()} (mean ± 95% CI)")
        savefig(f"ml_cv_{m}", w=W_SINGLE, h=H_SINGLE)

# -------------------
# ML: Radar per model (from *_mean fields)
# -------------------
def plot_radar_ml():
    js = ML_DIR / "statistical_summary.json"
    if not exists(js):
        return log_skip("ML radar (missing statistical_summary.json)")
    data = safe_json_load(js)
    for model_key, metrics in data.items():
        labels, vals = [], []
        for k, v in metrics.items():
            if k.endswith("_mean"):
                labels.append(k.replace("_mean","").upper())
                vals.append(float(v))
        if not labels:
            continue
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        vals_c = vals + vals[:1]
        angs   = angles + angles[:1]
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        ax.plot(angs, vals_c, "o-", linewidth=2, label=model_key.upper())
        ax.fill(angs, vals_c, alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        ax.set_title(f"ML {model_key.upper()} Statistical Summary (means)")
        savefig(f"ml_radar_{model_key}", w=W_SINGLE, h=W_SINGLE)

# -------------------
# ML: Feature importances if saved as npy in DL dir
# -------------------
def plot_feature_importance_ml():
    for tag in ["rf","xgb"]:
        f = DL_DIR / f"{tag}_feature_importance.npy"
        if not exists(f):
            continue
        imp = np.load(f)
        plt.figure()
        ax = plt.gca(); nice_ax(ax)
        ax.bar(np.arange(len(imp)), imp)
        ax.set_xlabel("Feature Index"); ax.set_ylabel("Importance")
        ax.set_title(f"ML {tag.upper()} Feature Importance (from saved .npy)")
        savefig(f"ml_{tag}_feature_importance", w=W_SINGLE, h=H_SINGLE)

# -------------------
# Main
# -------------------
def main():
    print(f"[info] results dir   = {DL_DIR}")
    print(f"[info] ml_results dir= {ML_DIR}")
    print(f"[info] figures dir   = {FIG_DIR}")
    print_plan()

    # DL
    plot_training_curves_dl()
    plot_confusion_and_curves_dl()
    plot_cv_dl()
    plot_radar_dl()
    plot_embeddings_dl()
    plot_baselines_dl()
    plot_feature_importance_dl()

    # ML
    for mdl in ["rf","xgb"]:
        plot_confusion_and_curves_ml(mdl)
    plot_cv_ml()
    plot_radar_ml()
    plot_feature_importance_ml()

    print(f"✅ All figures saved in {FIG_DIR}/")

if __name__ == "__main__":
    main()
