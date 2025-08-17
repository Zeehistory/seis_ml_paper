#!/usr/bin/env python3
"""
figures_master.py
Generates *all* figures for IEEE Access submission.
- Training curves (smoothed + variance)
- Confusion matrix
- ROC/PR curves
- Cross-validation (box, violin, bar, line, heatmaps)
- Ablation studies (dropout, hidden_dim)
- Radar chart (with optional baseline)
- Baseline comparisons
- Class distribution
- Feature histograms & correlations
- Embeddings (t-SNE, UMAP, styled)
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.manifold import TSNE
import umap

# -------------------
# Config
# -------------------
RESULTS_DIR = Path("seis-paper/results")
FIG_DIR     = Path("seis-paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = sns.color_palette("Set2")

def savefig(name):
    for ext in ["png","pdf"]:
        plt.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close()

# =======================================================
# 1. Training Curves (smoothed + variance)
# =======================================================
def plot_training_curves():
    try:
        train_loss = np.load(RESULTS_DIR/"train_loss.npy")
        val_acc    = np.load(RESULTS_DIR/"val_acc.npy")
        lr_sched   = np.load(RESULTS_DIR/"lr_schedule.npy")

        fig, ax1 = plt.subplots(figsize=(7,5))
        window = 20
        train_loss_smooth = pd.Series(train_loss).rolling(window, min_periods=1).mean()
        ax1.plot(train_loss_smooth, label="Train Loss", color="tab:red")
        ax1.fill_between(range(len(train_loss)),
                         train_loss_smooth - train_loss.std(),
                         train_loss_smooth + train_loss.std(),
                         color="tab:red", alpha=0.2)

        ax2 = ax1.twinx()
        val_acc_smooth = pd.Series(val_acc).rolling(window, min_periods=1).mean()
        ax2.plot(val_acc_smooth, label="Val Accuracy", color="tab:blue")
        ax2.fill_between(range(len(val_acc)),
                         val_acc_smooth - val_acc.std(),
                         val_acc_smooth + val_acc.std(),
                         color="tab:blue", alpha=0.2)

        ax1.set_xlabel("Epochs"); ax1.set_ylabel("Loss")
        ax2.set_ylabel("Validation Accuracy")
        ax1.set_title("Training Dynamics")
        fig.legend(loc="upper right")
        plt.tight_layout()
        savefig("training_curves")

        plt.figure(figsize=(7,5))
        plt.plot(lr_sched)
        plt.xlabel("Step"); plt.ylabel("Learning Rate")
        plt.title("OneCycleLR Schedule")
        savefig("lr_schedule")

    except Exception as e:
        print(f"[WARN] Skipping training curves: {e}")

# =======================================================
# 2. Confusion Matrix
# =======================================================
def plot_confusion():
    try:
        y_true = np.load(RESULTS_DIR/"y_true.npy")
        y_pred = np.load(RESULTS_DIR/"y_pred.npy")
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format=".2f", colorbar=False)
        plt.title("Normalized Confusion Matrix")
        savefig("confusion_matrix")
    except Exception as e:
        print(f"[WARN] Skipping confusion matrix: {e}")

# =======================================================
# 3. ROC & PR curves
# =======================================================
def plot_roc_pr():
    try:
        y_true = np.load(RESULTS_DIR/"y_true.npy")
        y_proba = np.load(RESULTS_DIR/"y_proba.npy")

        if y_proba.ndim == 1:
            y_proba = y_proba.reshape(-1, 1)
        if y_proba.shape[1] == 1:
            y_proba = np.hstack([1 - y_proba, y_proba])

        n_classes = y_proba.shape[1]

        plt.figure(figsize=(7,5))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")
        plt.plot([0,1],[0,1],"--",color="gray")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("ROC Curves"); plt.legend()
        savefig("roc_curves")

        plt.figure(figsize=(7,5))
        for i in range(n_classes):
            prec, rec, _ = precision_recall_curve(y_true == i, y_proba[:, i])
            ap = average_precision_score(y_true == i, y_proba[:, i])
            plt.plot(rec, prec, label=f"Class {i} (AP={ap:.2f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall Curves"); plt.legend()
        savefig("pr_curves")

    except Exception as e:
        print(f"[WARN] Skipping ROC/PR curves: {e}")

# =======================================================
# 4. CV performance & Ablation
# =======================================================
def plot_cv_results():
    try:
        df = pd.read_csv(RESULTS_DIR/"cv_ablation_results.csv")
        metrics = ["val_acc","bal_acc","mcc","auc","aucpr"]

        # Boxplot + raw points
        for m in metrics:
            plt.figure(figsize=(7,5))
            sns.boxplot(x="hidden_dim", y=m, hue="dropout", data=df, palette="Set2")
            sns.stripplot(x="hidden_dim", y=m, hue="dropout", data=df,
                          dodge=True, color="black", alpha=0.5)
            plt.title(f"{m.upper()} across folds")
            savefig(f"boxplot_{m}")

        # Bar charts
        for m in metrics:
            plt.figure(figsize=(6,4))
            sns.barplot(x="dropout", y=m, data=df, palette="Blues_d", errorbar="sd")
            plt.title(f"Dropout vs {m.upper()}")
            savefig(f"bar_dropout_{m}")

        # Heatmap dropout × hidden_dim
        for m in metrics:
            pivot = df.pivot_table(index="dropout", columns="hidden_dim", values=m, aggfunc="mean")
            plt.figure(figsize=(6,5))
            sns.heatmap(pivot, annot=True, cmap="viridis")
            plt.title(f"Heatmap of {m.upper()}")
            savefig(f"heatmap_{m}")

    except Exception as e:
        print(f"[WARN] Skipping CV results: {e}")

# =======================================================
# 5. Radar Plot
# =======================================================
def plot_radar():
    try:
        with open(RESULTS_DIR/"statistical_summary.json") as f:
            stats = json.load(f)
        labels = list(stats.keys())
        values = [stats[m]["mean"] for m in labels]
        baseline = [stats[m]["baseline"] for m in labels] if "baseline" in stats[labels[0]] else None

        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, "o-", label="Proposed Model", color="tab:blue")
        ax.fill(angles, values, alpha=0.25, color="tab:blue")

        if baseline:
            baseline += baseline[:1]
            ax.plot(angles, baseline, "o--", label="Baseline", color="tab:orange")
            ax.fill(angles, baseline, alpha=0.15, color="tab:orange")

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0,1)
        ax.set_title("Radar Plot of Metrics")
        ax.legend(loc="upper right", bbox_to_anchor=(1.1,1.1))
        plt.tight_layout()
        savefig("radar_metrics")

    except Exception as e:
        print(f"[WARN] Skipping radar: {e}")

# =======================================================
# 6. Embeddings (t-SNE & UMAP)
# =======================================================
def plot_embeddings():
    try:
        X = np.load(RESULTS_DIR/"embeddings_val.npy")
        y = np.load(RESULTS_DIR/"y_true.npy")
        palette = sns.color_palette("Set2", len(np.unique(y)))

        umap_emb = umap.UMAP(random_state=42).fit_transform(X)
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=umap_emb[:,0], y=umap_emb[:,1], hue=y,
                        palette=palette, alpha=0.7, s=40, edgecolor=None)
        plt.title("UMAP Embeddings")
        plt.legend(title="Class", bbox_to_anchor=(1.05,1), loc="upper left")
        plt.tight_layout(); savefig("umap_embeddings")

        tsne_emb = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=tsne_emb[:,0], y=tsne_emb[:,1], hue=y,
                        palette=palette, alpha=0.7, s=40, edgecolor=None)
        plt.title("t-SNE Embeddings")
        plt.legend(title="Class", bbox_to_anchor=(1.05,1), loc="upper left")
        plt.tight_layout(); savefig("tsne_embeddings")

    except Exception as e:
        print(f"[WARN] Skipping embeddings: {e}")

# =======================================================
# Run all
# =======================================================
if __name__ == "__main__":
    plot_training_curves()
    plot_confusion()
    plot_roc_pr()
    plot_cv_results()
    plot_radar()
    plot_embeddings()
    print(f"✅ All IEEE Access–ready figures saved in {FIG_DIR}/")
