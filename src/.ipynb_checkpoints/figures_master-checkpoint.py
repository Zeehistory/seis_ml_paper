#!/usr/bin/env python3
"""
figures_all.py
Generates *all* figures for IEEE Access submission.
Combines metrics suite + training curves + contextual figures.
Consumes results/ (npy, csv, json) and saves to figures/.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
import umap

# -------------------
# Config
# -------------------
RESULTS_DIR = Path("seis-paper/results")
FIG_DIR     = Path("seis-paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = sns.color_palette("tab10")

def savefig(name):
    for ext in ["png","pdf"]:
        plt.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close()

# =======================================================
# 1. Training curves
# =======================================================
def plot_training_curves():
    try:
        train_loss = np.load(RESULTS_DIR/"train_loss.npy")
        val_loss   = np.load(RESULTS_DIR/"val_loss.npy")
        val_acc    = np.load(RESULTS_DIR/"val_acc.npy")
        val_mcc    = np.load(RESULTS_DIR/"val_mcc.npy")
        lr_sched   = np.load(RESULTS_DIR/"lr_schedule.npy")
    except FileNotFoundError:
        print("⚠️ Missing training logs")
        return

    epochs = np.arange(1, len(train_loss)+1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training & Validation Loss")
    plt.legend(); savefig("training_loss")

    plt.figure()
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.plot(epochs, val_mcc, label="Val MCC")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Validation Accuracy & MCC")
    plt.legend(); savefig("training_metrics")

    plt.figure()
    plt.plot(lr_sched)
    plt.xlabel("Step"); plt.ylabel("Learning Rate")
    plt.title("OneCycleLR Schedule")
    savefig("lr_schedule")

# =======================================================
# 2. Confusion Matrix
# =======================================================
def plot_confusion():
    y_true = np.load(RESULTS_DIR/"y_true.npy")
    y_pred = np.load(RESULTS_DIR/"y_pred.npy")
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format=".2f", colorbar=False)
    plt.title("Normalized Confusion Matrix")
    savefig("confusion_matrix")

# =======================================================
# 3. ROC & PR curves
# =======================================================
def plot_roc_pr():
    y_true = np.load(RESULTS_DIR/"y_true.npy")
    y_proba = np.load(RESULTS_DIR/"y_proba.npy")
    n_classes = y_proba.shape[1] if y_proba.ndim>1 else 2
    classes = [f"Class {i}" for i in range(n_classes)]

    # ROC
    plt.figure()
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true==i, y_proba[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curves"); plt.legend()
    savefig("roc_curves")

    # PR
    plt.figure()
    for i, cls in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y_true==i, y_proba[:,i])
        ap = average_precision_score(y_true==i, y_proba[:,i])
        plt.plot(rec, prec, label=f"{cls} (AP={ap:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curves"); plt.legend()
    savefig("pr_curves")

# =======================================================
# 4. CV performance & Ablation
# =======================================================
def plot_cv_results():
    df = pd.read_csv(RESULTS_DIR/"cv_ablation_results.csv")
    metrics = ["val_acc","bal_acc","mcc","auc","aucpr"]

    # Boxplots
    for m in metrics:
        plt.figure()
        sns.boxplot(x="hidden_dim", y=m, hue="dropout", data=df, palette="Set2")
        plt.title(f"Cross-Validation {m.upper()}")
        savefig(f"boxplot_{m}")

    # Violin plots
    for m in metrics:
        plt.figure()
        sns.violinplot(x="hidden_dim", y=m, hue="dropout", data=df, palette="Set2", split=True)
        plt.title(f"Violin Plot {m.upper()}")
        savefig(f"violin_{m}")

    # Bar charts (by dropout)
    for m in metrics:
        plt.figure()
        sns.barplot(x="dropout", y=m, data=df, palette="muted")
        plt.title(f"Dropout vs {m.upper()}")
        savefig(f"bar_dropout_{m}")

    # Line plots (hidden_dim)
    for m in metrics:
        plt.figure()
        sns.lineplot(x="hidden_dim", y=m, hue="dropout", data=df,
                     marker="o", palette="Set2")
        plt.title(f"Hidden Dim vs {m.upper()}")
        savefig(f"line_hidden_dim_{m}")

    # Heatmaps
    corr = df[metrics].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Metrics")
    savefig("correlation_heatmap")

# =======================================================
# 5. Baselines
# =======================================================
def plot_baselines():
    with open(RESULTS_DIR/"baselines.json") as f:
        baselines = json.load(f)
    names = list(baselines.keys())
    accs  = [baselines[m]["acc"] for m in names]
    plt.figure()
    sns.barplot(x=names, y=accs, palette="muted")
    plt.ylabel("Accuracy"); plt.title("Baseline Comparisons")
    savefig("baseline_comparison")

# =======================================================
# 6. Class distribution
# =======================================================
def plot_class_distribution():
    with open(RESULTS_DIR/"class_distribution.json") as f:
        class_counts = json.load(f)
    plt.figure()
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette="pastel")
    plt.title("Class Distribution"); plt.ylabel("Count")
    savefig("class_distribution")

# =======================================================
# 7. Feature histograms & correlation
# =======================================================
def plot_feature_histograms():
    FEATURES_CSV = RESULTS_DIR/"features.csv"
    if not FEATURES_CSV.exists():
        print("⚠️ No features.csv found")
        return
    df = pd.read_csv(FEATURES_CSV)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df[numeric_cols].hist(bins=30, figsize=(16,12))
    plt.suptitle("Feature Distributions")
    savefig("feature_histograms")

    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    savefig("feature_correlation")

# =======================================================
# 8. Embeddings (t-SNE, UMAP)
# =======================================================
def plot_embeddings():
    feats = np.load(RESULTS_DIR/"embeddings_val.npy")
    y = np.load(RESULTS_DIR/"y_true.npy")
    tsne = TSNE(n_components=2, random_state=42).fit_transform(feats)
    reducer = umap.UMAP(random_state=42).fit_transform(feats)

    for emb, name in [(tsne,"tsne"), (reducer,"umap")]:
        plt.figure()
        sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=y, palette=PALETTE, s=10, alpha=0.8)
        plt.title(f"{name.upper()} Embedding")
        plt.legend(title="Class", bbox_to_anchor=(1.05,1), loc="upper left")
        savefig(f"{name}_embedding")

# =======================================================
# 9. Radar chart
# =======================================================
def plot_radar():
    with open(RESULTS_DIR/"statistical_summary.json") as f:
        stats = json.load(f)
    labels = list(stats.keys())
    values = [stats[m]["mean"] for m in labels]
    values += values[:1]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6,6), dpi=300)
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, "o-", linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    plt.title("Radar Chart of Metrics")
    savefig("radar_metrics")

# =======================================================
# Run all
# =======================================================
if __name__ == "__main__":
    plot_training_curves()
    plot_confusion()
    plot_roc_pr()
    plot_cv_results()
    plot_baselines()
    plot_class_distribution()
    plot_feature_histograms()
    plot_embeddings()
    plot_radar()
    print(f"✅ All figures saved in {FIG_DIR}/")
