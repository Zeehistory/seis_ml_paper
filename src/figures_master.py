#!/usr/bin/env python3
"""
Unified plotting script for IEEE Access paper
Generates: training curves, confusion matrices, ROC/PR curves,
class distributions, feature histograms, embeddings, etc.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import label_binarize

FIG_DIR = "seis-paper/figures"
RES_CSV = "seis-paper/results/cv_ablation_results.csv"
os.makedirs(FIG_DIR, exist_ok=True)

# ========== 1. TRAINING CURVES ==========
def plot_training_curves():
    try:
        loss = np.load("seis-paper/results/train_loss.npy")
        val_acc = np.load("seis-paper/results/val_acc.npy")
        lr = np.load("seis-paper/results/lr_schedule.npy")
    except FileNotFoundError:
        print("⚠️ Training logs not found (train_loss.npy, val_acc.npy, lr_schedule.npy)")
        return

    # Loss curve
    plt.figure()
    plt.plot(loss, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.savefig(f"{FIG_DIR}/training_loss.png", dpi=300)

    # Accuracy curve
    plt.figure()
    plt.plot(val_acc, label="Val Accuracy/MCC")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/MCC")
    plt.legend()
    plt.title("Validation Accuracy/MCC")
    plt.savefig(f"{FIG_DIR}/training_val_acc.png", dpi=300)

    # LR schedule
    plt.figure()
    plt.plot(lr, label="Learning Rate")
    plt.xlabel("Iteration")
    plt.ylabel("LR")
    plt.title("OneCycleLR Schedule")
    plt.legend()
    plt.savefig(f"{FIG_DIR}/lr_schedule.png", dpi=300)


# ========== 2. CONFUSION MATRIX ==========
def plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=300)


# ========== 3. ROC & PR CURVES ==========
def plot_roc_pr(y_true, y_proba, class_names):
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    plt.figure(figsize=(10,5))

    # ROC curves
    plt.subplot(1,2,1)
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_proba[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves")
    plt.legend()

    # PR curves
    plt.subplot(1,2,2)
    for i, cls in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_bin[:,i], y_proba[:,i])
        ap = average_precision_score(y_bin[:,i], y_proba[:,i])
        plt.plot(rec, prec, label=f"{cls} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/roc_pr_curves.png", dpi=300)


# ========== 4. CLASS DISTRIBUTIONS ==========
def plot_class_distribution(y, class_names):
    sns.countplot(x=y)
    plt.xticks(ticks=range(len(class_names)), labels=class_names)
    plt.title("Class Distribution")
    plt.savefig(f"{FIG_DIR}/class_distribution.png", dpi=300)


# ========== 5. FEATURE HISTOGRAMS ==========
def plot_feature_histograms(X, feature_names):
    df = pd.DataFrame(X, columns=feature_names)
    df.hist(bins=40, figsize=(15,10))
    plt.suptitle("Feature Distributions")
    plt.savefig(f"{FIG_DIR}/feature_histograms.png", dpi=300)


# ========== 6. T-SNE / UMAP EMBEDDINGS ==========
def plot_embeddings(features, labels, class_names):
    tsne = TSNE(n_components=2, random_state=42).fit_transform(features)
    reducer = umap.UMAP(random_state=42).fit_transform(features)

    for emb, name in [(tsne, "tsne"), (reducer, "umap")]:
        plt.figure()
        sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=labels, palette="tab10", legend="full")
        plt.title(f"{name.upper()} Embedding")
        plt.savefig(f"{FIG_DIR}/{name}_embedding.png", dpi=300)


if __name__ == "__main__":
    # Run training curve plots
    plot_training_curves()

    # Example usage placeholders
    # Replace with real arrays from your pipeline
    y_true = np.load("seis-paper/results/y_true.npy") if os.path.exists("seis-paper/results/y_true.npy") else None
    y_pred = np.load("seis-paper/results/y_pred.npy") if os.path.exists("seis-paper/results/y_pred.npy") else None
    y_proba = np.load("seis-paper/results/y_proba.npy") if os.path.exists("seis-paper/results/y_proba.npy") else None

    class_names = ["muon","pion","kaon","electron"]

    if y_true is not None and y_pred is not None:
        plot_confusion(y_true, y_pred, class_names)
    if y_true is not None and y_proba is not None:
        plot_roc_pr(y_true, y_proba, class_names)
        plot_class_distribution(y_true, class_names)

    # If you have features saved from the model
    if os.path.exists("seis-paper/results/hidden_features.npy"):
        features = np.load("seis-paper/results/hidden_features.npy")
        plot_embeddings(features, y_true, class_names)
