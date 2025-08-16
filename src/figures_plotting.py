#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# -----------------------
# CONFIG
# -----------------------
INPUT_CSV = "cv_ablation_results.csv"
OUTDIR = "seis-paper/figures"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv(INPUT_CSV)

# Parse 'report' column if it exists and contains dict-like strings
if "report" in df.columns:
    def parse_report(r):
        try:
            return ast.literal_eval(r)
        except Exception:
            return {}
    reports = df["report"].apply(parse_report)
    # Example: sklearn classification_report dict
    if len(reports) > 0 and isinstance(reports.iloc[0], dict):
        for i, rep in enumerate(reports):
            for label, metrics in rep.items():
                if isinstance(metrics, dict):
                    for m, v in metrics.items():
                        df.loc[i, f"{label}_{m}"] = v

# Metrics of interest
metric_cols = ["val_acc", "bal_acc", "mcc", "auc", "aucpr"]
extra_metrics = [c for c in df.columns if any(x in c for x in ["precision","recall","f1"])]
metric_cols += extra_metrics

# -----------------------
# 1. Boxplots
# -----------------------
for m in metric_cols:
    plt.figure()
    sns.boxplot(y=df[m])
    plt.title(f"Distribution of {m}")
    plt.savefig(f"{OUTDIR}/boxplot_{m}.png", dpi=300)
    plt.close()

# -----------------------
# 2. Violin plots
# -----------------------
for m in metric_cols:
    plt.figure()
    sns.violinplot(y=df[m])
    plt.title(f"Violin Plot of {m}")
    plt.savefig(f"{OUTDIR}/violin_{m}.png", dpi=300)
    plt.close()

# -----------------------
# 3. Heatmaps: dropout × hidden_dim vs performance
# -----------------------
for m in metric_cols:
    pivot = df.pivot_table(index="dropout", columns="hidden_dim", values=m, aggfunc=np.mean)
    plt.figure()
    sns.heatmap(pivot, annot=True, cmap="viridis")
    plt.title(f"{m} vs hidden_dim × dropout")
    plt.savefig(f"{OUTDIR}/heatmap_{m}.png", dpi=300)
    plt.close()

# -----------------------
# 4. Line plots: metric vs hidden_dim
# -----------------------
for m in metric_cols:
    plt.figure()
    sns.lineplot(x="hidden_dim", y=m, hue="dropout", data=df, marker="o")
    plt.title(f"{m} vs hidden_dim")
    plt.savefig(f"{OUTDIR}/line_hidden_dim_{m}.png", dpi=300)
    plt.close()

# -----------------------
# 5. Bar plots: average performance across dropout
# -----------------------
for m in metric_cols:
    plt.figure()
    sns.barplot(x="dropout", y=m, data=df, ci="sd")
    plt.title(f"{m} vs dropout")
    plt.savefig(f"{OUTDIR}/bar_dropout_{m}.png", dpi=300)
    plt.close()

# -----------------------
# 6. Pair plots (metrics correlation scatter)
# -----------------------
sns.pairplot(df[metric_cols].dropna())
plt.savefig(f"{OUTDIR}/pairplot_metrics.png", dpi=300)
plt.close()

# -----------------------
# 7. Correlation heatmap
# -----------------------
plt.figure(figsize=(8,6))
sns.heatmap(df[metric_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Metric Correlation Heatmap")
plt.savefig(f"{OUTDIR}/correlation_heatmap.png", dpi=300)
plt.close()

# -----------------------
# 8 & 9. ROC + PR curves (if aggregated available in report)
# -----------------------
# If "macro avg" from sklearn report exists
if any("macro avg" in c for c in df.columns):
    f1s = df[[c for c in df.columns if "f1-score" in c and "macro" in c]].mean(axis=0)
    plt.figure()
    f1s.plot(kind="bar")
    plt.title("Macro F1-scores from Reports")
    plt.savefig(f"{OUTDIR}/macro_f1_scores.png", dpi=300)
    plt.close()

# -----------------------
# 10. Confusion Matrix (if available)
# -----------------------
if "confusion_matrix" in df.columns:
    # Placeholder: parse stored confusion matrices if provided
    pass

# -----------------------
# 11. F1-score vs hyperparams
# -----------------------
f1_cols = [c for c in df.columns if "f1-score" in c]
for f1 in f1_cols:
    plt.figure()
    sns.lineplot(x="hidden_dim", y=f1, hue="dropout", data=df, marker="o")
    plt.title(f"{f1} vs hidden_dim/dropout")
    plt.savefig(f"{OUTDIR}/line_{f1}.png", dpi=300)
    plt.close()

# -----------------------
# 12. MCC vs hyperparams
# -----------------------
plt.figure()
sns.lineplot(x="hidden_dim", y="mcc", hue="dropout", data=df, marker="o")
plt.title("MCC vs hidden_dim/dropout")
plt.savefig(f"{OUTDIR}/line_mcc.png", dpi=300)
plt.close()

# -----------------------
# 13. Radar plot (multi-metric comparison)
# -----------------------
from math import pi
metrics = ["val_acc","bal_acc","mcc","auc","aucpr"]
means = [df[m].mean() for m in metrics]

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
means += means[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.plot(angles, means, "o-", linewidth=2, label="Average Model")
ax.fill(angles, means, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
plt.title("Radar Plot of Average Metrics")
plt.legend()
plt.savefig(f"{OUTDIR}/radar_metrics.png", dpi=300)
plt.close()

print(f"✅ All plots saved in {OUTDIR}")
