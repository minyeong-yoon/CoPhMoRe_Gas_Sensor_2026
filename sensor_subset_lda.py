"""
Gas Sensing - Odor Classification: LDA Analysis
================================================

Pipeline
--------
  1. Load sensor responses and average the 3 replicates per sensor (18-dim vector per product)
  2. StandardScaler + shrinkage LDA (lsqr, shrinkage="auto"), 5-fold GroupKFold split by product
  3. 2D LDA projection; decision boundaries from an additional LDA fitted to LD1-LD2
  4. L2 normalization of each response vector, then the same 2D LDA projection
  5. Cross-validated accuracy for every sensor combination at N = 1, 3, 6, 9, 12, 15, 18

Notes on library versions
-------------------------
  - The GroupKFold split of the 20 products depends on the scikit-learn / numpy version,
    so cross-validated accuracies can differ slightly between environments.
  - The absolute scale (and sign) of the LD scores also depends on the scikit-learn version;
    trace(Sb) and trace(Sw) scale together, so the Fisher discriminant ratio does not change.

Input data format
-----------------
One CSV file per product, named like:
    "<Label> <id>_<analyte>_<date> <rep>_response.csv"
    e.g. "Coffee 1_Ethiopia_260128 S2_response.csv"
Label is inferred from the filename (coffee / wine / perfume / softener / fabric).
CSV: header row; column 2 = response values, 3 consecutive replicates per sensor
(18 sensors x 3 = 54 rows).

Usage
-----
    python sensor_subset_lda.py --input_dir /path/to/csv_folder --outdir results
"""

import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from joblib import Parallel, delayed

from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"

N_SENSORS, N_REP = 18, 3
N_SPLITS = 5
SUBSET_SIZES = [1, 3, 6, 9, 12, 15, 18]
LABELS = ["Perfume", "Wine", "Coffee", "Softener"]
LABEL_COLORS = {"Perfume": "#C3292F", "Wine": "#1f77b4", "Coffee": "#2ca02c", "Softener": "#7b49a8"}
SENSOR_NAMES = [f"S{i + 1}" for i in range(N_SENSORS)]


# =========================================
# 1. Data loading
# =========================================
def parse_label_group(fname):
    """Label from the filename; group = product (analyte) identity."""
    stem = os.path.splitext(fname)[0]
    low = stem.lower()
    for key, label in [("coffee", "Coffee"), ("wine", "Wine"), ("perfume", "Perfume"),
                       ("softener", "Softener"), ("fabric", "Softener")]:
        if key in low:
            parts = stem.split("_")
            return label, f"{label}__{parts[1].strip() if len(parts) >= 2 else stem}"
    raise ValueError(f"Cannot infer label from filename: {fname}")


def load_dataset(folder):
    """Returns replicate-averaged responses X (n_products, 18), labels and groups."""
    X, y, groups = [], [], []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".csv"):
            continue
        resp = pd.to_numeric(pd.read_csv(os.path.join(folder, fn)).iloc[:, 1],
                             errors="coerce").dropna().to_numpy()
        if len(resp) != N_SENSORS * N_REP:
            raise ValueError(f"{fn}: {len(resp)} rows, expected {N_SENSORS * N_REP}")
        X.append(resp.reshape(N_SENSORS, N_REP).mean(axis=1))
        label, group = parse_label_group(fn)
        y.append(label)
        groups.append(group)
    return np.array(X), np.array(y), np.array(groups)


def lda_pipeline():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def heatmap(ax, df, cmap, vmin, vmax, annot=False):
    im = ax.imshow(df.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns)
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.tick_params(length=0)
    if annot:
        for (i, j), v in np.ndenumerate(df.to_numpy()):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v > 60 else "black")
    plt.colorbar(im, ax=ax, pad=0.02)


def save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.svg"), bbox_inches="tight")
    plt.close(fig)


# =========================================
# 2. Cross-validated classification
# =========================================
def cross_validated_classification(X, y, groups, outdir):
    y_pred = cross_val_predict(lda_pipeline(), X, y, groups=groups, cv=GroupKFold(n_splits=N_SPLITS))
    print(f"\n5-fold CV accuracy: {accuracy_score(y, y_pred):.3f}")

    cm = confusion_matrix(y, y_pred, labels=LABELS)
    cm_pct = pd.DataFrame(cm / cm.sum(axis=1, keepdims=True) * 100, index=LABELS, columns=LABELS)
    cm_pct.to_csv(os.path.join(outdir, "confusion_matrix_percent.csv"))
    print("Confusion matrix (%, rows = true):\n", cm_pct.round(1))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    heatmap(ax, cm_pct, "Blues", 0, 100, annot=True)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion matrix", fontsize=14)
    save(fig, outdir, "confusion_matrix")

    report = pd.DataFrame(classification_report(y, y_pred, labels=LABELS, output_dict=True)).T.loc[LABELS]
    report.to_csv(os.path.join(outdir, "classwise_metrics.csv"))
    print("Class-wise metrics:\n", report.round(3))

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(LABELS))
    for i, m in enumerate(["precision", "recall", "f1-score"]):
        ax.bar(x + (i - 1) * 0.25, report[m], width=0.25, label=m.capitalize())
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.2)
    ax.legend(ncol=3, loc="upper left")
    save(fig, outdir, "classwise_metrics")


# =========================================
# 3. LDA coefficient heatmap
# =========================================
def coefficient_heatmap(X, y, outdir):
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(StandardScaler().fit_transform(X), y)
    coef = pd.DataFrame(lda.coef_, index=lda.classes_,
                        columns=[f"R{i + 1}" for i in range(N_SENSORS)]).loc[LABELS]
    coef.to_csv(os.path.join(outdir, "lda_coefficients.csv"))

    fig, ax = plt.subplots(figsize=(12, 4))
    heatmap(ax, coef, "RdBu_r", -10, 10)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    ax.set_title("LDA coefficient heatmap", fontsize=14)
    save(fig, outdir, "lda_coefficient_heatmap")


# =========================================
# 4. LDA score plots + Fisher discriminant ratio
# =========================================
def fisher_ratio(Z, y):
    """trace(Sb) / trace(Sw) of the LD1-LD2 scores."""
    mu = Z.mean(axis=0)
    tr_b = sum((y == c).sum() * np.sum((Z[y == c].mean(axis=0) - mu) ** 2) for c in LABELS)
    tr_w = sum(np.sum((Z[y == c] - Z[y == c].mean(axis=0)) ** 2) for c in LABELS)
    return tr_b, tr_w, tr_b / tr_w


def lda_score_plot(X, y, outdir, name, title, grid_res=500, pad=10):
    lda = LinearDiscriminantAnalysis(n_components=2)
    Z = lda.fit_transform(StandardScaler().fit_transform(X), y)
    evr = lda.explained_variance_ratio_

    # decision boundaries: additional LDA classifier fitted to LD1-LD2 coordinates
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Z, y)
    xx, yy = np.meshgrid(np.linspace(Z[:, 0].min() - pad, Z[:, 0].max() + pad, grid_res),
                         np.linspace(Z[:, 1].min() - pad, Z[:, 1].max() + pad, grid_res))
    region = pd.Series(clf.predict(np.c_[xx.ravel(), yy.ravel()])).map(
        {c: i for i, c in enumerate(LABELS)}).to_numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, region, levels=np.arange(len(LABELS) + 1) - 0.5,
                cmap=ListedColormap([LABEL_COLORS[c] for c in LABELS]), alpha=0.2)
    ax.contour(xx, yy, region, levels=np.arange(len(LABELS) - 1) + 0.5, colors="white", linewidths=1.5)
    for c in LABELS:
        ax.scatter(Z[y == c, 0], Z[y == c, 1], s=60, color=LABEL_COLORS[c], label=c, zorder=3)
    ax.set_xlabel(f"LD1 ({evr[0] * 100:.1f}%)", fontsize=14)
    ax.set_ylabel(f"LD2 ({evr[1] * 100:.1f}%)", fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.legend(fancybox=False, edgecolor="k")
    save(fig, outdir, name)

    pd.DataFrame({"label": y, "LD1": Z[:, 0], "LD2": Z[:, 1]}).to_csv(
        os.path.join(outdir, f"{name}.csv"), index=False)
    return fisher_ratio(Z, y)


# =========================================
# 5. Accuracy vs number of sensors
# =========================================
def subset_accuracy(X, y, groups, outdir):
    cv = GroupKFold(n_splits=N_SPLITS)

    def score(idx):
        return cross_val_score(lda_pipeline(), X[:, idx], y, groups=groups, cv=cv).mean()

    rows = []
    for k in SUBSET_SIZES:
        combos = list(combinations(range(N_SENSORS), k))
        print(f"N = {k}: {len(combos):,} combinations")
        accs = Parallel(n_jobs=-1, batch_size=256)(delayed(score)(list(c)) for c in combos)
        rows += [{"n_sensors": k, "sensors": ", ".join(SENSOR_NAMES[j] for j in c), "accuracy": a}
                 for c, a in zip(combos, accs)]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, "accuracy_all_combinations.csv"), index=False)

    summary = res.groupby("n_sensors")["accuracy"].agg(n_combinations="size", mean="mean", std="std")
    summary.to_csv(os.path.join(outdir, "accuracy_vs_n_sensors.csv"))
    print(summary.round(3))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(summary.index, summary["mean"], yerr=summary["std"], fmt="o--", capsize=4,
                label=r"Mean over $_{18}C_N$ combinations")
    ax.set_xlabel("Number of sensors", fontsize=12)
    ax.set_ylabel("5-fold cross-validation accuracy", fontsize=12)
    ax.set_xticks(SUBSET_SIZES)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    save(fig, outdir, "accuracy_vs_n_sensors")


# =========================================
# Main
# =========================================
def main():
    parser = argparse.ArgumentParser(description="LDA classification analysis.")
    parser.add_argument("--input_dir", "-i", required=True, help="Folder with per-product response CSV files.")
    parser.add_argument("--outdir", "-o", default="results", help="Output folder. Default: results")
    parser.add_argument("--skip_subsets", action="store_true",
                        help="Skip the sensor-combination sweep, which takes the longest.")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    X, y, groups = load_dataset(args.input_dir)
    print("Products per class:", pd.Series(y).value_counts().to_dict())

    cross_validated_classification(X, y, groups, args.outdir)
    coefficient_heatmap(X, y, args.outdir)

    fdr = {"raw": lda_score_plot(X, y, args.outdir, "lda_scores", "LDA scores"),
           "L2_normalized": lda_score_plot(Normalizer(norm="l2").fit_transform(X), y, args.outdir,
                                           "lda_scores_L2-normalized", "LDA scores (L2-normalized)")}
    fdr = pd.DataFrame(fdr, index=["trace_Sb", "trace_Sw", "FDR"]).T
    fdr.to_csv(os.path.join(args.outdir, "fisher_discriminant_ratio.csv"))
    print("\nFisher discriminant ratio (LD1-LD2):\n", fdr.round(2))

    if not args.skip_subsets:
        subset_accuracy(X, y, groups, args.outdir)

    print("\nDone. Results saved to:", args.outdir)


if __name__ == "__main__":
    main()
