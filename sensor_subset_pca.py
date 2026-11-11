"""
Gas Sensing - Odor Classification: Sensor-Subset PCA Analysis
==============================================================

Pipeline
--------
  1. Load sensor responses (3 replicates per sensor) and average per product
  2. Greedy sensor ranking: normalized ANOVA F-score
     - alpha x mean |Spearman rho| to the sensors already selected
  3. For N = 2..18 top-ranked sensors: standardize, PCA, compute metrics
  4. Synthetic controls: first two ranked sensors kept, sensors 3..18 replaced by
     - noise sensors     : random product-level values unrelated to class
     - redundant sensors : copies of the first two sensors
     both with replicate noise matched to the measured data
     (median and interquartile range of N_CONTROL realizations)

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
    python sensor_subset_pca.py --input_dir /path/to/csv_folder --outdir results
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif

plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)   # harmless sqrt-glyph warning with Arial

N_SENSORS, N_REP = 18, 3
ALPHA = 0.35                        # redundancy penalty in the greedy ranking
SUBSET_SIZES = range(2, N_SENSORS + 1)
SCORE_PLOT_N = [2, 4, 8, 12, 15, 18]
N_CONTROL = 200                     # realizations per synthetic control
SEED = 20261029

LABELS = ["Perfume", "Wine", "Coffee", "Softener"]
LABEL_COLORS = {"Perfume": "#d62728", "Wine": "#1f77b4", "Coffee": "#2ca02c", "Softener": "#9467bd"}
HULL_CMAPS = {"Perfume": plt.cm.Reds, "Wine": plt.cm.Blues, "Coffee": plt.cm.Greens, "Softener": plt.cm.Purples}
C_REAL, C_RED, C_NOISE = "#111111", "#d95f02", "#9a9a9a"


# =========================================
# 1. Data loading
# =========================================
def parse_label(fname):
    low = fname.lower()
    for key, label in [("coffee", "Coffee"), ("wine", "Wine"), ("perfume", "Perfume"),
                       ("softener", "Softener"), ("fabric", "Softener")]:
        if key in low:
            return label
    raise ValueError(f"Cannot infer label from filename: {fname}")


def load_dataset(folder):
    """Returns replicate data (n_products, N_REP, N_SENSORS) and labels (n_products,)."""
    X_rep, y = [], []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".csv"):
            continue
        resp = pd.to_numeric(pd.read_csv(os.path.join(folder, fn)).iloc[:, 1],
                             errors="coerce").dropna().to_numpy()
        if len(resp) != N_SENSORS * N_REP:
            raise ValueError(f"{fn}: {len(resp)} rows, expected {N_SENSORS * N_REP}")
        X_rep.append(resp.reshape(N_SENSORS, N_REP).T)
        y.append(parse_label(fn))
    return np.array(X_rep), np.array(y)


# =========================================
# 2. Greedy sensor ranking (discrimination vs. redundancy)
# =========================================
def rank_sensors(X, y, alpha=ALPHA):
    """First sensor: highest ANOVA F-score. Next: max [F_norm - alpha * mean |Spearman rho|
    to the sensors already selected]."""
    f, _ = f_classif(X, y)
    f = (f - f.min()) / (f.max() - f.min())
    corr = pd.DataFrame(X).corr(method="spearman").abs().to_numpy()
    selected = [int(np.argmax(f))]
    remaining = [j for j in range(X.shape[1]) if j != selected[0]]
    while remaining:
        scores = [f[j] - alpha * corr[j, selected].mean() for j in remaining]
        selected.append(remaining.pop(int(np.argmax(scores))))
    return selected


# =========================================
# 3. Metrics
# =========================================
def pca_scores(X):
    pca = PCA(n_components=2)
    return pca.fit_transform(StandardScaler().fit_transform(X)), pca.explained_variance_ratio_


def class_centroids(Z, y):
    return np.array([Z[y == lab].mean(axis=0) for lab in LABELS])


def metrics(X, y):
    """X: product means of the selected sensors (standardized here, per subset)."""
    Z = StandardScaler().fit_transform(X)
    P, _ = pca_scores(X)

    # mean distance between the 4 class centroids (6 pairs), PC1-PC2 space
    pc2_centroid = pdist(class_centroids(P, y)).mean()

    # same in the full N-dim space, divided by the mean pairwise
    # distance between products of the same class
    full_centroid = pdist(class_centroids(Z, y)).mean()
    full_within = np.concatenate([pdist(Z[y == lab]) for lab in LABELS]).mean()

    # entropy-based effective rank of the covariance eigenvalues
    lam = np.clip(np.linalg.eigvalsh(np.cov(Z, rowvar=False)), 0, None)
    p = lam[lam > 0] / lam.sum()

    return {"pc2_centroid_dist": pc2_centroid,
            "full_centroid_dist": full_centroid,
            "full_within_dist": full_within,
            "full_norm_ratio": full_centroid / full_within,
            "erank": float(np.exp(-(p * np.log(p)).sum()))}


# =========================================
# 4. Synthetic controls
# =========================================
def synthetic_controls(X_rep, y, order, rng):
    """First two ranked sensors are real; sensors 3..18 are synthetic."""
    X_mean = X_rep.mean(axis=1)
    prod_sd = X_mean.std(axis=0, ddof=1)
    rep_sd = np.sqrt(X_rep.var(axis=1, ddof=1).mean(axis=0))
    tau = float(np.median(rep_sd / prod_sd))           # replicate noise / between-product SD
    rep_std = (X_rep - X_mean.mean(axis=0)) / prod_sd  # standardized replicate data
    base = order[:2]
    n_prod = len(y)

    def synth(kind):
        if kind == "noise":
            r = rng.normal(size=(n_prod, 1)) + rng.normal(scale=tau, size=(n_prod, N_REP))
        else:
            r = rep_std[:, :, base[rng.integers(2)]] + rng.normal(scale=tau, size=(n_prod, N_REP))
        return r.mean(axis=1)

    rows = []
    for kind in ["noise", "redundant"]:
        for b in range(N_CONTROL):
            Xc = np.column_stack([X_mean[:, base]] + [synth(kind) for _ in range(N_SENSORS - 2)])
            for k in SUBSET_SIZES:
                rows.append({"control": kind, "rep": b, "n_sensors": k, **metrics(Xc[:, :k], y)})
    return pd.DataFrame(rows), tau


# =========================================
# 5. Plots
# =========================================
def save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.svg"), bbox_inches="tight")
    plt.close(fig)


def score_plot(X, y, k, normalize, outdir):
    P, evr = pca_scores(X)
    axis = r"/$\sqrt{\mathrm{N}}$" if normalize else ""
    if normalize:
        P = P / np.sqrt(k)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for lab in LABELS:
        pts = P[y == lab]
        hull = pts[ConvexHull(pts).vertices]
        hull = np.vstack([hull, hull[0]])
        ax.fill(hull[:, 0], hull[:, 1], color=HULL_CMAPS[lab](0.65), alpha=0.12)
        ax.plot(hull[:, 0], hull[:, 1], color=HULL_CMAPS[lab](0.65), lw=1.6)
        ax.scatter(pts[:, 0], pts[:, 1], s=64, c=LABEL_COLORS[lab], zorder=3)
    ax.text(0.04, 0.95, rf"$N$ = {k}", transform=ax.transAxes, fontsize=18, va="top")
    ax.set_xlabel(f"PC1{axis} ({evr[0] * 100:.1f}%)", fontsize=18)
    ax.set_ylabel(f"PC2{axis} ({evr[1] * 100:.1f}%)", fontsize=18)
    if normalize:
        ax.set_xlim(-2.0, 2.4); ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
    else:
        ax.set_xlim(-8, 10); ax.set_ylim(-2, 6)
    ax.tick_params(labelsize=15, width=1, length=6)
    handles = [mlines.Line2D([], [], color=LABEL_COLORS[l], marker="o", ls="None", ms=8, label=l)
               for l in LABELS]
    ax.legend(handles=handles, loc="upper right", borderaxespad=0, fontsize=15, fancybox=False,
              edgecolor="k", framealpha=1, handletextpad=0.3, labelspacing=0.3, borderpad=0.35)
    fig.tight_layout()
    tag = "PCA_score_sqrtN" if normalize else "PCA_score"
    save(fig, outdir, f"{tag}_{k:02d}sensors")


def trend_style(ax, ylabel):
    ax.set_xlabel("Number of sensors", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xticks([2, 6, 10, 14, 18])
    ax.set_xlim(1.5, 18.5)
    ax.tick_params(labelsize=15, direction="out", top=False, right=False)
    for s in ax.spines.values():
        s.set_linewidth(1.2)


def control_plot(real, ctrl, key, control, ylabel, ylim, yticks, outdir, name, unity=False):
    """Measured curve vs one synthetic control (median + IQR band)."""
    color, label = {"noise": (C_NOISE, "Noise control"),
                    "redundant": (C_RED, "Redundant control")}[control]
    q = ctrl[ctrl.control == control].groupby("n_sensors")[key].quantile([0.25, 0.5, 0.75]).unstack()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.2, lw=0)
    ax.plot(q.index, q[0.5], color=color, lw=1.6, label=label)
    ax.plot(real.n_sensors, real[key], "o-", color=C_REAL, ms=5, lw=1.6, label="Measured")
    if unity:
        ax.axhline(1, color="#777", lw=1, ls=":")
    trend_style(ax, ylabel)
    ax.set_ylim(*ylim); ax.set_yticks(yticks)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper right", borderaxespad=0, fontsize=13, fancybox=False,
              edgecolor="k", handletextpad=0.3, labelspacing=0.3, borderpad=0.35)
    save(fig, outdir, name)


# =========================================
# Main
# =========================================
def main():
    parser = argparse.ArgumentParser(description="Sensor-subset PCA analysis.")
    parser.add_argument("--input_dir", "-i", required=True, help="Folder with per-product response CSV files.")
    parser.add_argument("--outdir", "-o", default="results", help="Output folder. Default: results")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    X_rep, y = load_dataset(args.input_dir)
    X = X_rep.mean(axis=1)                              # product means (n=3)
    print("Products per class:", pd.Series(y).value_counts().to_dict())

    order = rank_sensors(X, y)
    names = [f"S{i + 1}" for i in order]
    print("Sensor ranking:", ", ".join(names))

    real = pd.DataFrame([{"n_sensors": k, "added_sensor": names[k - 1], **metrics(X[:, order[:k]], y)}
                         for k in SUBSET_SIZES])
    real.to_csv(os.path.join(args.outdir, "measured_metrics.csv"), index=False)
    print(real.round(3).to_string(index=False))

    for k in SCORE_PLOT_N:
        score_plot(X[:, order[:k]], y, k, normalize=False, outdir=args.outdir)
        score_plot(X[:, order[:k]], y, k, normalize=True, outdir=args.outdir)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(real.n_sensors, real.pc2_centroid_dist, "o--", color="#f06080", ms=5, lw=1.6)
    trend_style(ax, "Inter-class centroid distance")
    ax.set_ylim(0, 6)
    save(fig, args.outdir, "centroid_distance_PC1-PC2")

    ctrl, tau = synthetic_controls(X_rep, y, order, np.random.default_rng(SEED))
    print(f"Replicate noise / between-product SD (tau) = {tau:.3f}")
    ctrl.to_csv(os.path.join(args.outdir, "control_metrics_all_reps.csv"), index=False)

    control_plot(real, ctrl, "full_norm_ratio", "noise", "Normalized centroid /\nwithin-class distance",
                 (0, 2.5), [0, 0.5, 1, 1.5, 2, 2.5], args.outdir, "normalized_distance", unity=True)
    control_plot(real, ctrl, "erank", "redundant", "Effective rank",
                 (1, 2.5), [1, 1.5, 2, 2.5], args.outdir, "effective_rank")

    print("\nDone. Results saved to:", args.outdir)


if __name__ == "__main__":
    main()
