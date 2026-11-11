"""
Gas Sensing - Cosine Similarity / Distance Analysis
====================================================

Loads a gas-response matrix (gas x sensor) from an Excel file and produces:
  1. Raw response heatmap
  2. Max response per sensor (sensor dominance check)
  3. Cosine distance matrix (heatmap + CSV)
  4. Cosine similarity matrix (heatmap + CSV)
  5. Hierarchical clustering dendrogram based on cosine distance

Usage
-----
    python gas_sensing_cosine_dendrogram.py --input /path/to/8_gases_response_n18_median.xlsx --outdir results

If --outdir is omitted, results are saved next to the input file.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity


# =========================================
# Config
# =========================================
HEATMAP_CMAP = "coolwarm"
LINKAGE_METHOD = "average"
METRIC = "cosine"

# Gas name -> LaTeX-style label for plotting
GAS_LABELS = {
    "H2": r"H$_2$",
    "CH4": r"CH$_4$",
    "NH3": r"NH$_3$",
    "CO2": r"CO$_2$",
    "Ethanol": r"C$_2$H$_5$OH",
    "Acetone": r"C$_3$H$_6$O",
    "Toluene": r"C$_7$H$_8$",
    "NO2": r"NO$_2$",
}


# =========================================
# Data loading
# =========================================
def load_data(file_path: str) -> pd.DataFrame:
    """Load gas x sensor response matrix from an Excel file."""
    df = pd.read_excel(file_path)

    sensor_col = df.columns[0]
    df = df.set_index(sensor_col)
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def format_gas_names(gas_names):
    return [GAS_LABELS.get(g, g) for g in gas_names]


# =========================================
# Plots
# =========================================
def plot_raw_heatmap(X, gas_names, sensor_names, save_dir):
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(X.values, aspect="auto", cmap=HEATMAP_CMAP)

    ax.set_xticks(np.arange(len(sensor_names)))
    ax.set_xticklabels(sensor_names, rotation=90)
    ax.set_yticks(np.arange(len(gas_names)))
    ax.set_yticklabels(gas_names)

    ax.set_xlabel("Sensor")
    ax.set_ylabel("Gas")
    ax.set_title("Raw Response Heatmap")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Response")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "01_heatmap.png"), dpi=300)
    plt.show()


def plot_sensor_dominance(X, sensor_names, save_dir):
    sensor_max = X.max(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sensor_names, sensor_max.values)

    ax.set_title("Max Response per Sensor")
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Max Response")

    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "02_sensor_max.png"), dpi=300)
    plt.show()


def compute_and_plot_cosine_distance(X, gas_names, save_dir):
    dist = squareform(pdist(X.values, metric=METRIC))
    dist_df = pd.DataFrame(dist, index=gas_names, columns=gas_names)

    print("\nCosine Distance")
    print(dist_df.round(3))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dist, cmap="bwr", vmin=0, vmax=2)

    ax.set_xticks(np.arange(len(gas_names)))
    ax.set_xticklabels(gas_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(gas_names)))
    ax.set_yticklabels(gas_names)

    ax.set_title("Cosine Distance", fontsize=14, y=1.02)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks(np.arange(0, 2.01, 0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "03_distance_cosine.png"), dpi=300)
    plt.show()

    return dist_df


def compute_and_plot_cosine_similarity(X, gas_names, save_dir):
    sim = cosine_similarity(X.values)
    sim_df = pd.DataFrame(sim, index=gas_names, columns=gas_names)

    print("\nCosine Similarity")
    print(sim_df.round(3))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, cmap="bwr", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(gas_names)))
    ax.set_xticklabels(gas_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(gas_names)))
    ax.set_yticklabels(gas_names)

    ax.set_title("Cosine Similarity", fontsize=14, y=1.02)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks(np.arange(-1, 1.01, 0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "04_cosine_similarity.png"), dpi=300)
    plt.show()

    return sim_df


def plot_cosine_dendrogram(X, gas_names, save_dir):
    Z = linkage(X.values, method=LINKAGE_METHOD, metric=METRIC)

    fig, ax = plt.subplots(figsize=(7, 6))
    dendrogram(Z, labels=gas_names, leaf_rotation=45, ax=ax)

    ax.set_ylabel("Cosine Distance", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(0, 1.4)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "05_dendrogram_cosine.png"), dpi=300)
    plt.show()

    return Z


# =========================================
# Saving results
# =========================================
def save_results(X, dist_df, sim_df, Z, save_dir):
    X.to_csv(os.path.join(save_dir, "00_raw_matrix_gas_x_sensor.csv"))
    dist_df.to_csv(os.path.join(save_dir, "distance_cosine.csv"))
    sim_df.to_csv(os.path.join(save_dir, "cosine_similarity.csv"))

    Z_df = pd.DataFrame(Z, columns=["cluster1", "cluster2", "distance", "sample_count"])
    Z_df.to_csv(os.path.join(save_dir, "linkage_cosine.csv"), index=False)


# =========================================
# Main
# =========================================
def main():
    parser = argparse.ArgumentParser(
        description="Cosine similarity / distance analysis for gas sensing data."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input Excel file (gas x sensor response matrix).",
    )
    parser.add_argument(
        "--outdir", "-o", default=None,
        help="Directory to save outputs. Defaults to the input file's directory.",
    )
    args = parser.parse_args()

    file_path = args.input
    save_dir = args.outdir or os.path.dirname(os.path.abspath(file_path))
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load data
    df = load_data(file_path)
    X = df.copy()

    gas_names_raw = X.index.tolist()
    sensor_names = X.columns.tolist()
    gas_names = format_gas_names(gas_names_raw)

    print("Shape (gas x sensor):", X.shape)

    # 2. Raw heatmap
    plot_raw_heatmap(X, gas_names, sensor_names, save_dir)

    # 3. Sensor dominance check
    plot_sensor_dominance(X, sensor_names, save_dir)

    # 4. Cosine distance matrix
    dist_df = compute_and_plot_cosine_distance(X, gas_names, save_dir)

    # 5. Cosine similarity matrix
    sim_df = compute_and_plot_cosine_similarity(X, gas_names, save_dir)

    # 6. Dendrogram (cosine distance)
    Z = plot_cosine_dendrogram(X, gas_names, save_dir)

    # 7. Save all results
    save_results(X, dist_df, sim_df, Z, save_dir)

    print("\nDone. Results saved to:", save_dir)


if __name__ == "__main__":
    main()
