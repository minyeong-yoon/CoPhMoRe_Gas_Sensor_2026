# Photonic Nose – Gas Sensing Data Analysis

This repository contains the data-analysis code used in:

> **A Scalable Photonic Nose Based on Corona Phase Molecular Recognition**
>
> Minyeong Yoon, Sihyung Lee, Jeonghyeon Cho, Seon Min Park, Seungju Lee, Sang Uck Lee, Soo-Yeon Cho
>
> Journal: *In Revision*
>
> DOI: *In Revision*

The scripts reproduce the response-pattern analysis (single-molecule gas discrimination) and the odor-classification analysis (complex, multi-component mixtures) reported in the manuscript.

## Contents

| Script | Purpose | Dataset |
|---|---|---|
| `cosine_similarity_heatmap_distance_dendrogram.py` | Cosine similarity / distance analysis and hierarchical clustering across single gas molecules | 8-gas response matrix (Excel) |
| `sensor_subset_pca.py` | Greedy sensor ranking and PCA as a function of the number of sensors, including dimension-normalized separation and effective rank with synthetic controls | Odor CSV files (Coffee / Wine / Perfume / Softener) |
| `sensor_subset_lda.py` | LDA classification of odor classes with 5-fold cross-validation, Fisher discriminant ratio before and after L2 normalization, and accuracy vs number of sensors | Odor CSV files (Coffee / Wine / Perfume / Softener) |

All three scripts are standalone command-line tools (`argparse`-based) — no file paths are hardcoded, so each can be run directly against your own data directory.

## Installation

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn openpyxl
```

(`openpyxl` is required by `pandas.read_excel` for `cosine_similarity_heatmap_distance_dendrogram.py`.)

The results in the manuscript were obtained with Python 3.12.4, numpy 1.26.4, scikit-learn 1.4.2, pandas 2.2.2 and matplotlib 3.8.4. To reproduce the cross-validated accuracies exactly, install these versions:

```bash
pip install numpy==1.26.4 scikit-learn==1.4.2 pandas==2.2.2 matplotlib==3.8.4 scipy seaborn openpyxl
```

---

## 1. `cosine_similarity_heatmap_distance_dendrogram.py`

Loads a gas × sensor response matrix (single-molecule measurements: H₂, CH₄, NH₃, CO₂, ethanol, acetone, toluene, NO₂) and analyzes how distinguishable the sensor response patterns are across gases, using cosine similarity/distance rather than raw magnitude.

**Input:** an Excel file where the first column is the gas name and the remaining columns are per-sensor response values (one row per gas).

**Usage**
```bash
python cosine_similarity_heatmap_distance_dendrogram.py --input /path/to/8_gases_response_n18_median.xlsx --outdir results
```

| Argument | Description | Default |
|---|---|---|
| `--input`, `-i` | Path to the input Excel file | required |
| `--outdir`, `-o` | Output directory | same folder as `--input` |

**Outputs**
- `01_heatmap.png` – raw gas × sensor response heatmap
- `02_sensor_max.png` – max response per sensor (sensor dominance check)
- `03_distance_cosine.png`, `distance_cosine.csv` – cosine distance matrix
- `04_cosine_similarity.png`, `cosine_similarity.csv` – cosine similarity matrix
- `05_dendrogram_cosine.png`, `linkage_cosine.csv` – hierarchical clustering dendrogram (average linkage, cosine distance)
- `00_raw_matrix_gas_x_sensor.csv` – copy of the loaded data matrix

---

## 2. `sensor_subset_pca.py`

Evaluates how the PCA representation of the complex-odor dataset (Perfume, Wine, Coffee, Fabric Softener; 20 products) changes as sensors are added. Sensors are ranked by a greedy rule (normalized ANOVA F-score − α × mean |Spearman ρ| to the sensors already selected, α = 0.35). For N = 2–18 top-ranked sensors, responses are standardized and analyzed by PCA. Dimension-normalized metrics are compared with two synthetic controls (first two ranked sensors kept, sensors 3–18 replaced by noise sensors or by redundant copies; 200 realizations, median and interquartile range).

**Input:** a folder with one response CSV file per product, named like:
```
Coffee 1_Ethiopia_260128 S2_response.csv
```
The label (Coffee/Wine/Perfume/Softener) is inferred from the filename. Each CSV has a header row; column 2 contains the response values, with 3 consecutive replicates per sensor (18 sensors × 3 = 54 rows).

**Usage**
```bash
python sensor_subset_pca.py --input_dir /path/to/csv_folder --outdir results
```

| Argument | Description | Default |
|---|---|---|
| `--input_dir`, `-i` | Folder of response CSV files | required |
| `--outdir`, `-o` | Output directory | `results` |

**Outputs**
- `PCA_score_XXsensors.png/.svg` – PCA score plots with convex hulls (N = 2, 4, 8, 12, 15, 18)
- `centroid_distance_PC1-PC2.png/.svg` – mean inter-class centroid distance in PC1–PC2 space
- `PCA_score_sqrtN_XXsensors.png/.svg` – PCA score plots with scores divided by √N
- `normalized_distance.png/.svg` – inter-class centroid distance / within-class distance in the full N-dimensional space, vs. noise control
- `effective_rank.png/.svg` – entropy-based effective rank of the covariance eigenvalues, vs. redundant control
- `measured_metrics.csv` – sensor ranking and all metrics for N = 2–18
- `control_metrics_all_reps.csv` – metrics for every control realization

**Note:** generating the synthetic controls (2 × 200 realizations × 17 subset sizes) takes about 1–2 minutes.

---

## 3. `sensor_subset_lda.py`

Classifies the complex-odor dataset (18-dimensional response vectors, averaged over the 3 replicates) into the four odor classes using LDA (StandardScaler + `lsqr` solver, `shrinkage="auto"`) with 5-fold cross-validation (GroupKFold, split by product). Decision boundaries in the 2D LDA score plot are obtained by fitting an additional LDA classifier to the LD1–LD2 coordinates. Class separability is quantified by the Fisher discriminant ratio, trace(Sb)/trace(Sw), in LD1–LD2 space, before and after L2 normalization of each response vector. Classification accuracy is also evaluated for every sensor combination at N = 1, 3, 6, 9, 12, 15 and 18.

**Input:** the same CSV folder format as `sensor_subset_pca.py`.

**Usage**
```bash
python sensor_subset_lda.py --input_dir /path/to/csv_folder --outdir results
```

| Argument | Description | Default |
|---|---|---|
| `--input_dir`, `-i` | Folder of response CSV files | required |
| `--outdir`, `-o` | Output directory | `results` |
| `--skip_subsets` | Skip the sensor-combination sweep | off |

**Outputs**
- `confusion_matrix.png/.svg`, `confusion_matrix_percent.csv` – confusion matrix from out-of-fold predictions
- `lda_scores.png/.svg`, `lda_scores.csv` – LDA score plot with decision boundaries
- `accuracy_vs_n_sensors.png/.svg`, `accuracy_vs_n_sensors.csv` – mean ± SD of cross-validated accuracy over all 18CN combinations
- `accuracy_all_combinations.csv` – accuracy of every sensor combination
- `lda_coefficient_heatmap.png/.svg`, `lda_coefficients.csv` – LDA coefficients per class and sensor
- `classwise_metrics.png/.svg`, `classwise_metrics.csv` – class-wise precision, recall and F1-score
- `lda_scores_L2-normalized.png/.svg`, `lda_scores_L2-normalized.csv` – LDA score plot after L2 normalization
- `fisher_discriminant_ratio.csv` – trace(Sb), trace(Sw) and their ratio in LD1–LD2 space (raw and L2-normalized)

**Notes**
- The sensor-combination sweep evaluates 87,399 combinations (e.g. 18 choose 9 = 48,620) and takes the longest; use `--skip_subsets` to run the other analyses only.
- The GroupKFold split of the 20 products depends on the scikit-learn / numpy version, so cross-validated accuracies can differ slightly between environments. The values in the manuscript were obtained with the versions listed under Installation.
- The absolute scale and sign of the LD scores also depend on the scikit-learn version. trace(Sb) and trace(Sw) scale together, so the Fisher discriminant ratio is unaffected.

---

## Repository structure

```
.
├── cosine_similarity_heatmap_distance_dendrogram.py
├── sensor_subset_pca.py
├── sensor_subset_lda.py
├── LICENSE
└── README.md
```

Each script is self-contained; run them independently against the appropriate dataset. No file paths are hardcoded — all inputs/outputs are passed via command-line arguments.
