# Customer Segmentation — Unsupervised Learning (K-Means + PCA)

## 1. Problem

A retail business wants to understand its customers without any pre-existing
labels — there is no "correct" segment column to predict. This is a classic
**unsupervised learning** problem: group customers into meaningful segments
purely from their behavioral/demographic attributes, so marketing, pricing,
and retention strategies can be tailored per segment.

## 2. Dataset

`data/Mall_Customers.csv` — 200 customers, 5 columns:

| Column | Type | Description |
|---|---|---|
| CustomerID | int | Unique identifier (not used for clustering) |
| Genre | categorical | Male / Female (not used for clustering) |
| Age | numeric | Customer age |
| Annual Income (k$) | numeric | Annual income in thousands of dollars |
| Spending Score (1-100) | numeric | Score assigned by the mall based on spending behavior |

This mirrors the well-known **Mall Customer Segmentation** dataset structure
widely used for K-Means clustering demonstrations. Because this environment
has no general internet access to Kaggle, the 200 records were generated with
`src/generate_data.py` using a **fixed random seed (42)** and five realistic
underlying customer archetypes (young high spenders, cautious high earners,
low-income low spenders, etc.) — the same kinds of segments the original
dataset is famous for revealing. This keeps the project fully reproducible
while preserving realistic, real-world structure. No target/label column
exists anywhere in the data — `Cluster` is created by the model, not present
in the source file.

## 3. Preprocessing

1. Loaded with Pandas, inspected shape/dtypes/summary statistics (`df.info()`, `df.describe()`).
2. Checked for missing values (none found) and dropped duplicates defensively.
3. Selected only the **numerical, non-ID features**: `Age`, `Annual Income (k$)`, `Spending Score (1-100)`.
   `CustomerID` (identifier) and `Genre` (categorical, not central to the segmentation goal) were excluded.
4. Standardized the selected features with `StandardScaler` (zero mean, unit variance) — essential for
   K-Means and PCA, both of which are distance/variance based and sensitive to feature scale.

## 4. Choosing the optimal k (no guessing)

K-Means was fit for every `k` in `range(2, 11)` on the scaled features. Two metrics were tracked:

- **Inertia** (within-cluster sum of squares) → elbow method
- **Silhouette score** → cluster cohesion/separation quality

### Computed results

| k | Inertia | Silhouette |
|---|---|---|
| 2 | 335.57 | 0.3994 |
| 3 | 240.17 | 0.4242 |
| 4 | 148.75 | 0.4912 |
| **5** | **104.74** | **0.5101** |
| 6 | 90.58 | 0.4942 |
| 7 | 77.43 | 0.4819 |
| 8 | 66.97 | 0.4503 |
| 9 | 59.46 | 0.4196 |
| 10 | 53.79 | 0.4211 |

**Optimal k = 5**, selected as the k with the maximum silhouette score
(0.5101). This also sits at a visible bend in the elbow curve (inertia drop
slows noticeably after k=5), so both methods agree.

- `outputs/elbow_plot.png`
- `outputs/silhouette_plot.png`

## 5. Final K-Means model

`KMeans(n_clusters=5, random_state=42, n_init=10)` fit on the scaled
features, producing a `Cluster` label (0–4) for every customer.

## 6. PCA (2D projection)

PCA was fit on the **same scaled features** and reduced to 2 components for
visualization only (clustering itself was done on the full standardized
feature space, not the PCA-reduced space).

- **PC1 explained variance ratio: 0.5702 (57.0%)**
- **PC2 explained variance ratio: 0.3330 (33.3%)**
- **Total variance captured by the 2D projection: 0.9032 (90.3%)**

A 2D scatter plot colored by K-Means cluster label is saved at
`outputs/pca_clusters.png` and shows five well-separated groups, confirming
that K-Means found real structure in the data (not an artifact of a poor k).

## 7. Cluster interpretation (from `outputs/cluster_summary.csv`)

| Cluster | Age (avg) | Income k$ (avg) | Spending Score (avg) | Count | % Female |
|---|---|---|---|---|---|
| 0 | 25.2 | 27.8 | 78.9 | 42 | 61.9% |
| 1 | 43.5 | 24.8 | 19.6 | 39 | 51.3% |
| 2 | 48.4 | 89.6 | 18.0 | 39 | 53.8% |
| 3 | 32.5 | 87.5 | 78.8 | 41 | 56.1% |
| 4 | 43.4 | 54.5 | 51.0 | 39 | 35.9% |

**Business interpretation:**

- **Cluster 0 — Young Impulsive Spenders:** Young, low income, but very high
  spending score. Likely students/young professionals who spend
  disproportionately relative to income — good targets for loyalty programs
  and flexible payment offers.
- **Cluster 1 — Budget-Conscious:** Middle-aged, low income, low spending
  score. Price-sensitive segment; best reached with discounts and value
  bundles.
- **Cluster 2 — Wealthy but Reserved:** Older, high income, low spending
  score. High potential value but currently under-engaged — a prime target
  for premium/personalized marketing to unlock spend.
- **Cluster 3 — High-Value Customers:** Younger-to-middle-aged, high income,
  high spending score. The most valuable segment — prioritize retention,
  VIP treatment, and upsell campaigns.
- **Cluster 4 — Average/Middle-of-the-Road:** Middle-aged, moderate income,
  moderate spending. Stable, average customers with no extreme behavior.

## 8. Outputs

All generated inside `outputs/`:

- `elbow_plot.png` — inertia vs. k
- `silhouette_plot.png` — silhouette score vs. k
- `pca_clusters.png` — 2D PCA projection colored by cluster
- `cluster_summary.csv` — per-cluster mean statistics
- `results.json` — every numeric result used in this README (machine-readable, reproducible)

## 9. Findings

- The optimal number of customer segments is **5**, backed by both the
  elbow method and the silhouette score (0.51 — a reasonably strong,
  well-separated clustering result for real-world behavioral data).
- Two principal components capture **90.3%** of the total variance in the
  three scaled features, so the 2D PCA plot is a faithful, low-distortion
  visualization of the true cluster structure.
- The five segments map cleanly onto actionable, real-world marketing
  personas (impulsive young spenders, budget-conscious, reserved wealthy,
  high-value, and average customers).

## 10. Limitations

- The dataset is small (200 rows) and, due to offline constraints, is a
  reproducibly generated stand-in built to match the structure and
  statistical character of the well-known Mall Customer Segmentation
  dataset — conclusions are illustrative of the method, not a real mall's
  actual customer base.
- K-Means assumes roughly spherical, similarly-sized clusters and is
  sensitive to feature scaling and initialization (mitigated here with
  `n_init=10` and `StandardScaler`).
- Only 3 numerical features were used; adding more behavioral features
  (purchase frequency, recency, category preferences) would likely produce
  richer, more actionable segments in a production setting.
- Categorical `Genre` was not used in clustering; a follow-up could
  one-hot encode it and test whether it changes the segmentation.

## 11. Project structure

```
project/
├── data/
│   └── Mall_Customers.csv        # generated dataset
├── src/
│   ├── generate_data.py          # creates data/Mall_Customers.csv
│   └── clustering.py             # full pipeline: load → clean → scale →
│                                  # k selection → K-Means → PCA → plots → stats
├── outputs/
│   ├── elbow_plot.png
│   ├── silhouette_plot.png
│   ├── pca_clusters.png
│   ├── cluster_summary.csv
│   └── results.json
├── requirements.txt
└── README.md
```

## 12. How to run

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd customer-segmentation-kmeans-pca

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) regenerate the dataset — a copy is already committed in data/
python3 src/generate_data.py

# 5. Run the full pipeline
python3 src/clustering.py
```

All plots and stats will be (re)written to `outputs/`, and console output
will show the same inertia/silhouette tables and PCA variance numbers
reported in this README.

## 13. Requirements

See `requirements.txt` — only `pandas`, `numpy`, `scikit-learn`, and
`matplotlib` are used.
