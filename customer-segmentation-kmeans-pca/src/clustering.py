"""
clustering.py
-------------
End-to-end unsupervised learning pipeline: Customer Segmentation using
K-Means clustering and PCA.

Run: python src/clustering.py
Outputs (saved to outputs/):
    - elbow_plot.png
    - silhouette_plot.png
    - pca_clusters.png
    - cluster_summary.csv
    - results.json  (all key numbers used in README, no fabrication)
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
DATA_PATH = "data/Mall_Customers.csv"
OUT_DIR = "outputs"

# ------------------------------------------------------------------
# 1. Load and inspect
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.info())
print(df.describe())

# ------------------------------------------------------------------
# 2. Clean data (handle missing / non-numeric values)
# ------------------------------------------------------------------
print("\nMissing values per column:\n", df.isnull().sum())
df = df.drop_duplicates()
df = df.dropna()  # no missing values expected, but handled defensively

# ------------------------------------------------------------------
# 3. Select numerical features (exclude ID and categorical Genre;
#    no target/label column exists — this is unsupervised)
# ------------------------------------------------------------------
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[FEATURES].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------
# 4. Determine optimal k: elbow (inertia) + silhouette score
# ------------------------------------------------------------------
k_range = range(2, 11)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

best_k_idx = int(np.argmax(sil_scores))
best_k = list(k_range)[best_k_idx]
best_sil = sil_scores[best_k_idx]

print("\nInertias:", dict(zip(k_range, inertias)))
print("Silhouette scores:", dict(zip(k_range, sil_scores)))
print(f"Optimal k chosen (max silhouette): {best_k} (silhouette={best_sil:.4f})")

# Elbow plot
plt.figure(figsize=(7, 5))
plt.plot(list(k_range), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("Elbow Method for Optimal k")
plt.axvline(best_k, color="red", linestyle="--", label=f"Chosen k={best_k}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/elbow_plot.png", dpi=150)
plt.close()

# Silhouette plot
plt.figure(figsize=(7, 5))
plt.plot(list(k_range), sil_scores, marker="o", color="green")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Score vs k")
plt.axvline(best_k, color="red", linestyle="--", label=f"Chosen k={best_k}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/silhouette_plot.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 5. Fit final K-Means with optimal k
# ------------------------------------------------------------------
final_km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
df["Cluster"] = final_km.fit_predict(X_scaled)

# ------------------------------------------------------------------
# 6. PCA to 2 components (same scaled features)
# ------------------------------------------------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pcs = pca.fit_transform(X_scaled)
df["PC1"] = pcs[:, 0]
df["PC2"] = pcs[:, 1]

evr = pca.explained_variance_ratio_
total_var = evr.sum()
print(f"\nPC1 explained variance ratio: {evr[0]:.4f}")
print(f"PC2 explained variance ratio: {evr[1]:.4f}")
print(f"Total variance captured by 2D projection: {total_var:.4f}")

# PCA scatter plot colored by cluster
plt.figure(figsize=(7, 6))
scatter = plt.scatter(df["PC1"], df["PC2"], c=df["Cluster"], cmap="tab10", s=40)
plt.xlabel(f"PC1 ({evr[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
plt.title(f"PCA Projection Colored by K-Means Clusters (k={best_k})")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pca_clusters.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 9. Cluster statistics / interpretation
# ------------------------------------------------------------------
cluster_summary = df.groupby("Cluster")[FEATURES].mean().round(2)
cluster_summary["Count"] = df.groupby("Cluster").size()
cluster_summary["Genre_Female_%"] = (
    df.groupby("Cluster")["Genre"].apply(lambda s: (s == "Female").mean() * 100).round(1)
)
cluster_summary.to_csv(f"{OUT_DIR}/cluster_summary.csv")
print("\nCluster summary:\n", cluster_summary)

# ------------------------------------------------------------------
# Save all real, computed results for the README (no fabrication)
# ------------------------------------------------------------------
results = {
    "n_samples": int(len(df)),
    "features_used": FEATURES,
    "k_range_tested": list(k_range),
    "inertias": {int(k): float(v) for k, v in zip(k_range, inertias)},
    "silhouette_scores": {int(k): float(v) for k, v in zip(k_range, sil_scores)},
    "optimal_k": int(best_k),
    "optimal_k_silhouette_score": float(best_sil),
    "pca_pc1_explained_variance_ratio": float(evr[0]),
    "pca_pc2_explained_variance_ratio": float(evr[1]),
    "pca_total_variance_2d": float(total_var),
    "cluster_summary": json.loads(cluster_summary.to_json(orient="index")),
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll outputs saved to outputs/. Done.")
