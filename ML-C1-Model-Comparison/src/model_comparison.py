"""
ML-C1: Model Comparison Report
================================
Compares Logistic Regression vs Random Forest on the Breast Cancer Wisconsin
dataset, then applies K-Means clustering (unsupervised) on the same feature
space for interpretation purposes only.

Run:
    pip install -r requirements.txt
    python src/model_comparison.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, silhouette_score
)

RANDOM_STATE = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
def load_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign
    target_names = data.target_names
    return X, y, target_names


# ---------------------------------------------------------------------------
# 2. SUPERVISED: TRAIN + EVALUATE
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name, scale=False):
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_pred


def plot_confusion_matrix(cm, target_names, model_name, filename):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. UNSUPERVISED: K-MEANS
# ---------------------------------------------------------------------------
def run_kmeans_analysis(X_scaled, y, target_names, k_range=range(2, 8)):
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Plot elbow + silhouette side by side
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), silhouettes, marker="o", color="darkorange")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "elbow_curve.png"), dpi=150)
    plt.close(fig)

    # Choose k with the best silhouette score (data-driven, not forced to match target)
    best_k = list(k_range)[int(np.argmax(silhouettes))]

    final_km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = final_km.fit_predict(X_scaled)

    return best_k, cluster_labels, inertias, silhouettes


def plot_pca_clusters(X_scaled, cluster_labels, y, target_names, filename):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", s=15)
    axes[0].set_title("PCA: K-Means Cluster Assignments")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    legend1 = axes[0].legend(*scatter1.legend_elements(), title="Cluster")
    axes[0].add_artist(legend1)

    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", s=15)
    axes[1].set_title("PCA: True Target Labels")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    handles, _ = scatter2.legend_elements()
    axes[1].legend(handles, target_names, title="Target")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)

    return pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("ML-C1: MODEL COMPARISON REPORT")
    print("=" * 70)

    # --- Load data ---
    X, y, target_names = load_data()
    print(f"\nDataset: Breast Cancer Wisconsin (Diagnostic)")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Target classes: {list(target_names)} (0=malignant, 1=benign)")

    # --- Train/test split (shared by both models, no leakage) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # --- Model 1: Logistic Regression (needs scaling) ---
    log_reg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    metrics_lr, cm_lr, _ = train_and_evaluate(
        X_train, X_test, y_train, y_test, log_reg, "Logistic Regression", scale=True
    )
    plot_confusion_matrix(cm_lr, target_names, "Logistic Regression", "confusion_matrix_model1.png")

    # --- Model 2: Random Forest (no scaling needed) ---
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    metrics_rf, cm_rf, _ = train_and_evaluate(
        X_train, X_test, y_train, y_test, rf, "Random Forest", scale=False
    )
    plot_confusion_matrix(cm_rf, target_names, "Random Forest", "confusion_matrix_model2.png")

    # --- Comparison table ---
    results_df = pd.DataFrame([metrics_lr, metrics_rf]).set_index("Model")
    print("\n" + "-" * 70)
    print("SUPERVISED MODEL COMPARISON (test set)")
    print("-" * 70)
    print(results_df.round(4).to_string())

    # --- Unsupervised: K-Means on full feature set (target NOT used) ---
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)

    best_k, cluster_labels, inertias, silhouettes = run_kmeans_analysis(X_scaled_full, y, target_names)
    print("\n" + "-" * 70)
    print("K-MEANS CLUSTERING")
    print("-" * 70)
    print(f"Selected k = {best_k} (highest silhouette score = {max(silhouettes):.4f})")

    explained_var = plot_pca_clusters(X_scaled_full, cluster_labels, y, target_names, "kmeans_pca.png")
    print(f"PCA (2D) explained variance: {explained_var[0]*100:.1f}% + {explained_var[1]*100:.1f}% "
          f"= {sum(explained_var)*100:.1f}% total")

    # Compare cluster assignments with actual target (interpretation only)
    cross_tab = pd.crosstab(
        pd.Series(cluster_labels, name="Cluster"),
        pd.Series(y.values, name="Actual Target").map({0: target_names[0], 1: target_names[1]})
    )
    print("\nCluster vs. Actual Target (interpretation only, not used for training):")
    print(cross_tab.to_string())

    # Simple agreement measure: for each cluster, % matching majority class
    majority_match = 0
    for cluster_id in cross_tab.index:
        majority_match += cross_tab.loc[cluster_id].max()
    agreement_rate = majority_match / len(y)
    print(f"\nCluster-to-majority-class agreement rate: {agreement_rate*100:.1f}%")

    # --- Save comparison table to outputs for reference ---
    results_df.round(4).to_csv(os.path.join(OUTPUT_DIR, "comparison_table.csv"))
    cross_tab.to_csv(os.path.join(OUTPUT_DIR, "cluster_vs_target.csv"))

    # --- Final printed conclusions ---
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    better_model = results_df["F1"].idxmax()
    print(f"Better performing model (by F1-score): {better_model}")
    print(f"Recall is the most clinically relevant metric here, since missing a")
    print(f"malignant case (false negative) is far costlier than a false alarm.")
    print(f"K-Means (k={best_k}) found natural groupings that agree with the true")
    print(f"diagnosis labels for {agreement_rate*100:.1f}% of samples, suggesting the two")
    print(f"underlying classes are largely separable by feature geometry alone,")
    print(f"which supports (but does not replace) the supervised results.")
    print("\nDone. Plots saved to outputs/. See README.md for full analysis.")


if __name__ == "__main__":
    main()
