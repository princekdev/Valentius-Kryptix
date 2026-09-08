"""
train.py
--------
End-to-end training pipeline for the "Linear Regression From Scratch"
project.

Pipeline steps:
    1. Load the raw dataset with Pandas.
    2. Clean the data (handle missing values).
    3. Select features (X) and target (y) for MULTIPLE linear regression.
    4. Split into train/test sets (sklearn.model_selection.train_test_split
       is explicitly allowed by the task for this step only).
    5. Standardize features (fit scaler on TRAIN data only, to avoid data
       leakage from the test set).
    6. Train the custom, from-scratch LinearRegressionScratch model using
       Gradient Descent.
    7. Plot and save the cost-convergence curve.
    8. Save all artifacts (model parameters, scaler stats, train/test split)
       needed by evaluate.py, so evaluation always uses the EXACT SAME split.

Run with:
    python src/train.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from linear_regression import LinearRegressionScratch

# --------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------- #
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------------------------------------------------- #
# Paths (relative to project root, regardless of where the script is run)
# --------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_COLUMN = "median_house_value"

# Numeric feature columns used for MULTIPLE linear regression.
# `ocean_proximity` (categorical) is intentionally excluded to keep the
# from-scratch implementation focused on pure numeric matrix algebra
# (see README "Limitations" section for discussion).
FEATURE_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

# Gradient Descent hyperparameters
LEARNING_RATE = 0.3
N_ITERATIONS = 5000


def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Load the dataset with Pandas and perform basic cleaning:
      - Report shape and missing values.
      - Impute missing numeric values with the column median
        (robust to outliers, unlike the mean).
    """
    df = pd.read_csv(path)

    print(f"Loaded dataset with shape: {df.shape}")
    print("\nMissing values per column (before cleaning):")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Impute missing values in numeric columns using the median.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    assert df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().sum() == 0, \
        "Missing values remain after cleaning!"

    print("\nMissing values per column (after cleaning): 0 (all handled)")
    return df


def standardize_features(X_train: np.ndarray, X_test: np.ndarray):
    """
    Standardize features to zero mean and unit variance:

        X_scaled = (X - mean) / std

    The scaler statistics (mean, std) are computed ONLY on the training
    set and then applied to both train and test sets, to prevent
    information from the test set leaking into training.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero for constant columns

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, mean, std


def plot_cost_convergence(cost_history: list, save_path: str) -> None:
    """Plot MSE cost vs. Gradient Descent iteration and save as PNG."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cost_history) + 1), cost_history, color="#2563eb", linewidth=2)
    plt.title("Cost Convergence During Gradient Descent")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Cost  J(theta)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved cost convergence plot to: {save_path}")


def main():
    print("=" * 70)
    print("STEP 1-2: Load & clean data")
    print("=" * 70)
    df = load_and_clean_data(DATA_PATH)

    print("\n" + "=" * 70)
    print("STEP 3: Prepare features (X) and target (y)")
    print("=" * 70)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    print(f"Feature matrix X shape: {X.shape}")
    print(f"Target vector y shape:  {y.shape}")

    print("\n" + "=" * 70)
    print("STEP 4: Train/test split (80/20, sklearn.train_test_split)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set:  X={X_test.shape}, y={y_test.shape}")

    print("\n" + "=" * 70)
    print("STEP 5: Standardize features (fit on train only)")
    print("=" * 70)
    X_train_scaled, X_test_scaled, feature_mean, feature_std = standardize_features(
        X_train, X_test
    )
    print("Feature means (train):", np.round(feature_mean, 3))
    print("Feature stds  (train):", np.round(feature_std, 3))

    print("\n" + "=" * 70)
    print("STEP 6: Train custom LinearRegressionScratch (Gradient Descent)")
    print("=" * 70)
    model = LinearRegressionScratch(
        learning_rate=LEARNING_RATE,
        n_iterations=N_ITERATIONS,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    print(f"Training complete after {N_ITERATIONS} iterations.")
    print(f"Initial cost: {model.cost_history[0]:.4f}")
    print(f"Final cost:   {model.cost_history[-1]:.4f}")
    print(f"Learned intercept (theta_0): {model.intercept_:.4f}")
    print("Learned coefficients (theta_1..n):")
    for name, coef in zip(FEATURE_COLUMNS, model.coef_):
        print(f"  {name:20s}: {coef:.4f}")

    print("\n" + "=" * 70)
    print("STEP 7: Plot cost convergence")
    print("=" * 70)
    plot_cost_convergence(
        model.cost_history, os.path.join(RESULTS_DIR, "cost_convergence.png")
    )

    print("\n" + "=" * 70)
    print("STEP 8: Save artifacts for evaluate.py")
    print("=" * 70)
    artifacts_path = os.path.join(RESULTS_DIR, "artifacts.npz")
    np.savez(
        artifacts_path,
        theta=model.theta,
        feature_mean=feature_mean,
        feature_std=feature_std,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        cost_history=np.array(model.cost_history),
        feature_columns=np.array(FEATURE_COLUMNS),
    )
    print(f"Saved artifacts to: {artifacts_path}")
    print("\nTraining pipeline finished successfully.")


if __name__ == "__main__":
    main()
