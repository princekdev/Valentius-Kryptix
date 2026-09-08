"""
simple_demo.py
--------------
A small, self-contained demonstration of SIMPLE Linear Regression
(a single feature) using the same LinearRegressionScratch class used for
the multiple-feature model in train.py / evaluate.py.

This satisfies the task requirement to demonstrate both:
    - Simple Linear Regression  (this script — 1 feature)
    - Multiple Linear Regression (train.py / evaluate.py — 8 features)

Feature used : median_income  (single strongest predictor)
Target       : median_house_value

Run with:
    python src/simple_demo.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from linear_regression import (
    LinearRegressionScratch,
    mean_squared_error_manual,
    r2_score_manual,
)

RANDOM_STATE = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLUMN = "median_income"
TARGET_COLUMN = "median_house_value"


def main():
    print("=" * 70)
    print("SIMPLE LINEAR REGRESSION DEMO (single feature: median_income)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[FEATURE_COLUMN, TARGET_COLUMN])

    X = df[[FEATURE_COLUMN]].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Standardize the single feature
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000,
                                     random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_mse = mean_squared_error_manual(y_train, y_pred_train)
    test_mse = mean_squared_error_manual(y_test, y_pred_test)
    train_r2 = r2_score_manual(y_train, y_pred_train)
    test_r2 = r2_score_manual(y_test, y_pred_test)

    print(f"Intercept (theta_0): {model.intercept_:.4f}")
    print(f"Coefficient (theta_1) for '{FEATURE_COLUMN}': {model.coef_[0]:.4f}")
    print(f"Train MSE: {train_mse:,.4f}")
    print(f"Test  MSE: {test_mse:,.4f}")
    print(f"Train R^2: {train_r2:.6f}")
    print(f"Test  R^2: {test_r2:.6f}")

    # ------------------------------------------------------------------ #
    # Visualization: scatter of test data + fitted regression line
    # ------------------------------------------------------------------ #
    order = np.argsort(X_test_scaled[:, 0])
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test[:, 0], y_test, alpha=0.2, s=10, color="#64748b",
                label="Test data")
    plt.plot(X_test[order, 0], y_pred_test[order], color="#dc2626",
              linewidth=2, label="Fitted line (from scratch)")
    plt.title("Simple Linear Regression: median_income vs. median_house_value")
    plt.xlabel("median_income")
    plt.ylabel("median_house_value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "simple_regression_plot.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved simple regression plot to: {out_path}")


if __name__ == "__main__":
    main()
