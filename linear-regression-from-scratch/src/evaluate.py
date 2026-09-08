"""
evaluate.py
-----------
Evaluates the custom from-scratch Linear Regression model on the test set,
using ONLY manual NumPy implementations of MSE and R^2 (no sklearn.metrics).

It then trains scikit-learn's LinearRegression on the EXACT SAME
train/test split (loaded from the artifacts saved by train.py) purely as
an independent sanity check, and prints a side-by-side comparison of:
    - coefficients
    - intercept
    - test MSE
    - test R^2

Run with:
    python src/train.py      # must be run first to generate artifacts.npz
    python src/evaluate.py
"""

import os

import numpy as np
from sklearn.linear_model import LinearRegression

from linear_regression import (
    LinearRegressionScratch,
    mean_squared_error_manual,
    r2_score_manual,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ARTIFACTS_PATH = os.path.join(RESULTS_DIR, "artifacts.npz")


def load_artifacts():
    if not os.path.exists(ARTIFACTS_PATH):
        raise FileNotFoundError(
            f"Could not find {ARTIFACTS_PATH}. Run `python src/train.py` first."
        )
    data = np.load(ARTIFACTS_PATH, allow_pickle=True)
    return data


def rebuild_custom_model(data) -> LinearRegressionScratch:
    """Reconstruct the trained custom model from saved theta + cost history."""
    model = LinearRegressionScratch()
    model.theta = data["theta"]
    model.cost_history = list(data["cost_history"])
    return model


def main():
    data = load_artifacts()

    feature_columns = list(data["feature_columns"])
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # ------------------------------------------------------------------ #
    # 1. Evaluate the CUSTOM from-scratch model
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("CUSTOM FROM-SCRATCH MODEL (trained with manual Gradient Descent)")
    print("=" * 70)
    custom_model = rebuild_custom_model(data)

    y_pred_custom_train = custom_model.predict(X_train_scaled)
    y_pred_custom_test = custom_model.predict(X_test_scaled)

    custom_train_mse = mean_squared_error_manual(y_train, y_pred_custom_train)
    custom_test_mse = mean_squared_error_manual(y_test, y_pred_custom_test)
    custom_train_r2 = r2_score_manual(y_train, y_pred_custom_train)
    custom_test_r2 = r2_score_manual(y_test, y_pred_custom_test)

    print(f"Intercept (theta_0): {custom_model.intercept_:.4f}")
    print("Coefficients:")
    for name, coef in zip(feature_columns, custom_model.coef_):
        print(f"  {name:20s}: {coef:.4f}")
    print(f"\nTrain MSE: {custom_train_mse:,.4f}")
    print(f"Test  MSE: {custom_test_mse:,.4f}")
    print(f"Train R^2: {custom_train_r2:.6f}")
    print(f"Test  R^2: {custom_test_r2:.6f}")

    # ------------------------------------------------------------------ #
    # 2. Train scikit-learn's LinearRegression on the SAME split
    #    (final verification / sanity check ONLY)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("SCIKIT-LEARN LinearRegression (sanity-check only, same split)")
    print("=" * 70)
    sk_model = LinearRegression()
    sk_model.fit(X_train_scaled, y_train)

    y_pred_sk_train = sk_model.predict(X_train_scaled)
    y_pred_sk_test = sk_model.predict(X_test_scaled)

    # Metrics still computed manually (no sklearn.metrics used, per task rules)
    sk_train_mse = mean_squared_error_manual(y_train, y_pred_sk_train)
    sk_test_mse = mean_squared_error_manual(y_test, y_pred_sk_test)
    sk_train_r2 = r2_score_manual(y_train, y_pred_sk_train)
    sk_test_r2 = r2_score_manual(y_test, y_pred_sk_test)

    print(f"Intercept: {sk_model.intercept_:.4f}")
    print("Coefficients:")
    for name, coef in zip(feature_columns, sk_model.coef_):
        print(f"  {name:20s}: {coef:.4f}")
    print(f"\nTrain MSE: {sk_train_mse:,.4f}")
    print(f"Test  MSE: {sk_test_mse:,.4f}")
    print(f"Train R^2: {sk_train_r2:.6f}")
    print(f"Test  R^2: {sk_test_r2:.6f}")

    # ------------------------------------------------------------------ #
    # 3. Side-by-side comparison
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("COMPARISON: Custom (from scratch) vs. scikit-learn")
    print("=" * 70)
    header = f"{'Metric':<15}{'From Scratch':>18}{'Scikit-learn':>18}{'Abs. Diff':>15}"
    print(header)
    print("-" * len(header))
    print(f"{'Intercept':<15}{custom_model.intercept_:>18.4f}{sk_model.intercept_:>18.4f}"
          f"{abs(custom_model.intercept_ - sk_model.intercept_):>15.6f}")
    print(f"{'Test MSE':<15}{custom_test_mse:>18.4f}{sk_test_mse:>18.4f}"
          f"{abs(custom_test_mse - sk_test_mse):>15.4f}")
    print(f"{'Test R^2':<15}{custom_test_r2:>18.6f}{sk_test_r2:>18.6f}"
          f"{abs(custom_test_r2 - sk_test_r2):>15.6f}")
    print()
    print(f"{'Coefficient':<20}{'From Scratch':>15}{'Scikit-learn':>15}{'Abs. Diff':>15}")
    print("-" * 65)
    for name, c_coef, sk_coef in zip(feature_columns, custom_model.coef_, sk_model.coef_):
        print(f"{name:<20}{c_coef:>15.4f}{sk_coef:>15.4f}{abs(c_coef - sk_coef):>15.6f}")

    # Prediction differences
    pred_diff = np.abs(y_pred_custom_test - y_pred_sk_test)
    print(f"\nMean absolute difference between custom and sklearn test predictions: "
          f"{pred_diff.mean():.4f}")
    print(f"Max absolute difference between custom and sklearn test predictions: "
          f"{pred_diff.max():.4f}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<12}{'From Scratch':>16}{'Scikit-learn':>16}")
    print(f"{'Test MSE':<12}{custom_test_mse:>16,.2f}{sk_test_mse:>16,.2f}")
    print(f"{'Test R2':<12}{custom_test_r2:>16.4f}{sk_test_r2:>16.4f}")


if __name__ == "__main__":
    main()
