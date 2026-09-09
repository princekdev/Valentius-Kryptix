"""
evaluate_models.py
-------------------
Evaluates every trained model on the SAME held-out test set using
Accuracy, Precision, Recall, and F1-score (weighted average — see README
for justification), generates a confusion matrix plot per model, builds
the final comparison table (CSV) and comparison bar chart (PNG).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Weighted averaging is used because the target classes are imbalanced
# (~76% <=50K vs ~24% >50K); weighted precision/recall/F1 accounts for
# class support instead of treating both classes as equally frequent.
AVERAGE_METHOD = "weighted"


def evaluate_model(name: str, pipeline, X_test, y_test) -> dict:
    """Compute Accuracy/Precision/Recall/F1 for one fitted pipeline."""
    y_pred = pipeline.predict(X_test)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=AVERAGE_METHOD, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=AVERAGE_METHOD, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average=AVERAGE_METHOD, zero_division=0),
    }


def save_confusion_matrix(name: str, pipeline, X_test, y_test) -> None:
    """Plot + save a confusion matrix PNG for one model."""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["<=50K", ">50K"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()

    filename = f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    out_path = RESULTS_DIR / filename
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def evaluate_all(fitted_pipelines: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate every model, save confusion matrices, return results table."""
    rows = []
    for name, pipeline in fitted_pipelines.items():
        print(f"Evaluating {name} ...")
        rows.append(evaluate_model(name, pipeline, X_test, y_test))
        save_confusion_matrix(name, pipeline, X_test, y_test)

    results_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    return results_df


def save_comparison_table(results_df: pd.DataFrame) -> Path:
    out_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved comparison table to: {out_path}")
    return out_path


def plot_comparison_chart(results_df: pd.DataFrame,
                           metric: str = "F1 Score") -> Path:
    """Professional bar chart comparing all 5 models on the primary metric."""
    df_sorted = results_df.sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(df_sorted["Model"], df_sorted[metric], color="#2563eb")

    ax.set_title(f"Model Comparison — {metric} (weighted)", fontsize=13)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)

    for bar, value in zip(bars, df_sorted[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01,
                 f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_path = RESULTS_DIR / "model_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison chart to: {out_path}")
    return out_path


def plot_accuracy_chart(results_df: pd.DataFrame) -> Path:
    """Optional secondary chart comparing Accuracy across models."""
    df_sorted = results_df.sort_values("Accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(df_sorted["Model"], df_sorted["Accuracy"], color="#16a34a")

    ax.set_title("Model Comparison — Accuracy", fontsize=13)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)

    for bar, value in zip(bars, df_sorted["Accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01,
                 f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_path = RESULTS_DIR / "model_comparison_accuracy.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved accuracy chart to: {out_path}")
    return out_path
