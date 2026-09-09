"""
main.py
-------
Runs the complete pipeline end to end:
    1. Load + inspect raw data
    2. Stratified subsample (for SVM tractability) + train/test split
    3. Build the shared preprocessing pipeline (fit on train only)
    4. Train all 5 models
    5. Evaluate all 5 models on the SAME test set
    6. Save comparison table (CSV), comparison charts (PNG), confusion
       matrices (PNG)

Run from the repository root with:
    python src/main.py
"""

from data_preprocessing import (
    build_preprocessor,
    inspect_data,
    load_raw_data,
    split_data,
    stratified_subsample,
)
from evaluate_models import (
    evaluate_all,
    plot_accuracy_chart,
    plot_comparison_chart,
    save_comparison_table,
)
from train_models import build_pipelines, train_all


def main():
    print("=" * 70)
    print("STEP 1: Load and inspect raw data")
    print("=" * 70)
    df = load_raw_data()
    inspect_data(df)

    print("\n" + "=" * 70)
    print("STEP 2: Stratified subsample + train/test split")
    print("=" * 70)
    df_sample = stratified_subsample(df)
    print(f"Subsampled dataset shape: {df_sample.shape}")
    print(f"Subsampled target distribution:\n{df_sample['income'].value_counts()}")

    X_train, X_test, y_train, y_test = split_data(df_sample)
    print(f"\nTrain set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    print(f"Train target balance:\n{y_train.value_counts(normalize=True)}")
    print(f"Test target balance:\n{y_test.value_counts(normalize=True)}")

    print("\n" + "=" * 70)
    print("STEP 3: Build shared preprocessing pipeline (fit on train only)")
    print("=" * 70)
    preprocessor = build_preprocessor()
    pipelines = build_pipelines(preprocessor)
    print(f"Built pipelines for: {list(pipelines.keys())}")

    print("\n" + "=" * 70)
    print("STEP 4: Train all 5 models on the SAME training data")
    print("=" * 70)
    fitted_pipelines = train_all(pipelines, X_train, y_train)

    print("\n" + "=" * 70)
    print("STEP 5: Evaluate all 5 models on the SAME test set")
    print("=" * 70)
    results_df = evaluate_all(fitted_pipelines, X_test, y_test)

    print("\n" + "=" * 70)
    print("STEP 6: Save comparison table + charts")
    print("=" * 70)
    save_comparison_table(results_df)
    plot_comparison_chart(results_df, metric="F1 Score")
    plot_accuracy_chart(results_df)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE (ranked by F1 Score)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    worst = results_df.iloc[-1]
    print(f"\nBest model:  {best['Model']}  (F1 = {best['F1 Score']:.4f})")
    print(f"Worst model: {worst['Model']}  (F1 = {worst['F1 Score']:.4f})")

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
