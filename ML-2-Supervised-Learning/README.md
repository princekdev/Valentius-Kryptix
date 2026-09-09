# ML-2 — Supervised Learning: Comparative Study of 5 Algorithms

A fair, reproducible comparison of five distinct supervised classification
algorithms — **Logistic Regression, Decision Tree, KNN, SVM, and Gaussian
Naive Bayes** — trained and evaluated on the **same** real-world dataset,
the **same** train/test split, and the **same** preprocessing pipeline.

## Overview

This project answers a practical question: *given one real dataset, how
differently do five standard supervised learning algorithms actually
perform, and why?* Every model shares an identical `ColumnTransformer`
preprocessing pipeline (fit only on training data), the same stratified
80/20 split (`random_state=42`), and is scored on the same held-out test
set with Accuracy, Precision, Recall, and F1-score.

## Objective

Comparing multiple algorithms under strictly controlled, identical
conditions isolates what actually varies: the **algorithm's inductive
bias** (linear vs. non-linear decision boundaries, distance-based vs.
probabilistic vs. rule-based reasoning) — rather than differences caused by
inconsistent data handling. This mirrors real model-selection practice.

## Dataset

- **Name:** Adult Census Income (UCI / Kohavi, 1996)
- **Source:** public GitHub mirror of the UCI dataset (see
  [`data/README.md`](data/README.md) for the exact source and rationale)
- **Full size:** 48,842 rows × 14 features
- **Used for training/evaluation:** a **stratified subsample of 8,000
  rows** (6,400 train / 1,600 test) — done purely for computational
  tractability, since the SVM (RBF kernel) scales poorly with sample size;
  class proportions are preserved exactly via stratified sampling
- **Target variable:** `income` — binary (`<=50K` = 0, `>50K` = 1)
- **Class balance:** ~76% / 24% (real imbalance, preserved in both splits)
- **Features:** 6 numerical (age, fnlwgt, education-num, capital-gain,
  capital-loss, hours-per-week) + 8 categorical (workclass, education,
  marital-status, occupation, relationship, race, sex, native-country) →
  **104 columns** after one-hot encoding
- **Missing values:** `workclass` (2,799), `occupation` (2,809),
  `native-country` (857) in the full dataset, encoded as `"?"` in the raw
  file
- **Why this dataset:** it is real-world and non-trivial — classes overlap
  substantially (income is not linearly or cleanly separable from these
  features), it mixes numeric and categorical data, has genuine missing
  values, and has real class imbalance — exactly the conditions where
  different algorithms are expected to diverge in performance, unlike
  Iris.

## Algorithms Used

1. **Logistic Regression** — fits a linear decision boundary by modeling
   the log-odds of the positive class as a linear function of the
   (scaled) features.
2. **Decision Tree** — recursively splits the feature space on the single
   feature/threshold that best separates the classes at each node, forming
   an interpretable set of if/else rules.
3. **K-Nearest Neighbors (KNN)** — classifies a point by majority vote
   among its `k` closest training points in feature space; makes no
   assumption about the decision boundary's shape.
4. **Support Vector Machine (SVM, RBF kernel)** — finds the boundary that
   maximizes the margin between classes, using the RBF kernel to model
   non-linear boundaries implicitly.
5. **Gaussian Naive Bayes** — applies Bayes' theorem assuming every feature
   is conditionally independent given the class, and that continuous
   features are normally distributed.

## Data Preprocessing

- **Missing values:** `"?"` markers converted to `NaN`; numerical columns
  imputed with the **median**, categorical columns imputed with the **most
  frequent** category (both fit on training data only).
- **Categorical encoding:** `OneHotEncoder(handle_unknown="ignore")` on the
  8 categorical columns (98 dummy columns produced); `handle_unknown` makes
  the pipeline robust to any unseen category in the test set.
- **Numerical scaling:** `StandardScaler` on the 6 numerical columns —
  benefits Logistic Regression, KNN, and SVM directly; is harmless for
  Decision Tree and Naive Bayes since it's applied identically and doesn't
  change their relative feature information.
- **Train/test split:** performed **before** any preprocessing is fit.
- **Leakage prevention:** the entire preprocessing logic lives inside an
  sklearn `ColumnTransformer`, wrapped in a `Pipeline` with each classifier.
  `pipeline.fit(X_train, y_train)` fits imputers/scaler/encoder on
  **training data only**; `pipeline.predict(X_test)` reuses those already-
  fitted transformers to transform the test data — the test set is never
  used to fit anything.

## Experimental Setup

| Setting | Value |
|---|---|
| Train/test ratio | 80% / 20% (stratified) |
| `random_state` | 42 (dataset subsample, split, and all applicable models) |
| Preprocessing | Shared `ColumnTransformer` (median/mode imputation + `StandardScaler` + `OneHotEncoder`), identical across all 5 models |
| Class imbalance | Yes — ~76% `<=50K` vs. ~24% `>50K`, preserved via stratified sampling/splitting |
| Primary evaluation metric | **F1-score (weighted average)** |

## Evaluation Metrics

- **Accuracy** — fraction of correct predictions overall.
- **Precision** — of predicted positives, how many were actually positive.
- **Recall** — of actual positives, how many were correctly identified.
- **F1-score** — harmonic mean of precision and recall.

All four are computed with **weighted averaging** (`average="weighted"`),
which weights each class's score by its number of true instances
(support). This was chosen over macro-averaging because the dataset is
imbalanced (~76/24): weighted averaging reflects performance on the actual
class distribution the model will see, rather than treating the rare class
as equally important to the common one (which macro-averaging would do,
potentially overstating the impact of the minority class). **F1-score** was
selected as the primary ranking metric because accuracy alone can be
misleading under class imbalance (a model predicting the majority class
100% of the time would score ~76% accuracy while being useless for the
minority class); F1 balances precision and recall in a single number.

## Results

*(Actual output from running `python src/main.py`, 6,400 train / 1,600
test rows, `random_state=42`.)*

| Model               |   Accuracy |   Precision |   Recall |   F1 Score |
|:--------------------|-----------:|------------:|---------:|-----------:|
| SVM                 |     0.8588 |      0.8526 |   0.8588 |     0.8529 |
| Decision Tree       |     0.8550 |      0.8483 |   0.8550 |     0.8470 |
| Logistic Regression |     0.8531 |      0.8467 |   0.8531 |     0.8475 |
| KNN                 |     0.8381 |      0.8319 |   0.8381 |     0.8338 |
| Naive Bayes         |     0.3750 |      0.7768 |   0.3750 |     0.3431 |

(Table ranked by F1-score; also saved at
[`results/model_comparison.csv`](results/model_comparison.csv).)

## Visualization

- Primary comparison chart (F1-score):
  [`results/model_comparison.png`](results/model_comparison.png)
- Secondary comparison chart (Accuracy):
  [`results/model_comparison_accuracy.png`](results/model_comparison_accuracy.png)
- Per-model confusion matrices:
  - [`results/confusion_matrix_svm.png`](results/confusion_matrix_svm.png)
  - [`results/confusion_matrix_decision_tree.png`](results/confusion_matrix_decision_tree.png)
  - [`results/confusion_matrix_logistic_regression.png`](results/confusion_matrix_logistic_regression.png)
  - [`results/confusion_matrix_knn.png`](results/confusion_matrix_knn.png)
  - [`results/confusion_matrix_naive_bayes.png`](results/confusion_matrix_naive_bayes.png)

## Model Analysis

**Why SVM (RBF) won (F1 = 0.853):** Income vs. these census features is not
linearly separable — e.g. the effect of `hours-per-week` or `age` on income
plausibly interacts non-linearly with `education-num` and
`marital-status`. The RBF kernel lets SVM carve a flexible, non-linear
boundary in the 104-dimensional standardized feature space without
explicit feature engineering, and margin maximization makes it robust to
the class overlap inherent in income prediction. It essentially tied with
Decision Tree and Logistic Regression, though — the margin over the next
two models (Decision Tree 0.847, Logistic Regression 0.848) is small,
suggesting the dominant signal (largely driven by `education-num`,
`marital-status`, `capital-gain`, and `hours-per-week`) is close to
linearly separable for the *bulk* of cases, with SVM's non-linear boundary
picking up the remaining harder cases at the margin.

**Why Decision Tree and Logistic Regression performed almost identically
well:** a single tree of depth 12 can approximate the same broad decision
regions a linear model finds, when a handful of features (education,
marital status, capital gain) dominate the signal — consistent with this
dataset's known structure, where `marital-status`/`relationship` and
`education-num` are strong, fairly monotonic predictors of income.

**Why KNN lagged behind (F1 = 0.834):** after one-hot encoding, the feature
space has **104 dimensions**, mostly sparse binary categorical dummies. In
this high-dimensional space, Euclidean distance becomes less meaningful
(the "curse of dimensionality") — most points end up roughly equidistant,
diluting the value of "nearest" neighbors and blending signal from the
dominant numerical features with noise from many sparse categorical
dimensions.

**Why Naive Bayes struggled badly (F1 = 0.343, Accuracy 0.375):** its
confusion matrix shows it drastically over-predicts the `>50K` class (984
false positives out of 1,217 true `<=50K` test cases). Two of Naive Bayes'
core assumptions are strongly violated by this dataset: (1) **feature
independence** — `education` and `education-num` encode the same
information, and `marital-status`/`relationship` are highly correlated, so
Naive Bayes effectively "double counts" the same evidence multiple times
when combining per-feature likelihoods; (2) **Gaussian-distributed
features** — `capital-gain` and `capital-loss` are extremely skewed (mostly
zero with rare large values), nothing like a normal distribution, which
badly miscalibrates the likelihood Naive Bayes assigns to those features.
Combined with 98 sparse one-hot dummy "features" (each modeled as if it
were Gaussian, which is a poor fit for a 0/1 indicator), these violations
compound and push Naive Bayes' predictions systematically toward the
positive class.

## Key Learnings

- No single algorithm dominates by default — the "best" model depends on
  how well its assumptions match the data (linearity, feature independence,
  distance-metric meaningfulness in high dimensions).
- One-hot encoding categorical features can quietly create high-dimensional,
  sparse spaces that hurt distance-based methods (KNN) even when they don't
  hurt margin-based or tree-based methods as much.
- Naive Bayes' simplicity (fast, no tuning) comes at a real cost when its
  independence and distributional assumptions are violated — as they
  clearly are here.
- A fair comparison requires controlling the *entire* pipeline (split,
  imputation, scaling, encoding), not just the model — otherwise apparent
  differences in performance could just be preprocessing artifacts.

## How to Run

```bash
git clone <your-repo-url>
cd ML-2-Supervised-Learning
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

All results (comparison table, comparison charts, confusion matrices) are
(re)generated in `results/`.

## Reproducibility

`random_state=42` is fixed everywhere randomness occurs: the stratified
dataset subsample, the train/test split, and every model/algorithm that
accepts a seed (Logistic Regression, Decision Tree, SVM). Because every
model is trained through the identical `Pipeline([preprocessor,
classifier])` object — fit once per model, always on the same `X_train`/
`y_train` — preprocessing statistics (imputation values, scaling
mean/std, one-hot categories) are guaranteed identical across models, and
re-running `python src/main.py` reproduces the exact table above.

## Conclusion

Under identical data and preprocessing, SVM (RBF), Decision Tree, and
Logistic Regression all perform comparably well (F1 ≈ 0.85) on this income
classification task, KNN trails modestly due to high-dimensional sparse
features, and Gaussian Naive Bayes fails significantly because its
independence and normality assumptions are strongly violated by this
dataset's structure. The experiment demonstrates concretely that algorithm
choice should be informed by how well an algorithm's assumptions match the
data's actual characteristics — not just by ease of implementation.

---

## Final Audit

1. **Dataset used:** Adult Census Income (UCI, via GitHub mirror), 48,842
   rows full / 8,000-row stratified subsample used for training (6,400
   train / 1,600 test), 14 raw features → 104 after preprocessing.
2. **Five algorithms:** Logistic Regression, Decision Tree, KNN, SVM (RBF),
   Gaussian Naive Bayes.
3. **Train/test split:** 80/20, stratified, `random_state=42`, identical
   for all 5 models.
4. **Actual results table:** see [Results](#results) above.
5. **Best model:** SVM, F1 = 0.8529.
6. **Worst model:** Naive Bayes, F1 = 0.3431.
7. **Files created:** `src/data_preprocessing.py`, `src/train_models.py`,
   `src/evaluate_models.py`, `src/main.py`, `data/adult.csv`,
   `data/README.md`, `results/model_comparison.csv`,
   `results/model_comparison.png`, `results/model_comparison_accuracy.png`,
   5× `results/confusion_matrix_*.png`, `README.md`, `requirements.txt`,
   `.gitignore`.
8. **Exact command to run:** `python src/main.py` (from the repository
   root, after `pip install -r requirements.txt`).
9. **Limitations/assumptions:** a 8,000-row stratified subsample (not the
   full 48,842 rows) was used so the RBF-kernel SVM trains in reasonable
   time; hyperparameters are reasonable defaults, not extensively tuned;
   `fnlwgt` (a census sampling weight) was kept as a plain numeric feature
   rather than excluded, though it is not a natural predictor of income.
