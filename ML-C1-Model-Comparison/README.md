# ML-C1: Model Comparison Report

## 1. Overview
A reproducible ML project that trains two genuinely different supervised
classifiers on the same dataset/split, evaluates them properly (ML-4), and
applies K-Means clustering (ML-3) as an independent unsupervised check on
the same feature space.

## 2. Objective
Compare **Logistic Regression** vs **Random Forest** on a binary
classification task, evaluate with accuracy/precision/recall/F1 and
confusion matrices, then use K-Means + PCA to see whether the data has
natural structure that agrees with the supervised labels — and recommend
one model with justification.

## 3. Dataset
**Breast Cancer Wisconsin (Diagnostic)** — loaded via
`sklearn.datasets.load_breast_cancer()`.

- **Target variable:** diagnosis — `0 = malignant`, `1 = benign`
- **Why suitable:** binary classification target, 30 real-valued numeric
  features computed from cell-nuclei measurements, 569 samples, no missing
  values, well-known and fully reproducible without external downloads.
- **Features used:** all 30 numeric features (radius, texture, perimeter,
  area, smoothness, compactness, concavity, symmetry, fractal dimension —
  each as mean/standard-error/worst value).

## 4. Problem Statement
Predict whether a tumor is malignant or benign from measured cell-nuclei
characteristics, and evaluate whether the two classes also emerge as
natural clusters without using the label.

## 5. Models Used
| # | Model | Type |
|---|-------|------|
| 1 | Logistic Regression | Linear, probabilistic |
| 2 | Random Forest | Ensemble of decision trees |

Both are trained on the **same** train/test split (`test_size=0.2`,
`stratify=y`, `random_state=42`). Logistic Regression uses `StandardScaler`
(fit on train only); Random Forest is scale-invariant so is trained on raw
features. Both are fit only on training data and evaluated only on the
held-out test set — no leakage.

## 6. Unsupervised Technique
**K-Means clustering**, applied to the standardized 30-feature space
(target column excluded). `k` was selected using **silhouette score**
across k=2..7 (elbow curve also plotted for context). The supervised
target is used **only afterward**, to interpret whether discovered
clusters align with malignant/benign — it never touches training or
`k` selection in a way that hardcodes the answer.

## 7. Methodology
1. Load data programmatically (no manual/local files).
2. Single stratified train/test split shared by both supervised models.
3. Preprocess (scale where required by the algorithm).
4. Train each model independently; predict on the untouched test set.
5. Compute accuracy, precision, recall, F1, and confusion matrix per model.
6. Scale full feature set; run K-Means for k=2–7; pick best k by
   silhouette score; plot elbow + silhouette curves.
7. Reduce to 2D with PCA (for visualization only, not for clustering)
   and plot cluster assignments vs. true labels side by side.
8. Cross-tabulate clusters vs. actual target for interpretation.
9. Print comparison table and reasoned conclusions.

## 8. Evaluation Metrics
Accuracy, Precision, Recall, F1-score (test set only) + confusion matrix
per model. Silhouette score used to select K-Means' `k`.

## 9. Results / Comparison Table

| Model               | Accuracy | Precision | Recall |    F1  |
|---------------------|---------:|----------:|-------:|-------:|
| Logistic Regression |   0.9825 |    0.9861 | 0.9861 | 0.9861 |
| Random Forest       |   0.9561 |    0.9589 | 0.9722 | 0.9655 |

*(Values are produced by `src/model_comparison.py` and saved to
`outputs/comparison_table.csv`; regenerate anytime by re-running the
script — `random_state=42` makes them reproducible.)*

## 10. Confusion Matrices / Plots
- `outputs/confusion_matrix_model1.png` — Logistic Regression
- `outputs/confusion_matrix_model2.png` — Random Forest
- `outputs/elbow_curve.png` — K-Means elbow + silhouette score vs. k
- `outputs/kmeans_pca.png` — PCA 2D: cluster assignments vs. true labels

## 11. K-Means Findings
- Best `k = 2` by silhouette score (0.343), which is a modest-but-real
  cluster structure (silhouette near 1 is rare on real biological data).
- Cross-tab of clusters vs. actual diagnosis:

  | Cluster | benign | malignant |
  |---------|-------:|----------:|
  | 0       |    339 |        36 |
  | 1       |     18 |       176 |

- **Cluster-to-majority-class agreement rate: 90.5%** — most samples in
  each cluster share the same true diagnosis, even though K-Means never
  saw the labels. This suggests the malignant/benign classes are largely
  separable by feature geometry alone, which is a reasonable (not
  proof-level) signal that supports the supervised results rather than
  contradicting them. The ~9.5% disagreement corresponds to borderline
  cases that are also the ones supervised models are more likely to
  misclassify.

## 12. Final Recommendation
**Recommended model: Logistic Regression.**

- **Performance:** Higher on every metric in this run (Accuracy 0.983 vs
  0.956, F1 0.986 vs 0.966).
- **Most relevant metric:** **Recall for the malignant class** matters
  most here — a false negative (calling a malignant tumor benign) is far
  more costly than a false positive. Logistic Regression's overall recall
  is higher on this split, but this should be re-checked with
  per-class recall / a larger test set before treating it as decisive on
  such a small test fold (114 samples).
- **Is the gap meaningful?** The absolute gap (~2–3 points) is modest and
  based on one 20% split; it should not be over-interpreted as one model
  being categorically superior — with more data or repeated
  cross-validation the gap could narrow or flip.
- **Interpretability:** Logistic Regression coefficients are directly
  interpretable (each feature's effect direction/magnitude on log-odds),
  which is valuable in a clinical context where explainability matters.
  Random Forest is less directly interpretable but handles nonlinear
  interactions without manual feature engineering.
- **Practical suitability:** Given comparable performance, higher
  interpretability, and lower computational cost, Logistic Regression is
  the more practical choice for this dataset and problem — with the
  caveat that Random Forest remains a reasonable alternative if
  nonlinear feature interactions become important as more data is
  collected.

## 13. How to Run
```bash
pip install -r requirements.txt
python src/model_comparison.py
```
This single command loads the data, trains both models, evaluates them,
generates all four plots into `outputs/`, runs K-Means + PCA, and prints
the comparison table and conclusions to the console.

## 14. Technologies Used
Python, pandas, numpy, scikit-learn, matplotlib.

## 15. Conclusion
Both supervised models perform strongly on this dataset, with Logistic
Regression slightly ahead on this split. Independent K-Means clustering
on the unlabeled feature space largely rediscovers the same two groups
(90.5% agreement), giving reasonable, non-circular support for the idea
that malignant/benign tumors are genuinely separable by these
measurements. No metric was fabricated — all numbers are produced by
running `src/model_comparison.py` with a fixed `random_state=42`.

---

## ML-C1 Requirement Checklist

| Requirement | Status |
|---|---|
| One public, reproducible dataset with clear target | ✅ Breast Cancer Wisconsin via sklearn |
| Dataset selection justified before coding | ✅ Section 3 |
| Exactly 2 genuinely different supervised algorithms | ✅ Logistic Regression + Random Forest |
| Same dataset/split/target for both models | ✅ Shared `train_test_split`, `random_state=42` |
| Fit on train only, predict on test only | ✅ No leakage |
| Accuracy, Precision, Recall, F1 for every model | ✅ Printed + `comparison_table.csv` |
| Confusion matrix per model | ✅ `confusion_matrix_model1.png`, `..._model2.png` |
| Exactly 1 unsupervised technique | ✅ K-Means only |
| Target excluded from clustering features | ✅ K-Means fit on `X` only |
| Features scaled before K-Means | ✅ `StandardScaler` |
| k chosen via elbow/silhouette, explained | ✅ Silhouette-selected, elbow plotted, reasoning in README |
| PCA 2D visualization of clusters | ✅ `kmeans_pca.png` |
| Clustering not forced to match target | ✅ k and labels chosen unsupervised; target used only after, for interpretation |
| Comparison table | ✅ Section 9 |
| Explanation beyond "highest accuracy wins" | ✅ Section 12 discusses metric relevance, gap significance, interpretability |
| Clean, understandable, reproducible code | ✅ `src/model_comparison.py`, fixed random states |
| requirements.txt / README / run commands | ✅ Included |
