# ML-4: Model Evaluation on an Imbalanced Binary Classification Task

## Objective
Rigorously evaluate two classifiers on an imbalanced binary classification
dataset using Stratified K-Fold cross-validation, confusion matrices, ROC
curves, and Precision-Recall curves — and demonstrate why accuracy alone is
an unreliable metric under class imbalance.

## Dataset
`sklearn.datasets.load_digits` (1,797 handwritten-digit images, 8x8 pixels,
64 features), converted into a binary problem: **"is this digit an 8?"**
(positive class) vs. **"all other digits"** (negative class). This is a
built-in scikit-learn dataset (no manual download required) and produces a
real, naturally imbalanced binary classification task.

**Why it's imbalanced:** only 1 of 10 digit classes is treated as positive,
so the positive class makes up roughly 1/10th of the data:

| Class          | Count | Percentage |
|----------------|-------|-----------|
| 0 (not 8)      | 1623  | 90.3%     |
| 1 (is 8)       | 174   | 9.7%      |

## Models Used
- **Logistic Regression** (with `StandardScaler`)
- **Random Forest Classifier** (with `StandardScaler`, 200 trees)

Both wrapped in a scikit-learn `Pipeline` so preprocessing is fit only on
training folds/data — no leakage from validation or test data.

## Evaluation Methodology
1. Stratified train/test split (80/20), stratified on the label.
2. **Stratified 5-Fold Cross-Validation** on the training set only
   (`StratifiedKFold(n_splits=5)`), reporting mean ± std Accuracy and F1.
3. Final fit on the full training set; evaluation on the **untouched
   held-out test set** for confusion matrices, ROC curves, and PR curves.
4. All metrics are computed from actual model predictions — nothing is
   hardcoded.

## Cross-Validation Results (training set, 5 folds)
| Model               | CV Accuracy        | CV F1               |
|---------------------|---------------------|----------------------|
| Logistic Regression | 0.9589 ± 0.0102     | 0.7746 ± 0.0533      |
| Random Forest        | 0.9680 ± 0.0041     | 0.8012 ± 0.0288      |

## Held-Out Test Set Results
| Model               | Accuracy | F1     | ROC-AUC | Avg Precision |
|---------------------|----------|--------|---------|----------------|
| Logistic Regression | 0.9639   | 0.7937 | 0.9756  | 0.8928         |
| Random Forest        | 0.9667   | 0.7931 | 0.9952  | 0.9649         |

**Confusion matrix (test set):**
- Logistic Regression: TN=322, FP=3, FN=10, TP=25
- Random Forest: TN=325, FP=0, FN=12, TP=23

See `outputs/confusion_matrices.png` for the visual confusion matrices,
`outputs/roc_curve.png` for the combined ROC curve, and
`outputs/precision_recall_curve.png` for the combined PR curve.

### Confusion Matrix Explanation
- **TP (True Positive):** correctly predicted "is 8"
- **TN (True Negative):** correctly predicted "not 8"
- **FP (False Positive):** predicted "is 8" but the true digit was not 8 (false alarm)
- **FN (False Negative):** predicted "not 8" but the true digit was 8 (missed minority case — the costliest error under imbalance)

### ROC-AUC Explanation
ROC-AUC measures how well a model ranks positive examples above negative
examples across all thresholds, using True Positive Rate vs. False Positive
Rate. Both models scored highly here (0.976 and 0.995), but ROC-AUC can look
deceptively strong under imbalance because the large number of true
negatives dominates the False Positive Rate term.

### Precision-Recall Explanation
The PR curve plots Precision vs. Recall using only the positive class,
without reference to true negatives. Average Precision summarizes this
curve into a single number. On this dataset, Average Precision (0.893 vs.
0.965) shows a larger gap between the two models than ROC-AUC does — making
PR curves more sensitive to actual minority-class performance differences.

### Why PR Matters More Under Imbalance
With ~90% negative samples, a model can achieve a high ROC-AUC while still
missing a meaningful fraction of the minority class, because the FPR
denominator (all negatives) is large. Precision-Recall curves ignore true
negatives entirely and focus on how well the model handles the class we
actually care about — so they expose minority-class weaknesses that ROC-AUC
can mask.

### Accuracy Limitation (demonstrated)
A trivial classifier that always predicts "not 8" would score **90.3%
accuracy** while catching **0%** of the minority class — higher accuracy
than a genuinely useless model has any right to claim. Both trained models
score 96–97% accuracy, only marginally above this trivial baseline, while
their F1, ROC-AUC, and PR performance reveal the real story: they correctly
identify the minority class, unlike the trivial baseline. Accuracy alone
cannot distinguish "trivially ignore the minority class" from "genuinely
model it well" — F1/ROC-AUC/PR-AUC can.

## Model Comparison
| Metric               | Logistic Regression | Random Forest |
|-----------------------|----------------------|----------------|
| CV Accuracy (mean)    | 0.9589               | **0.9680**     |
| CV F1 (mean)          | 0.7746               | **0.8012**     |
| Test Accuracy         | 0.9639               | **0.9667**     |
| Test F1               | **0.7937**           | 0.7931 (~tied) |
| Test ROC-AUC          | 0.9756               | **0.9952**     |
| Test Avg. Precision   | 0.8928               | **0.9649**     |
| Test FP / FN          | 3 / 10               | **0 / 12**     |

## Final Deployment Decision
**Deploy: Random Forest.**

Random Forest has the higher mean CV F1 across all 5 folds (0.8012 vs.
0.7746) — a more reliable, multi-fold estimate of minority-class
performance than a single test split. This is corroborated on the held-out
test set, where Random Forest has substantially higher ROC-AUC (0.9952 vs.
0.9756) and Average Precision (0.9649 vs. 0.8928), and produces **zero
false positives**. Test-set F1 is essentially a tie (0.7931 vs. 0.7937, a
1-sample difference out of 360), so it should not be treated as
decisive — the more robust CV signal and the clearly better ranking quality
(ROC-AUC/AP) favor Random Forest overall.

*(Exact numbers above come from running `model_evaluation.py` with
`RANDOM_STATE = 42`; re-running will reproduce identical results.)*

## How to Run
```bash
pip install -r requirements.txt
python model_evaluation.py
```
Outputs (confusion matrices, ROC curve, PR curve) are written to `outputs/`.

## Project Structure
```
ml-4-model-evaluation/
├── README.md
├── requirements.txt
├── model_evaluation.py
└── outputs/
    ├── confusion_matrices.png
    ├── roc_curve.png
    └── precision_recall_curve.png
```

## Key Learning Outcomes
- Accuracy is a poor standalone metric under class imbalance; it can be
  dominated by the majority class.
- Stratified K-Fold cross-validation gives a more robust performance
  estimate than a single train/test split, especially with few minority
  samples.
- ROC-AUC can overstate performance under imbalance; Precision-Recall /
  Average Precision better reflect minority-class quality.
- Model selection should weigh multiple metrics (CV stability, F1, ROC-AUC,
  PR-AUC, confusion matrix) rather than any single number, and near-ties on
  one metric should be resolved using the more robust or more relevant
  ones.
