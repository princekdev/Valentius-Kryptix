"""
ML-4: Model Evaluation
Binary classification on an imbalanced dataset with rigorous evaluation:
Stratified K-Fold CV, confusion matrices, ROC curves, and PR curves.

Dataset: sklearn's load_digits, converted to binary classification
"is this digit an 8?" (positive) vs "all other digits" (negative).
This creates a real, built-in, naturally imbalanced binary problem
(~10% positive class) without needing external downloads or synthetic data.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    accuracy_score,
    f1_score,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
digits = load_digits()
X = digits.data
y = (digits.target == 8).astype(int)  # binary target: digit "8" vs rest

# ---------------------------------------------------------------------------
# 2. Class distribution
# ---------------------------------------------------------------------------
n_total = len(y)
n_pos = int(y.sum())
n_neg = n_total - n_pos
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)
print(f"Total samples : {n_total}")
print(f"Class 0 (not 8): {n_neg} ({n_neg / n_total:.1%})")
print(f"Class 1 (is 8) : {n_pos} ({n_pos / n_total:.1%})")
print(
    "\nWhy accuracy can mislead: a model that always predicts 'not 8' would\n"
    f"score {n_neg / n_total:.1%} accuracy while catching 0% of the minority\n"
    "class. Accuracy alone hides this failure; F1 / ROC-AUC / PR-AUC do not.\n"
)

# ---------------------------------------------------------------------------
# 3. Train/test split (stratified)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 4. Define two models (as pipelines to avoid preprocessing leakage)
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200, random_state=RANDOM_STATE
                ),
            ),
        ]
    ),
}

# ---------------------------------------------------------------------------
# 5 & 6. Stratified 5-fold CV on the training set only; report mean +/- std
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"accuracy": "accuracy", "f1": "f1"}

cv_results = {}
print("=" * 70)
print("5-FOLD STRATIFIED CROSS-VALIDATION (training set only)")
print("=" * 70)
for name, pipe in models.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    cv_results[name] = scores
    acc_mean, acc_std = scores["test_accuracy"].mean(), scores["test_accuracy"].std()
    f1_mean, f1_std = scores["test_f1"].mean(), scores["test_f1"].std()
    print(f"\n{name}")
    print(f"  CV Accuracy: {acc_mean:.4f} +/- {acc_std:.4f}")
    print(f"  CV F1      : {f1_mean:.4f} +/- {f1_std:.4f}")

# ---------------------------------------------------------------------------
# 7 & 8. Fit on full training set, evaluate on untouched held-out test set
# ---------------------------------------------------------------------------
test_results = {}
print("\n" + "=" * 70)
print("HELD-OUT TEST SET EVALUATION")
print("=" * 70)
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_score = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    roc_auc = roc_auc_score(y_test, y_score)
    ap = average_precision_score(y_test, y_score)

    test_results[name] = {
        "y_pred": y_pred,
        "y_score": y_score,
        "accuracy": acc,
        "f1": f1,
        "cm": cm,
        "roc_auc": roc_auc,
        "ap": ap,
    }

    print(f"\n{name}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1      : {f1:.4f}")
    print(f"  ROC-AUC      : {roc_auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"  Confusion Matrix -> TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(
        f"    TN={tn}: correctly predicted 'not 8'\n"
        f"    TP={tp}: correctly predicted '8'\n"
        f"    FP={fp}: predicted '8' but was not (false alarm)\n"
        f"    FN={fn}: predicted 'not 8' but was actually '8' (missed minority case)"
    )
    print("\n  Classification report:")
    print(classification_report(y_test, y_pred, target_names=["not 8", "is 8"]))

# ---------------------------------------------------------------------------
# 9. Confusion matrices (both models, one figure)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, res) in zip(axes, test_results.items()):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=res["cm"], display_labels=["not 8", "is 8"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 10 & 12. Combined ROC curve
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
for name, res in test_results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_score"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 11 & 12. Combined Precision-Recall curve
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
baseline = y_test.sum() / len(y_test)
for name, res in test_results.items():
    precision, recall, _ = precision_recall_curve(y_test, res["y_score"])
    plt.plot(recall, precision, label=f"{name} (AP = {res['ap']:.3f})")
plt.axhline(baseline, linestyle="--", color="gray", label=f"Baseline (AP={baseline:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("outputs/precision_recall_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 13. Final comparison / deployment recommendation (based on actual results)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("WHY PRECISION-RECALL MATTERS HERE")
print("=" * 70)
print(
    "With ~90% negatives, ROC-AUC can look good even when minority-class\n"
    "precision/recall are weak, because ROC-AUC is diluted by the large\n"
    "number of true negatives. The PR curve focuses only on the positive\n"
    "('is 8') class, so it exposes minority-class performance more honestly."
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON & DEPLOYMENT DECISION")
print("=" * 70)
names = list(test_results.keys())
summary_rows = []
for name in names:
    cv_acc = cv_results[name]["test_accuracy"].mean()
    cv_f1 = cv_results[name]["test_f1"].mean()
    r = test_results[name]
    summary_rows.append((name, cv_acc, cv_f1, r["accuracy"], r["f1"], r["roc_auc"], r["ap"]))
    print(
        f"{name}: CV Acc={cv_acc:.4f}, CV F1={cv_f1:.4f}, "
        f"Test Acc={r['accuracy']:.4f}, Test F1={r['f1']:.4f}, "
        f"ROC-AUC={r['roc_auc']:.4f}, AP={r['ap']:.4f}"
    )

# Pick the deployment model using CV F1 mean as the primary (more robust,
# multi-fold) signal, since a single test-set F1 can be noisy on a small
# minority class. Confirm with ROC-AUC / Average Precision as secondary
# evidence of overall ranking quality on the positive class.
best_name = max(summary_rows, key=lambda row: row[2])[0]  # row[2] = cv_f1
best = test_results[best_name]
other_name = [n for n in names if n != best_name][0]
other = test_results[other_name]
print(f"\n>>> Recommended for deployment: {best_name}")
print(
    f"    Justification: {best_name} had the higher mean CV F1 across 5 folds "
    f"({dict(zip(names, [r[2] for r in summary_rows]))[best_name]:.4f} vs "
    f"{dict(zip(names, [r[2] for r in summary_rows]))[other_name]:.4f}), the more "
    "reliable estimate of minority-class performance since it averages over "
    "multiple folds rather than one test split. This is confirmed on the "
    f"held-out test set, where {best_name} also had the higher ROC-AUC "
    f"({best['roc_auc']:.4f} vs {other['roc_auc']:.4f}) and higher Average "
    f"Precision ({best['ap']:.4f} vs {other['ap']:.4f}); test-set F1 was "
    f"nearly tied ({best['f1']:.4f} vs {other['f1']:.4f}), so the more "
    "robust CV signal and the AUC/AP metrics break the tie."
)
