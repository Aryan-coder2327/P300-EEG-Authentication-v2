import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "pca_svm_cross_validation"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "cross_validation_results.json"
)


# ============================================================
# LOAD RESULTS
# ============================================================

with open(RESULTS_FILE, "r") as f:
    results = json.load(f)


folds = results["folds"]
overall = results["overall"]


# ============================================================
# 1. EER BY FOLD
# ============================================================

fold_numbers = [
    fold["fold"]
    for fold in folds
]

test_eers = [
    fold["test_eer"] * 100
    for fold in folds
]

plt.figure(figsize=(10, 6))

plt.bar(
    fold_numbers,
    test_eers
)

plt.axhline(
    overall["mean_test_eer"] * 100,
    linestyle="--",
    label=(
        f'Mean EER = '
        f'{overall["mean_test_eer"] * 100:.2f}%'
    )
)

plt.xlabel("Cross-validation fold")
plt.ylabel("Test EER (%)")
plt.title(
    "PCA + SVM Cross-Session EER by Fold"
)

plt.xticks(fold_numbers)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "cv_eer_by_fold.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. ROC-AUC BY FOLD
# ============================================================

test_aucs = [
    fold["test_auc"]
    for fold in folds
]

plt.figure(figsize=(10, 6))

plt.bar(
    fold_numbers,
    test_aucs
)

plt.axhline(
    overall["mean_test_auc"],
    linestyle="--",
    label=(
        f'Mean AUC = '
        f'{overall["mean_test_auc"]:.4f}'
    )
)

plt.xlabel("Cross-validation fold")
plt.ylabel("ROC-AUC")
plt.title(
    "PCA + SVM ROC-AUC Across Test Sessions"
)

plt.xticks(fold_numbers)

plt.ylim(
    0,
    1
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "cv_auc_by_fold.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. VALIDATION EER VS FINAL TEST EER
# ============================================================

validation_eers = [
    fold["validation_eer"] * 100
    for fold in folds
]

plt.figure(figsize=(10, 6))

x = np.arange(
    len(folds)
)

width = 0.35

plt.bar(
    x - width / 2,
    validation_eers,
    width,
    label="Validation EER"
)

plt.bar(
    x + width / 2,
    test_eers,
    width,
    label="Unseen Test EER"
)

plt.xlabel("Cross-validation fold")
plt.ylabel("EER (%)")

plt.title(
    "Validation vs Unseen Test EER"
)

plt.xticks(
    x,
    [
        f"Fold {n}"
        for n in fold_numbers
    ]
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "cv_validation_vs_test_eer.png",
    dpi=300
)

plt.close()


# ============================================================
# 4. SESSION GENERALIZATION
# ============================================================

labels = []

for fold in folds:

    labels.append(
        f'R{fold["enrollment_recording"]}'
        f'→R{fold["test_recording"]}'
    )


plt.figure(figsize=(11, 6))

plt.bar(
    labels,
    test_eers
)

plt.axhline(
    overall["mean_test_eer"] * 100,
    linestyle="--",
    label=(
        f'Mean = '
        f'{overall["mean_test_eer"] * 100:.2f}%'
    )
)

plt.xlabel(
    "Enrollment session → unseen test session"
)

plt.ylabel("Test EER (%)")

plt.title(
    "Cross-Session Authentication Performance"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "cv_session_generalization.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. SUMMARY
# ============================================================

print("=" * 70)
print("FINAL CROSS-VALIDATION GRAPHS GENERATED")
print("=" * 70)

print()

print(
    f'Mean Test EER : '
    f'{overall["mean_test_eer"] * 100:.2f}%'
)

print(
    f'Std Test EER  : '
    f'{overall["std_test_eer"] * 100:.2f}%'
)

print(
    f'Mean ROC-AUC  : '
    f'{overall["mean_test_auc"]:.4f}'
)

print(
    f'Std ROC-AUC   : '
    f'{overall["std_test_auc"]:.4f}'
)

print()

print("Generated:")

print("✓ cv_eer_by_fold.png")
print("✓ cv_auc_by_fold.png")
print("✓ cv_validation_vs_test_eer.png")
print("✓ cv_session_generalization.png")

print()
print(
    f"Location: {RESULTS_DIR}"
)
