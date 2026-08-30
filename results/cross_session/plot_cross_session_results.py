import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "cross_session"

# ============================================================
# LOAD RESULTS
# ============================================================

genuine_scores = np.load(
    RESULTS_DIR / "genuine_scores.npy"
)

impostor_scores = np.load(
    RESULTS_DIR / "impostor_scores.npy"
)

with open(
    RESULTS_DIR / "cross_session_results.json",
    "r"
) as f:
    results = json.load(f)

overall = results["overall_result"]
pair_results = results["session_pair_results"]

print("=" * 70)
print("CROSS-SESSION AUTHENTICATION DIAGNOSTICS")
print("=" * 70)

print("\nScore statistics")

print(
    f"Genuine  : mean={genuine_scores.mean():.4f}, "
    f"std={genuine_scores.std():.4f}"
)

print(
    f"Impostor : mean={impostor_scores.mean():.4f}, "
    f"std={impostor_scores.std():.4f}"
)

print(
    f"\nOverall EER       : "
    f"{overall['eer'] * 100:.2f}%"
)

print(
    f"Overall threshold : "
    f"{overall['threshold']:.4f}"
)

print(
    f"Overall FAR       : "
    f"{overall['far'] * 100:.2f}%"
)

print(
    f"Overall FRR       : "
    f"{overall['frr'] * 100:.2f}%"
)


# ============================================================
# PLOT 1 — SCORE DISTRIBUTIONS
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    genuine_scores,
    bins=40,
    alpha=0.6,
    label="Genuine"
)

plt.hist(
    impostor_scores,
    bins=40,
    alpha=0.6,
    label="Impostor"
)

plt.axvline(
    overall["threshold"],
    linestyle="--",
    linewidth=2,
    label=(
        f"EER threshold "
        f"({overall['threshold']:.4f})"
    )
)

plt.xlabel("Cosine Similarity Score")
plt.ylabel("Number of Authentication Windows")

plt.title(
    "Cross-Session Genuine vs Impostor Scores"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "score_distribution.png",
    dpi=200
)

plt.show()


# ============================================================
# ROC DATA
# ============================================================

thresholds = np.linspace(
    0,
    1,
    1000
)

far_values = []
tpr_values = []

for threshold in thresholds:

    false_accepts = np.sum(
        impostor_scores >= threshold
    )

    impostor_total = len(
        impostor_scores
    )

    far = (
        false_accepts
        / impostor_total
    )

    true_accepts = np.sum(
        genuine_scores >= threshold
    )

    genuine_total = len(
        genuine_scores
    )

    tpr = (
        true_accepts
        / genuine_total
    )

    far_values.append(far)
    tpr_values.append(tpr)

far_values = np.asarray(far_values)
tpr_values = np.asarray(tpr_values)


# ============================================================
# PLOT 2 — ROC CURVE
# ============================================================

plt.figure(figsize=(8, 7))

plt.plot(
    far_values,
    tpr_values,
    linewidth=2,
    label="Cross-session ROC"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Chance"
)

plt.xlabel("False Acceptance Rate")
plt.ylabel("True Acceptance Rate")

plt.title(
    "Cross-Session ROC Curve"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "roc_curve.png",
    dpi=200
)

plt.show()


# ============================================================
# PLOT 3 — FAR / FRR VS THRESHOLD
# ============================================================

frr_values = 1 - tpr_values

plt.figure(figsize=(10, 6))

plt.plot(
    thresholds,
    far_values,
    linewidth=2,
    label="FAR"
)

plt.plot(
    thresholds,
    frr_values,
    linewidth=2,
    label="FRR"
)

plt.axvline(
    overall["threshold"],
    linestyle="--",
    linewidth=2,
    label=(
        f"EER threshold "
        f"({overall['threshold']:.4f})"
    )
)

plt.xlabel("Authentication Threshold")
plt.ylabel("Error Rate")

plt.title(
    "Cross-Session FAR and FRR"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "far_frr_curve.png",
    dpi=200
)

plt.show()


# ============================================================
# PLOT 4 — EER FOR EACH RECORDING PAIR
# ============================================================

labels = []
eers = []

for result in pair_results:

    label = (
        f"R{result['enrollment_recording']}"
        f"→R{result['test_recording']}"
    )

    labels.append(label)

    eers.append(
        result["eer"] * 100
    )

plt.figure(figsize=(10, 6))

plt.bar(
    labels,
    eers
)

plt.axhline(
    overall["eer"] * 100,
    linestyle="--",
    linewidth=2,
    label=(
        f"Overall EER "
        f"({overall['eer'] * 100:.2f}%)"
    )
)

plt.xlabel("Enrollment → Test Recording")
plt.ylabel("EER (%)")

plt.title(
    "Cross-Session EER by Recording Pair"
)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "eer_by_recording_pair.png",
    dpi=200
)

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GENERATED FILES")
print("=" * 70)

print(
    "✓ score_distribution.png"
)

print(
    "✓ roc_curve.png"
)

print(
    "✓ far_frr_curve.png"
)

print(
    "✓ eer_by_recording_pair.png"
)

print(
    f"\nOverall cross-session EER: "
    f"{overall['eer'] * 100:.2f}%"
)

print("=" * 70)

