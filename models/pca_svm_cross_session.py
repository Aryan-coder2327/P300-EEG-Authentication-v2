import numpy as np
import json
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUBJECTS = [f"A{i:02d}" for i in range(1, 11)]
RECORDINGS = [1, 2, 3]

EPOCHS_PER_WINDOW = 10
N_PCA_COMPONENTS = 50

RANDOM_STATE = 42

DATA_DIR = PROJECT_ROOT / "data" / "preprocessed_sessions"
OUTPUT_DIR = PROJECT_ROOT / "results" / "pca_svm"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(eeg):
    """
    Extract 224 features from EEG windows.

    Input:
        eeg: (n_windows, 16, 256)

    Output:
        features: (n_windows, 224)

    224 features =
        16 channels ×
        (6 temporal + 4 spectral + 4 frequency/FFT features)
    """

    from scipy.signal import welch
    from scipy.fft import rfft, rfftfreq

    features = []

    bands = [
        (0.5, 4.0),    # Delta
        (4.0, 8.0),    # Theta
        (8.0, 12.0),   # Alpha
        (12.0, 30.0)   # Beta
    ]

    fs = 256

    for window in eeg:

        window_features = []

        for channel in window:

            # ------------------------------------------------
            # TEMPORAL FEATURES
            # ------------------------------------------------

            mean = np.mean(channel)
            std = np.std(channel)
            maximum = np.max(channel)
            minimum = np.min(channel)
            ptp = np.ptp(channel)
            variance = np.var(channel)

            window_features.extend([
                mean,
                std,
                maximum,
                minimum,
                ptp,
                variance
            ])

            # ------------------------------------------------
            # SPECTRAL FEATURES
            # ------------------------------------------------

            freqs, psd = welch(
                channel,
                fs=fs,
                nperseg=min(256, len(channel))
            )

            for low, high in bands:

                mask = (freqs >= low) & (freqs < high)

                if np.any(mask):
                    band_power = np.mean(psd[mask])
                else:
                    band_power = 0.0

                window_features.append(band_power)

            # ------------------------------------------------
            # FFT FEATURES
            # ------------------------------------------------

            fft_values = np.abs(rfft(channel))
            fft_freqs = rfftfreq(len(channel), 1 / fs)

            fft_bands = [
                (0.5, 4.0),
                (4.0, 8.0),
                (8.0, 12.0),
                (12.0, 30.0)
            ]

            for low, high in fft_bands:

                mask = (
                    (fft_freqs >= low) &
                    (fft_freqs < high)
                )

                if np.any(mask):
                    fft_feature = np.mean(fft_values[mask])
                else:
                    fft_feature = 0.0

                window_features.append(fft_feature)

        features.append(window_features)

    return np.asarray(features)


# ============================================================
# CREATE AUTHENTICATION WINDOWS
# ============================================================

def create_windows(epochs):
    """
    Combine 10 consecutive P300 epochs into one
    authentication window.

    Input:
        (1536, 16, 256)

    Output:
        (153, 16, 256)
    """

    n_complete = (
        epochs.shape[0] // EPOCHS_PER_WINDOW
    )

    usable_epochs = (
        n_complete * EPOCHS_PER_WINDOW
    )

    epochs = epochs[:usable_epochs]

    # Group into windows
    grouped = epochs.reshape(
        n_complete,
        EPOCHS_PER_WINDOW,
        epochs.shape[1],
        epochs.shape[2]
    )

    # Average the 10 epochs
    windows = grouped.mean(axis=1)

    return windows


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PCA + SVM CROSS-SESSION EEG AUTHENTICATION")
print("=" * 70)

print("\nConfiguration:")
print("Subjects:", SUBJECTS)
print("Recordings:", RECORDINGS)
print("Epochs per authentication window:", EPOCHS_PER_WINDOW)
print("PCA components:", N_PCA_COMPONENTS)


all_data = {}

print("\n" + "=" * 70)
print("LOADING PREPROCESSED DATA")
print("=" * 70)

for subject in SUBJECTS:

    all_data[subject] = {}

    for recording in RECORDINGS:

        file_path = (
            DATA_DIR /
            subject /
            f"recording_{recording}_epochs.npy"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file:\n{file_path}"
            )

        epochs = np.load(file_path)

        print(
            f"{subject} | Recording {recording} | "
            f"Epochs: {epochs.shape}"
        )

        windows = create_windows(epochs)

        print(
            f"    Authentication windows: "
            f"{windows.shape}"
        )

        all_data[subject][recording] = windows


# ============================================================
# EER CALCULATION
# ============================================================

def compute_eer(genuine_scores, impostor_scores):

    """
    Compute EER using arbitrary SVM decision scores.

    Unlike the old baseline implementation, this does NOT
    assume scores are restricted to 0-1.
    """

    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)

    scores = np.concatenate([
        genuine_scores,
        impostor_scores
    ])

    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    fnr = 1 - tpr

    difference = np.abs(
        fpr - fnr
    )

    index = np.argmin(difference)

    eer = (
        fpr[index] +
        fnr[index]
    ) / 2

    threshold = thresholds[index]

    return eer, threshold, fpr, tpr


# ============================================================
# TRAIN / TEST ONE RECORDING PAIR
# ============================================================

def run_experiment(
    enrollment_recording,
    test_recording
):

    print("\n" + "=" * 70)
    print(
        f"ENROLLMENT: Recording {enrollment_recording} "
        f"→ TEST: Recording {test_recording}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Build enrollment dataset
    # --------------------------------------------------------

    train_windows = []
    train_labels = []

    for subject_index, subject in enumerate(SUBJECTS):

        windows = all_data[
            subject
        ][enrollment_recording]

        train_windows.append(windows)

        train_labels.extend(
            [subject_index] * len(windows)
        )

    train_windows = np.concatenate(
        train_windows,
        axis=0
    )

    train_labels = np.asarray(
        train_labels
    )

    print(
        "\nEnrollment windows:",
        train_windows.shape
    )

    print(
        "Enrollment labels:",
        train_labels.shape
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print("\nExtracting enrollment features...")

    X_train = extract_features(
        train_windows
    )

    print(
        "Enrollment feature matrix:",
        X_train.shape
    )

    # --------------------------------------------------------
    # Fit scaler ONLY on enrollment data
    # --------------------------------------------------------

    print("\nFitting StandardScaler...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # --------------------------------------------------------
    # Fit PCA ONLY on enrollment data
    # --------------------------------------------------------

    print(
        f"\nApplying PCA "
        f"(224 → {N_PCA_COMPONENTS})..."
    )

    pca = PCA(
        n_components=N_PCA_COMPONENTS,
        random_state=RANDOM_STATE
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )

    print(
        f"PCA explained variance: "
        f"{explained_variance * 100:.2f}%"
    )

    print(
        "PCA training shape:",
        X_train_pca.shape
    )

    # --------------------------------------------------------
    # Prepare test data
    # --------------------------------------------------------

    genuine_scores = []
    impostor_scores = []

    pair_predictions = []

    print(
        "\nTraining one-vs-rest SVM authentication "
        "models..."
    )

    # --------------------------------------------------------
    # One binary SVM per registered subject
    # --------------------------------------------------------

    for claimed_subject_index, claimed_subject in enumerate(
        SUBJECTS
    ):

        print(
            f"\nTesting claimed identity: "
            f"{claimed_subject}"
        )

        # ----------------------------------------------------
        # Binary labels for this claimed identity
        # ----------------------------------------------------

        y_binary = (
            train_labels ==
            claimed_subject_index
        ).astype(int)

        # ----------------------------------------------------
        # Train binary SVM
        # ----------------------------------------------------

        svm = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

        svm.fit(
            X_train_pca,
            y_binary
        )

        # ----------------------------------------------------
        # Test every subject from test session
        # ----------------------------------------------------

        for test_subject_index, test_subject in enumerate(
            SUBJECTS
        ):

            test_windows = all_data[
                test_subject
            ][test_recording]

            X_test = extract_features(
                test_windows
            )

            X_test_scaled = scaler.transform(
                X_test
            )

            X_test_pca = pca.transform(
                X_test_scaled
            )

            scores = svm.decision_function(
                X_test_pca
            )

            # ------------------------------------------------
            # Genuine
            # ------------------------------------------------

            if test_subject_index == claimed_subject_index:

                genuine_scores.extend(
                    scores.tolist()
                )

            # ------------------------------------------------
            # Impostor
            # ------------------------------------------------

            else:

                impostor_scores.extend(
                    scores.tolist()
                )

    # ========================================================
    # EER
    # ========================================================

    genuine_scores = np.asarray(
        genuine_scores
    )

    impostor_scores = np.asarray(
        impostor_scores
    )

    print("\n" + "-" * 70)
    print("SCORE STATISTICS")
    print("-" * 70)

    print(
        f"Genuine  : "
        f"mean={np.mean(genuine_scores):.4f}, "
        f"std={np.std(genuine_scores):.4f}"
    )

    print(
        f"Impostor : "
        f"mean={np.mean(impostor_scores):.4f}, "
        f"std={np.std(impostor_scores):.4f}"
    )

    eer, threshold, fpr, tpr = compute_eer(
        genuine_scores,
        impostor_scores
    )

    # --------------------------------------------------------
    # FAR / FRR at EER threshold
    # --------------------------------------------------------

    far = np.mean(
        impostor_scores >= threshold
    )

    frr = np.mean(
        genuine_scores < threshold
    )

    print("\n" + "-" * 70)
    print("RESULT")
    print("-" * 70)

    print(
        f"EER       : {eer * 100:.2f}%"
    )

    print(
        f"Threshold : {threshold:.6f}"
    )

    print(
        f"FAR       : {far * 100:.2f}%"
    )

    print(
        f"FRR       : {frr * 100:.2f}%"
    )

    print(
        f"Genuine scores  : {len(genuine_scores)}"
    )

    print(
        f"Impostor scores : {len(impostor_scores)}"
    )

    return {
        "eer": float(eer),
        "threshold": float(threshold),
        "far": float(far),
        "frr": float(frr),
        "genuine_mean": float(
            np.mean(genuine_scores)
        ),
        "genuine_std": float(
            np.std(genuine_scores)
        ),
        "impostor_mean": float(
            np.mean(impostor_scores)
        ),
        "impostor_std": float(
            np.std(impostor_scores)
        ),
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "fpr": fpr,
        "tpr": tpr,
        "pca_explained_variance": float(
            explained_variance
        )
    }


# ============================================================
# RUN ALL SIX CROSS-SESSION COMBINATIONS
# ============================================================

pair_results = {}

all_genuine_scores = []
all_impostor_scores = []

recording_pairs = [
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 3),
    (3, 1),
    (3, 2)
]

for enrollment_recording, test_recording in recording_pairs:

    key = (
        f"R{enrollment_recording}"
        f"_R{test_recording}"
    )

    result = run_experiment(
        enrollment_recording,
        test_recording
    )

    pair_results[key] = result

    all_genuine_scores.extend(
        result["genuine_scores"]
    )

    all_impostor_scores.extend(
        result["impostor_scores"]
    )


# ============================================================
# OVERALL RESULT
# ============================================================

print("\n" + "=" * 70)
print("OVERALL PCA + SVM CROSS-SESSION RESULT")
print("=" * 70)

all_genuine_scores = np.asarray(
    all_genuine_scores
)

all_impostor_scores = np.asarray(
    all_impostor_scores
)

overall_eer, overall_threshold, overall_fpr, overall_tpr = (
    compute_eer(
        all_genuine_scores,
        all_impostor_scores
    )
)

overall_far = np.mean(
    all_impostor_scores >= overall_threshold
)

overall_frr = np.mean(
    all_genuine_scores < overall_threshold
)

print(
    f"\nOverall Genuine mean : "
    f"{np.mean(all_genuine_scores):.4f}"
)

print(
    f"Overall Genuine std  : "
    f"{np.std(all_genuine_scores):.4f}"
)

print(
    f"Overall Impostor mean: "
    f"{np.mean(all_impostor_scores):.4f}"
)

print(
    f"Overall Impostor std : "
    f"{np.std(all_impostor_scores):.4f}"
)

print(
    f"\nOverall EER       : "
    f"{overall_eer * 100:.2f}%"
)

print(
    f"Overall Threshold : "
    f"{overall_threshold:.6f}"
)

print(
    f"Overall FAR       : "
    f"{overall_far * 100:.2f}%"
)

print(
    f"Overall FRR       : "
    f"{overall_frr * 100:.2f}%"
)


# ============================================================
# COMPARE WITH COSINE BASELINE
# ============================================================

BASELINE_EER = 0.4738

improvement = (
    BASELINE_EER -
    overall_eer
)

improvement_percentage = (
    improvement /
    BASELINE_EER
) * 100

print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

print(
    f"\nCosine baseline EER : "
    f"{BASELINE_EER * 100:.2f}%"
)

print(
    f"PCA + SVM EER       : "
    f"{overall_eer * 100:.2f}%"
)

print(
    f"Absolute change     : "
    f"{improvement * 100:+.2f} percentage points"
)

print(
    f"Relative improvement: "
    f"{improvement_percentage:+.2f}%"
)


# ============================================================
# SAVE NUMPY SCORES
# ============================================================

np.save(
    OUTPUT_DIR /
    "genuine_scores.npy",
    all_genuine_scores
)

np.save(
    OUTPUT_DIR /
    "impostor_scores.npy",
    all_impostor_scores
)


# ============================================================
# SAVE JSON RESULTS
# ============================================================

json_results = {
    "method": "PCA + RBF SVM",
    "subjects": SUBJECTS,
    "recordings": RECORDINGS,
    "epochs_per_authentication_window":
        EPOCHS_PER_WINDOW,
    "input_features": 224,
    "pca_components": N_PCA_COMPONENTS,
    "pca_explained_variance":
        float(
            pair_results[
                "R1_R2"
            ]["pca_explained_variance"]
        ),

    "baseline_cosine_eer":
        BASELINE_EER,

    "overall": {
        "eer": float(overall_eer),
        "threshold": float(overall_threshold),
        "far": float(overall_far),
        "frr": float(overall_frr),
        "genuine_mean":
            float(np.mean(all_genuine_scores)),
        "genuine_std":
            float(np.std(all_genuine_scores)),
        "impostor_mean":
            float(np.mean(all_impostor_scores)),
        "impostor_std":
            float(np.std(all_impostor_scores)),
        "absolute_eer_change":
            float(improvement),
        "relative_improvement":
            float(improvement_percentage / 100)
    },

    "recording_pairs": {}
}

for key, result in pair_results.items():

    json_results["recording_pairs"][key] = {
        "eer": result["eer"],
        "threshold": result["threshold"],
        "far": result["far"],
        "frr": result["frr"],
        "genuine_mean":
            result["genuine_mean"],
        "genuine_std":
            result["genuine_std"],
        "impostor_mean":
            result["impostor_mean"],
        "impostor_std":
            result["impostor_std"]
    }


with open(
    OUTPUT_DIR /
    "pca_svm_results.json",
    "w"
) as f:

    json.dump(
        json_results,
        f,
        indent=2
    )


# ============================================================
# PLOT 1 — SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    all_genuine_scores,
    bins=60,
    alpha=0.6,
    label="Genuine"
)

plt.hist(
    all_impostor_scores,
    bins=60,
    alpha=0.6,
    label="Impostor"
)

plt.axvline(
    overall_threshold,
    linestyle="--",
    linewidth=2,
    label=(
        f"EER threshold "
        f"({overall_threshold:.4f})"
    )
)

plt.xlabel("SVM Decision Score")
plt.ylabel(
    "Number of Authentication Windows"
)

plt.title(
    "PCA + SVM Genuine vs Impostor Scores"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "score_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 2 — ROC CURVE
# ============================================================

roc_auc = auc(
    overall_fpr,
    overall_tpr
)

plt.figure(figsize=(8, 7))

plt.plot(
    overall_fpr,
    overall_tpr,
    linewidth=2,
    label=f"ROC (AUC = {roc_auc:.3f})"
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
    "PCA + SVM Cross-Session ROC Curve"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "roc_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 3 — FAR / FRR
# ============================================================

thresholds = np.linspace(
    min(
        np.min(all_genuine_scores),
        np.min(all_impostor_scores)
    ),
    max(
        np.max(all_genuine_scores),
        np.max(all_impostor_scores)
    ),
    500
)

far_values = []
frr_values = []

for threshold in thresholds:

    far_values.append(
        np.mean(
            all_impostor_scores >= threshold
        )
    )

    frr_values.append(
        np.mean(
            all_genuine_scores < threshold
        )
    )

plt.figure(figsize=(10, 6))

plt.plot(
    thresholds,
    far_values,
    label="FAR",
    linewidth=2
)

plt.plot(
    thresholds,
    frr_values,
    label="FRR",
    linewidth=2
)

plt.axvline(
    overall_threshold,
    linestyle="--",
    linewidth=2,
    label=(
        f"EER threshold "
        f"({overall_threshold:.4f})"
    )
)

plt.xlabel("Authentication Threshold")
plt.ylabel("Error Rate")

plt.title(
    "PCA + SVM Cross-Session FAR and FRR"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "far_frr_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 4 — EER BY RECORDING PAIR
# ============================================================

pair_names = list(pair_results.keys())

pair_eers = [
    pair_results[key]["eer"] * 100
    for key in pair_names
]

plt.figure(figsize=(10, 6))

plt.bar(
    pair_names,
    pair_eers
)

plt.axhline(
    overall_eer * 100,
    linestyle="--",
    linewidth=2,
    label=(
        f"Overall EER "
        f"({overall_eer * 100:.2f}%)"
    )
)

plt.xlabel(
    "Enrollment → Test Recording"
)

plt.ylabel("EER (%)")

plt.title(
    "PCA + SVM Cross-Session EER by Recording Pair"
)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "eer_by_recording_pair.png",
    dpi=300
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print("\nResults saved to:")

print(
    OUTPUT_DIR /
    "pca_svm_results.json"
)

print(
    OUTPUT_DIR /
    "genuine_scores.npy"
)

print(
    OUTPUT_DIR /
    "impostor_scores.npy"
)

print("\nGenerated plots:")

print("✓ score_distribution.png")
print("✓ roc_curve.png")
print("✓ far_frr_curve.png")
print("✓ eer_by_recording_pair.png")

print("\n" + "=" * 70)
