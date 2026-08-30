import numpy as np
import json
from pathlib import Path

from scipy.signal import welch
from scipy.stats import skew, kurtosis

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
FS = 256

# P300 analysis window inside the epoch
# Epoch = -200 ms → +800 ms
P300_START_MS = 200
P300_END_MS = 800

P300_START = int(P300_START_MS / 1000 * FS)
P300_END = int(P300_END_MS / 1000 * FS)

RANDOM_STATE = 42

DATA_DIR = PROJECT_ROOT / "data" / "preprocessed_sessions"
OUTPUT_DIR = PROJECT_ROOT / "results" / "enhanced_pca_svm"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(eeg):
    """
    Enhanced EEG feature extraction.

    Input:
        eeg: (n_windows, 16, 256)

    Output:
        feature matrix

    Features per channel:

        1. Mean
        2. Standard deviation
        3. Maximum
        4. Minimum
        5. Peak-to-peak
        6. Variance
        7. RMS
        8. Skewness
        9. Kurtosis
        10. Line length

        11-14. Absolute band power
              Delta
              Theta
              Alpha
              Beta

        15-18. Relative band power
              Delta
              Theta
              Alpha
              Beta

        19. P300 peak amplitude
        20. P300 peak latency
        21. P300 area

        22. Hjorth activity
        23. Hjorth mobility
        24. Hjorth complexity

    Total:
        24 features/channel
        × 16 channels
        = 384 features
    """

    bands = [
        (0.5, 4.0),     # Delta
        (4.0, 8.0),     # Theta
        (8.0, 12.0),    # Alpha
        (12.0, 30.0)    # Beta
    ]

    features = []

    for window in eeg:

        window_features = []

        for channel in window:

            # =================================================
            # TIME DOMAIN
            # =================================================

            mean_value = np.mean(channel)
            std_value = np.std(channel)
            max_value = np.max(channel)
            min_value = np.min(channel)
            ptp_value = np.ptp(channel)
            variance = np.var(channel)

            rms = np.sqrt(
                np.mean(channel ** 2)
            )

            skewness = skew(
                channel,
                bias=False
            )

            kurt = kurtosis(
                channel,
                bias=False
            )

            line_length = np.sum(
                np.abs(np.diff(channel))
            )

            window_features.extend([
                mean_value,
                std_value,
                max_value,
                min_value,
                ptp_value,
                variance,
                rms,
                skewness,
                kurt,
                line_length
            ])

            # =================================================
            # FREQUENCY DOMAIN
            # =================================================

            freqs, psd = welch(
                channel,
                fs=FS,
                nperseg=min(256, len(channel))
            )

            absolute_band_powers = []

            for low, high in bands:

                mask = (
                    (freqs >= low) &
                    (freqs < high)
                )

                if np.any(mask):

                    power = np.trapezoid(
                        psd[mask],
                        freqs[mask]
                    )

                else:

                    power = 0.0

                absolute_band_powers.append(
                    power
                )

            total_power = np.sum(
                absolute_band_powers
            )

            # Absolute power
            window_features.extend(
                absolute_band_powers
            )

            # Relative power
            if total_power > 0:

                relative_band_powers = [
                    power / total_power
                    for power
                    in absolute_band_powers
                ]

            else:

                relative_band_powers = [
                    0.0
                ] * 4

            window_features.extend(
                relative_band_powers
            )

            # =================================================
            # P300 / ERP FEATURES
            # =================================================

            p300_segment = channel[
                P300_START:P300_END
            ]

            if len(p300_segment) > 0:

                # P300 is generally a positive ERP peak
                peak_index = np.argmax(
                    p300_segment
                )

                peak_amplitude = (
                    p300_segment[peak_index]
                )

                peak_latency = (
                    P300_START +
                    peak_index
                ) / FS * 1000

                area = np.trapezoid(
                    p300_segment
                )

            else:

                peak_amplitude = 0.0
                peak_latency = 0.0
                area = 0.0

            window_features.extend([
                peak_amplitude,
                peak_latency,
                area
            ])

            # =================================================
            # HJORTH PARAMETERS
            # =================================================

            activity = np.var(channel)

            first_derivative = np.diff(
                channel
            )

            second_derivative = np.diff(
                first_derivative
            )

            variance_d1 = np.var(
                first_derivative
            )

            variance_d2 = np.var(
                second_derivative
            )

            if activity > 0:

                mobility = np.sqrt(
                    variance_d1 /
                    activity
                )

            else:

                mobility = 0.0

            if variance_d1 > 0:

                mobility_d1 = np.sqrt(
                    variance_d2 /
                    variance_d1
                )

            else:

                mobility_d1 = 0.0

            if mobility > 0:

                complexity = (
                    mobility_d1 /
                    mobility
                )

            else:

                complexity = 0.0

            window_features.extend([
                activity,
                mobility,
                complexity
            ])

        features.append(
            window_features
        )

    return np.asarray(
        features,
        dtype=np.float64
    )


# ============================================================
# CREATE AUTHENTICATION WINDOWS
# ============================================================

def create_windows(epochs):

    n_complete = (
        epochs.shape[0] //
        EPOCHS_PER_WINDOW
    )

    usable_epochs = (
        n_complete *
        EPOCHS_PER_WINDOW
    )

    epochs = epochs[
        :usable_epochs
    ]

    grouped = epochs.reshape(
        n_complete,
        EPOCHS_PER_WINDOW,
        epochs.shape[1],
        epochs.shape[2]
    )

    # Average the 10 P300 epochs
    windows = grouped.mean(
        axis=1
    )

    return windows


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ENHANCED PCA + SVM CROSS-SESSION EEG AUTHENTICATION")
print("=" * 70)

print("\nConfiguration:")
print("Subjects:", SUBJECTS)
print("Recordings:", RECORDINGS)
print(
    "Epochs per authentication window:",
    EPOCHS_PER_WINDOW
)
print(
    "PCA components:",
    N_PCA_COMPONENTS
)
print(
    "P300 analysis window:",
    f"{P300_START_MS}-{P300_END_MS} ms"
)

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

        epochs = np.load(
            file_path
        )

        windows = create_windows(
            epochs
        )

        print(
            f"{subject} | R{recording} | "
            f"epochs={epochs.shape} | "
            f"windows={windows.shape}"
        )

        all_data[
            subject
        ][recording] = windows


# ============================================================
# EER
# ============================================================

def compute_eer(
    genuine_scores,
    impostor_scores
):

    genuine_scores = np.asarray(
        genuine_scores
    )

    impostor_scores = np.asarray(
        impostor_scores
    )

    scores = np.concatenate([
        genuine_scores,
        impostor_scores
    ])

    labels = np.concatenate([
        np.ones(
            len(genuine_scores)
        ),
        np.zeros(
            len(impostor_scores)
        )
    ])

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    fnr = 1 - tpr

    index = np.argmin(
        np.abs(fpr - fnr)
    )

    eer = (
        fpr[index] +
        fnr[index]
    ) / 2

    threshold = thresholds[index]

    return (
        eer,
        threshold,
        fpr,
        tpr
    )


# ============================================================
# ONE CROSS-SESSION EXPERIMENT
# ============================================================

def run_experiment(
    enrollment_recording,
    test_recording
):

    print("\n" + "=" * 70)

    print(
        f"ENROLLMENT: R{enrollment_recording} "
        f"→ TEST: R{test_recording}"
    )

    print("=" * 70)

    # ========================================================
    # ENROLLMENT DATA
    # ========================================================

    train_windows = []
    train_labels = []

    for subject_index, subject in enumerate(
        SUBJECTS
    ):

        windows = all_data[
            subject
        ][enrollment_recording]

        train_windows.append(
            windows
        )

        train_labels.extend(
            [subject_index] *
            len(windows)
        )

    train_windows = np.concatenate(
        train_windows,
        axis=0
    )

    train_labels = np.asarray(
        train_labels
    )

    print(
        "\nExtracting enhanced features "
        "from enrollment data..."
    )

    X_train = extract_features(
        train_windows
    )

    print(
        "Enhanced feature matrix:",
        X_train.shape
    )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    expected_features = 384

    if X_train.shape[1] != expected_features:

        raise RuntimeError(
            f"Expected {expected_features} "
            f"features but got "
            f"{X_train.shape[1]}"
        )

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    print(
        "\nFitting StandardScaler "
        "on enrollment data..."
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # ========================================================
    # PCA
    # ========================================================

    print(
        f"Applying PCA "
        f"({X_train.shape[1]} → "
        f"{N_PCA_COMPONENTS})..."
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
        "PCA explained variance:",
        f"{explained_variance * 100:.2f}%"
    )

    # ========================================================
    # TRAIN ONE-VS-REST SVM PER SUBJECT
    # ========================================================

    genuine_scores = []
    impostor_scores = []

    for claimed_index, claimed_subject in enumerate(
        SUBJECTS
    ):

        print(
            f"\nClaimed identity: "
            f"{claimed_subject}"
        )

        binary_labels = (
            train_labels ==
            claimed_index
        ).astype(int)

        svm = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

        svm.fit(
            X_train_pca,
            binary_labels
        )

        # ====================================================
        # TEST ALL SUBJECTS
        # ====================================================

        for test_index, test_subject in enumerate(
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

            if (
                test_index ==
                claimed_index
            ):

                genuine_scores.extend(
                    scores.tolist()
                )

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

    eer, threshold, fpr, tpr = compute_eer(
        genuine_scores,
        impostor_scores
    )

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
        f"Genuine mean : "
        f"{np.mean(genuine_scores):.4f}"
    )

    print(
        f"Impostor mean: "
        f"{np.mean(impostor_scores):.4f}"
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
        "pca_explained_variance": float(
            explained_variance
        ),
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "fpr": fpr,
        "tpr": tpr
    }


# ============================================================
# RUN ALL CROSS-SESSION PAIRS
# ============================================================

recording_pairs = [
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 3),
    (3, 1),
    (3, 2)
]

pair_results = {}

all_genuine = []
all_impostor = []


for enrollment, test in recording_pairs:

    key = f"R{enrollment}_R{test}"

    result = run_experiment(
        enrollment,
        test
    )

    pair_results[key] = result

    all_genuine.extend(
        result["genuine_scores"]
    )

    all_impostor.extend(
        result["impostor_scores"]
    )


# ============================================================
# OVERALL RESULTS
# ============================================================

all_genuine = np.asarray(
    all_genuine
)

all_impostor = np.asarray(
    all_impostor
)

(
    overall_eer,
    overall_threshold,
    overall_fpr,
    overall_tpr
) = compute_eer(
    all_genuine,
    all_impostor
)

overall_far = np.mean(
    all_impostor >=
    overall_threshold
)

overall_frr = np.mean(
    all_genuine <
    overall_threshold
)

roc_auc = auc(
    overall_fpr,
    overall_tpr
)


print("\n" + "=" * 70)
print("OVERALL ENHANCED PCA + SVM RESULT")
print("=" * 70)

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

print(
    f"ROC AUC           : "
    f"{roc_auc:.4f}"
)


# ============================================================
# COMPARE WITH CURRENT BEST
# ============================================================

CURRENT_BEST_EER = 0.1575

absolute_change = (
    CURRENT_BEST_EER -
    overall_eer
)

relative_change = (
    absolute_change /
    CURRENT_BEST_EER
) * 100


print("\n" + "=" * 70)
print("COMPARISON WITH CURRENT BEST")
print("=" * 70)

print(
    f"\nExisting PCA + SVM : "
    f"{CURRENT_BEST_EER * 100:.2f}%"
)

print(
    f"Enhanced PCA + SVM : "
    f"{overall_eer * 100:.2f}%"
)

print(
    f"Absolute EER change: "
    f"{absolute_change * 100:+.2f} percentage points"
)

print(
    f"Relative change    : "
    f"{relative_change:+.2f}%"
)


# ============================================================
# SAVE SCORES
# ============================================================

np.save(
    OUTPUT_DIR /
    "genuine_scores.npy",
    all_genuine
)

np.save(
    OUTPUT_DIR /
    "impostor_scores.npy",
    all_impostor
)


# ============================================================
# SAVE JSON
# ============================================================

json_results = {

    "method":
        "Enhanced PCA + RBF SVM",

    "subjects":
        SUBJECTS,

    "recordings":
        RECORDINGS,

    "epochs_per_window":
        EPOCHS_PER_WINDOW,

    "original_features":
        224,

    "enhanced_features":
        384,

    "pca_components":
        N_PCA_COMPONENTS,

    "overall": {

        "eer":
            float(overall_eer),

        "threshold":
            float(overall_threshold),

        "far":
            float(overall_far),

        "frr":
            float(overall_frr),

        "roc_auc":
            float(roc_auc),

        "genuine_mean":
            float(np.mean(all_genuine)),

        "genuine_std":
            float(np.std(all_genuine)),

        "impostor_mean":
            float(np.mean(all_impostor)),

        "impostor_std":
            float(np.std(all_impostor))
    },

    "comparison": {

        "previous_eer":
            CURRENT_BEST_EER,

        "absolute_change":
            float(absolute_change),

        "relative_change":
            float(relative_change / 100)
    },

    "recording_pairs": {}
}


for key, result in pair_results.items():

    json_results[
        "recording_pairs"
    ][key] = {

        "eer":
            result["eer"],

        "threshold":
            result["threshold"],

        "far":
            result["far"],

        "frr":
            result["frr"],

        "genuine_mean":
            result["genuine_mean"],

        "genuine_std":
            result["genuine_std"],

        "impostor_mean":
            result["impostor_mean"],

        "impostor_std":
            result["impostor_std"],

        "pca_explained_variance":
            result[
                "pca_explained_variance"
            ]
    }


with open(
    OUTPUT_DIR /
    "enhanced_pca_svm_results.json",
    "w"
) as f:

    json.dump(
        json_results,
        f,
        indent=2
    )


# ============================================================
# GRAPH 1 — SCORE DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    all_genuine,
    bins=60,
    alpha=0.6,
    label="Genuine"
)

plt.hist(
    all_impostor,
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

plt.xlabel(
    "SVM Decision Score"
)

plt.ylabel(
    "Number of Authentication Windows"
)

plt.title(
    "Enhanced PCA + SVM Score Distribution"
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
# GRAPH 2 — ROC
# ============================================================

plt.figure(
    figsize=(8, 7)
)

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

plt.xlabel(
    "False Acceptance Rate"
)

plt.ylabel(
    "True Acceptance Rate"
)

plt.title(
    "Enhanced PCA + SVM ROC Curve"
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
# GRAPH 3 — FAR / FRR
# ============================================================

threshold_range = np.linspace(
    min(
        np.min(all_genuine),
        np.min(all_impostor)
    ),
    max(
        np.max(all_genuine),
        np.max(all_impostor)
    ),
    500
)

far_values = []
frr_values = []

for threshold in threshold_range:

    far_values.append(
        np.mean(
            all_impostor >= threshold
        )
    )

    frr_values.append(
        np.mean(
            all_genuine < threshold
        )
    )


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    threshold_range,
    far_values,
    label="FAR",
    linewidth=2
)

plt.plot(
    threshold_range,
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

plt.xlabel(
    "Authentication Threshold"
)

plt.ylabel(
    "Error Rate"
)

plt.title(
    "Enhanced PCA + SVM FAR / FRR"
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
# GRAPH 4 — EER BY RECORDING PAIR
# ============================================================

pair_names = list(
    pair_results.keys()
)

pair_eers = [
    pair_results[key]["eer"] * 100
    for key in pair_names
]

plt.figure(
    figsize=(10, 6)
)

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

plt.ylabel(
    "EER (%)"
)

plt.title(
    "Enhanced PCA + SVM Cross-Session EER"
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
print("ENHANCED EXPERIMENT COMPLETE")
print("=" * 70)

print(
    "\nResults directory:"
)

print(
    OUTPUT_DIR
)

print("\nGenerated files:")

print(
    "✓ enhanced_pca_svm_results.json"
)

print(
    "✓ genuine_scores.npy"
)

print(
    "✓ impostor_scores.npy"
)

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

print("\n" + "=" * 70)
