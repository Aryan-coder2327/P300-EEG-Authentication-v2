import numpy as np
import json
from pathlib import Path

from scipy.signal import welch
from scipy.fft import rfft, rfftfreq

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUBJECTS = [f"A{i:02d}" for i in range(1, 11)]

RECORDINGS = [1, 2, 3]

EPOCHS_PER_WINDOW = 10

FS = 256

RANDOM_STATE = 42

DATA_DIR = (
    PROJECT_ROOT /
    "data" /
    "preprocessed_sessions"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "pca_svm_optimization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SEARCH SPACE
# ============================================================

PCA_COMPONENTS = [
    20,
    30,
    40,
    50,
    75,
    100
]

SVM_C_VALUES = [
    0.1,
    1.0,
    10.0,
    100.0
]

SVM_GAMMA_VALUES = [
    "scale",
    0.001,
    0.01,
    0.1
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(eeg):

    """
    Same 224-feature extractor as the established
    baseline PCA + SVM experiment.

    16 channels × 14 features = 224 features.

    Per channel:

    6 temporal:
        mean
        std
        maximum
        minimum
        peak-to-peak
        variance

    4 Welch spectral:
        delta
        theta
        alpha
        beta

    4 FFT:
        delta
        theta
        alpha
        beta
    """

    bands = [
        (0.5, 4.0),
        (4.0, 8.0),
        (8.0, 12.0),
        (12.0, 30.0)
    ]

    features = []

    for window in eeg:

        window_features = []

        for channel in window:

            # ------------------------------------------------
            # TEMPORAL
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
            # WELCH SPECTRAL FEATURES
            # ------------------------------------------------

            freqs, psd = welch(
                channel,
                fs=FS,
                nperseg=min(
                    256,
                    len(channel)
                )
            )

            for low, high in bands:

                mask = (
                    (freqs >= low) &
                    (freqs < high)
                )

                if np.any(mask):

                    band_power = np.mean(
                        psd[mask]
                    )

                else:

                    band_power = 0.0

                window_features.append(
                    band_power
                )

            # ------------------------------------------------
            # FFT FEATURES
            # ------------------------------------------------

            fft_values = np.abs(
                rfft(channel)
            )

            fft_freqs = rfftfreq(
                len(channel),
                1 / FS
            )

            for low, high in bands:

                mask = (
                    (fft_freqs >= low) &
                    (fft_freqs < high)
                )

                if np.any(mask):

                    fft_feature = np.mean(
                        fft_values[mask]
                    )

                else:

                    fft_feature = 0.0

                window_features.append(
                    fft_feature
                )

        features.append(
            window_features
        )

    return np.asarray(
        features,
        dtype=np.float64
    )


# ============================================================
# AUTHENTICATION WINDOWS
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

    windows = grouped.mean(
        axis=1
    )

    return windows


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PCA + SVM HYPERPARAMETER OPTIMIZATION")
print("=" * 70)

print("\nSubjects:", SUBJECTS)
print("Recordings:", RECORDINGS)
print(
    "Epochs per authentication window:",
    EPOCHS_PER_WINDOW
)

print("\nSearch space:")

print(
    "PCA components:",
    PCA_COMPONENTS
)

print(
    "SVM C:",
    SVM_C_VALUES
)

print(
    "SVM gamma:",
    SVM_GAMMA_VALUES
)


all_data = {}


print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)


for subject in SUBJECTS:

    all_data[subject] = {}

    for recording in RECORDINGS:

        path = (
            DATA_DIR /
            subject /
            f"recording_{recording}_epochs.npy"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing:\n{path}"
            )

        epochs = np.load(path)

        windows = create_windows(
            epochs
        )

        all_data[
            subject
        ][recording] = windows

        print(
            f"{subject} | R{recording} | "
            f"epochs={epochs.shape} | "
            f"windows={windows.shape}"
        )


# ============================================================
# PRECOMPUTE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING FEATURES")
print("=" * 70)


feature_data = {}


for subject in SUBJECTS:

    feature_data[subject] = {}

    for recording in RECORDINGS:

        print(
            f"Extracting {subject} R{recording}..."
        )

        feature_data[
            subject
        ][recording] = extract_features(
            all_data[
                subject
            ][recording]
        )


print(
    "\nFeature extraction complete."
)

print(
    "Features per window:",
    feature_data[
        SUBJECTS[0]
    ][1].shape[1]
)


# ============================================================
# BUILD MULTI-SUBJECT DATASET
# ============================================================

def get_dataset(recording):

    X = []
    y = []

    for subject_index, subject in enumerate(
        SUBJECTS
    ):

        features = feature_data[
            subject
        ][recording]

        X.append(features)

        y.extend(
            [subject_index] *
            len(features)
        )

    return (
        np.concatenate(X, axis=0),
        np.asarray(y)
    )


# ============================================================
# TRAIN ONE-VS-REST MODELS
# ============================================================

def train_models(
    X,
    y,
    n_components,
    C,
    gamma
):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    models = []

    for subject_index in range(
        len(SUBJECTS)
    ):

        binary_labels = (
            y == subject_index
        ).astype(int)

        svm = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight="balanced"
        )

        svm.fit(
            X_pca,
            binary_labels
        )

        models.append(svm)

    return (
        scaler,
        pca,
        models
    )


# ============================================================
# COLLECT AUTHENTICATION SCORES
# ============================================================

def get_scores(
    X,
    y,
    scaler,
    pca,
    models
):

    X_scaled = scaler.transform(
        X
    )

    X_pca = pca.transform(
        X_scaled
    )

    genuine_scores = []
    impostor_scores = []

    for claimed_index, model in enumerate(
        models
    ):

        scores = model.decision_function(
            X_pca
        )

        for actual_index, score in zip(
            y,
            scores
        ):

            if actual_index == claimed_index:

                genuine_scores.append(
                    score
                )

            else:

                impostor_scores.append(
                    score
                )

    return (
        np.asarray(genuine_scores),
        np.asarray(impostor_scores)
    )


# ============================================================
# EER CALCULATION
# ============================================================

def compute_eer(
    genuine,
    impostor
):

    scores = np.concatenate([
        genuine,
        impostor
    ])

    labels = np.concatenate([
        np.ones(len(genuine)),
        np.zeros(len(impostor))
    ])

    # Sort scores descending
    order = np.argsort(
        scores
    )[::-1]

    scores = scores[order]
    labels = labels[order]

    genuine_total = np.sum(
        labels == 1
    )

    impostor_total = np.sum(
        labels == 0
    )

    genuine_accepted = np.cumsum(
        labels == 1
    )

    impostor_accepted = np.cumsum(
        labels == 0
    )

    far = (
        impostor_accepted /
        impostor_total
    )

    frr = (
        1 -
        genuine_accepted /
        genuine_total
    )

    index = np.argmin(
        np.abs(far - frr)
    )

    eer = (
        far[index] +
        frr[index]
    ) / 2

    threshold = scores[index]

    return (
        float(eer),
        float(threshold)
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate_pair(
    enrollment_recording,
    validation_recording,
    n_components,
    C,
    gamma
):

    X_train, y_train = get_dataset(
        enrollment_recording
    )

    X_validation, y_validation = get_dataset(
        validation_recording
    )

    scaler, pca, models = train_models(
        X_train,
        y_train,
        n_components,
        C,
        gamma
    )

    genuine, impostor = get_scores(
        X_validation,
        y_validation,
        scaler,
        pca,
        models
    )

    eer, threshold = compute_eer(
        genuine,
        impostor
    )

    return eer


# ============================================================
# GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("GRID SEARCH")
print("=" * 70)

print(
    "\nEach configuration is evaluated "
    "using cross-session validation."
)


# Use R1 -> R2 as the development validation pair.
#
# IMPORTANT:
# R3 will NOT be used for choosing parameters.
#
# This prevents the final test recording from
# influencing model selection.

VALIDATION_ENROLLMENT = 1
VALIDATION_RECORDING = 2


grid_results = []

total_combinations = (
    len(PCA_COMPONENTS) *
    len(SVM_C_VALUES) *
    len(SVM_GAMMA_VALUES)
)

current = 0


for n_components in PCA_COMPONENTS:

    for C in SVM_C_VALUES:

        for gamma in SVM_GAMMA_VALUES:

            current += 1

            print(
                f"[{current}/{total_combinations}] "
                f"PCA={n_components}, "
                f"C={C}, "
                f"gamma={gamma}"
            )

            eer = evaluate_pair(
                VALIDATION_ENROLLMENT,
                VALIDATION_RECORDING,
                n_components,
                C,
                gamma
            )

            grid_results.append({
                "pca_components":
                    n_components,

                "C":
                    C,

                "gamma":
                    gamma,

                "validation_eer":
                    eer
            })

            print(
                f"    Validation EER: "
                f"{eer * 100:.2f}%"
            )


# ============================================================
# SELECT BEST CONFIGURATION
# ============================================================

grid_results.sort(
    key=lambda x:
    x["validation_eer"]
)

best = grid_results[0]


print("\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(
    "\nPCA components:",
    best["pca_components"]
)

print(
    "SVM C:",
    best["C"]
)

print(
    "SVM gamma:",
    best["gamma"]
)

print(
    "Validation EER:",
    f"{best['validation_eer'] * 100:.2f}%"
)


# ============================================================
# SAVE GRID RESULTS
# ============================================================

with open(
    OUTPUT_DIR /
    "grid_search_results.json",
    "w"
) as f:

    json.dump(
        {
            "validation_enrollment":
                VALIDATION_ENROLLMENT,

            "validation_recording":
                VALIDATION_RECORDING,

            "best_configuration":
                best,

            "all_results":
                grid_results
        },
        f,
        indent=2
    )


# ============================================================
# FINAL CONFIRMATION TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL CONFIRMATION TEST")
print("=" * 70)

print(
    "\nThe selected configuration will now be "
    "tested on the previously unseen Recording 3."
)

X_train, y_train = get_dataset(1)

X_test, y_test = get_dataset(3)

scaler, pca, models = train_models(
    X_train,
    y_train,
    best["pca_components"],
    best["C"],
    best["gamma"]
)

genuine, impostor = get_scores(
    X_test,
    y_test,
    scaler,
    pca,
    models
)

test_eer, test_threshold = compute_eer(
    genuine,
    impostor
)


print(
    "\nCONFIRMATION RESULT"
)

print(
    "Enrollment : Recording 1"
)

print(
    "Validation : Recording 2"
)

print(
    "Final Test : Recording 3"
)

print(
    "\nSelected PCA:",
    best["pca_components"]
)

print(
    "Selected C:",
    best["C"]
)

print(
    "Selected gamma:",
    best["gamma"]
)

print(
    f"\nValidation EER : "
    f"{best['validation_eer'] * 100:.2f}%"
)

print(
    f"Final Test EER : "
    f"{test_eer * 100:.2f}%"
)


# ============================================================
# SAVE FINAL RESULT
# ============================================================

final_results = {

    "method":
        "PCA + RBF SVM hyperparameter optimization",

    "subjects":
        SUBJECTS,

    "features":
        224,

    "epochs_per_window":
        EPOCHS_PER_WINDOW,

    "validation": {

        "enrollment_recording":
            VALIDATION_ENROLLMENT,

        "recording":
            VALIDATION_RECORDING,

        "eer":
            best["validation_eer"]
    },

    "selected_parameters": {

        "pca_components":
            best["pca_components"],

        "C":
            best["C"],

        "gamma":
            best["gamma"]
    },

    "confirmation_test": {

        "enrollment_recording":
            1,

        "test_recording":
            3,

        "eer":
            test_eer,

        "threshold":
            test_threshold
    }
}


with open(
    OUTPUT_DIR /
    "optimization_results.json",
    "w"
) as f:

    json.dump(
        final_results,
        f,
        indent=2
    )


print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR /
    "grid_search_results.json"
)

print(
    OUTPUT_DIR /
    "optimization_results.json"
)
