import numpy as np
import json
from pathlib import Path

from scipy.signal import welch
from scipy.fft import rfft, rfftfreq

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc


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
    "pca_svm_cross_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HYPERPARAMETER SEARCH SPACE
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
    Extract the same 224 features used by
    the established PCA + SVM model.

    Input:
        (n_windows, 16, 256)

    Output:
        (n_windows, 224)

    16 channels × 14 features:

        6 temporal:
            mean
            standard deviation
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
# CREATE AUTHENTICATION WINDOWS
# ============================================================

def create_windows(epochs):

    """
    Average every 10 consecutive P300 epochs.

    1536 epochs → 153 complete windows.
    """

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
print("PROPER 3-SESSION CROSS-VALIDATION")
print("PCA + RBF SVM EEG AUTHENTICATION")
print("=" * 70)

print("\nSubjects:", SUBJECTS)
print("Recordings:", RECORDINGS)
print(
    "Epochs per authentication window:",
    EPOCHS_PER_WINDOW
)

print(
    "\nPCA search:",
    PCA_COMPONENTS
)

print(
    "C search:",
    SVM_C_VALUES
)

print(
    "Gamma search:",
    SVM_GAMMA_VALUES
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

        all_data[
            subject
        ][recording] = windows

        print(
            f"{subject} | "
            f"R{recording} | "
            f"epochs={epochs.shape} | "
            f"windows={windows.shape}"
        )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE EXTRACTION")
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
    "Feature dimensions:",
    feature_data[
        SUBJECTS[0]
    ][1].shape
)


# ============================================================
# BUILD DATASET FOR ONE RECORDING
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

        X.append(
            features
        )

        y.extend(
            [subject_index] *
            len(features)
        )

    return (
        np.concatenate(
            X,
            axis=0
        ),
        np.asarray(y)
    )


# ============================================================
# TRAIN ONE-VS-REST SVM MODELS
# ============================================================

def train_models(
    X,
    y,
    n_components,
    C,
    gamma
):

    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    # --------------------------------------------------------
    # ONE-VS-REST SVM
    # --------------------------------------------------------

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

        models.append(
            svm
        )

    return (
        scaler,
        pca,
        models
    )


# ============================================================
# AUTHENTICATION SCORES
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

    # --------------------------------------------------------
    # Every subject is tested as a claimed identity
    # --------------------------------------------------------

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
        np.asarray(
            genuine_scores
        ),
        np.asarray(
            impostor_scores
        )
    )


# ============================================================
# EER
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
        np.ones(
            len(genuine)
        ),
        np.zeros(
            len(impostor)
        )
    ])

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    fnr = 1 - tpr

    index = np.nanargmin(
        np.abs(
            fpr - fnr
        )
    )

    eer = (
        fpr[index] +
        fnr[index]
    ) / 2

    threshold = thresholds[index]

    roc_auc = auc(
        fpr,
        tpr
    )

    return (
        float(eer),
        float(threshold),
        float(roc_auc)
    )


# ============================================================
# EVALUATE ONE SESSION PAIR
# ============================================================

def evaluate_pair(
    enrollment_recording,
    evaluation_recording,
    n_components,
    C,
    gamma
):

    X_train, y_train = get_dataset(
        enrollment_recording
    )

    X_test, y_test = get_dataset(
        evaluation_recording
    )

    scaler, pca, models = train_models(
        X_train,
        y_train,
        n_components,
        C,
        gamma
    )

    genuine, impostor = get_scores(
        X_test,
        y_test,
        scaler,
        pca,
        models
    )

    eer, threshold, roc_auc = compute_eer(
        genuine,
        impostor
    )

    return (
        eer,
        threshold,
        roc_auc
    )


# ============================================================
# SIX ROTATING FOLDS
# ============================================================

folds = [

    # Enrollment, Validation, Test
    (1, 2, 3),
    (1, 3, 2),

    (2, 1, 3),
    (2, 3, 1),

    (3, 1, 2),
    (3, 2, 1)
]


all_fold_results = []


# ============================================================
# PROCESS EACH FOLD
# ============================================================

for fold_number, (
    enrollment_recording,
    validation_recording,
    test_recording
) in enumerate(
    folds,
    start=1
):

    print("\n\n")
    print("=" * 70)

    print(
        f"FOLD {fold_number}/6"
    )

    print(
        f"Enrollment: R{enrollment_recording}"
    )

    print(
        f"Validation: R{validation_recording}"
    )

    print(
        f"Final Test: R{test_recording}"
    )

    print("=" * 70)


    # ========================================================
    # GRID SEARCH ON VALIDATION SESSION
    # ========================================================

    best_validation_eer = np.inf

    best_parameters = None

    validation_results = []


    total = (
        len(PCA_COMPONENTS) *
        len(SVM_C_VALUES) *
        len(SVM_GAMMA_VALUES)
    )

    counter = 0


    print(
        "\nHyperparameter optimization..."
    )


    for n_components in PCA_COMPONENTS:

        for C in SVM_C_VALUES:

            for gamma in SVM_GAMMA_VALUES:

                counter += 1

                print(
                    f"[{counter}/{total}] "
                    f"PCA={n_components}, "
                    f"C={C}, "
                    f"gamma={gamma}"
                )

                eer, threshold, roc_auc = evaluate_pair(
                    enrollment_recording,
                    validation_recording,
                    n_components,
                    C,
                    gamma
                )

                result = {

                    "pca_components":
                        n_components,

                    "C":
                        C,

                    "gamma":
                        gamma,

                    "validation_eer":
                        eer,

                    "validation_auc":
                        roc_auc
                }

                validation_results.append(
                    result
                )

                if eer < best_validation_eer:

                    best_validation_eer = eer

                    best_parameters = {

                        "pca_components":
                            n_components,

                        "C":
                            C,

                        "gamma":
                            gamma
                    }


    # ========================================================
    # BEST VALIDATION CONFIGURATION
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "BEST VALIDATION CONFIGURATION"
    )

    print(
        "PCA:",
        best_parameters[
            "pca_components"
        ]
    )

    print(
        "C:",
        best_parameters[
            "C"
        ]
    )

    print(
        "Gamma:",
        best_parameters[
            "gamma"
        ]
    )

    print(
        f"Validation EER: "
        f"{best_validation_eer * 100:.2f}%"
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "FINAL UNSEEN TEST"
    )

    print(
        f"R{enrollment_recording} "
        f"→ R{test_recording}"
    )


    test_eer, test_threshold, test_auc = evaluate_pair(
        enrollment_recording,
        test_recording,
        best_parameters[
            "pca_components"
        ],
        best_parameters[
            "C"
        ],
        best_parameters[
            "gamma"
        ]
    )


    print(
        f"\nFINAL TEST EER: "
        f"{test_eer * 100:.2f}%"
    )

    print(
        f"FINAL TEST AUC: "
        f"{test_auc:.4f}"
    )


    # ========================================================
    # SAVE FOLD RESULT
    # ========================================================

    fold_result = {

        "fold":
            fold_number,

        "enrollment_recording":
            enrollment_recording,

        "validation_recording":
            validation_recording,

        "test_recording":
            test_recording,

        "best_parameters":
            best_parameters,

        "validation_eer":
            best_validation_eer,

        "test_eer":
            test_eer,

        "test_threshold":
            test_threshold,

        "test_auc":
            test_auc,

        "validation_grid":
            validation_results
    }


    all_fold_results.append(
        fold_result
    )


# ============================================================
# OVERALL RESULT
# ============================================================

test_eers = np.asarray([
    result["test_eer"]
    for result in all_fold_results
])


test_aucs = np.asarray([
    result["test_auc"]
    for result in all_fold_results
])


mean_eer = np.mean(
    test_eers
)

std_eer = np.std(
    test_eers
)

mean_auc = np.mean(
    test_aucs
)

std_auc = np.std(
    test_aucs
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n\n")

print("=" * 70)
print("FINAL 3-SESSION CROSS-VALIDATION RESULT")
print("=" * 70)


print(
    "\nFold results:"
)


for result in all_fold_results:

    print(
        f"Fold {result['fold']}: "
        f"R{result['enrollment_recording']} "
        f"→ R{result['validation_recording']} "
        f"→ R{result['test_recording']} | "
        f"Test EER = "
        f"{result['test_eer'] * 100:.2f}%"
    )


print("\n" + "-" * 70)


print(
    f"\nMean Test EER: "
    f"{mean_eer * 100:.2f}%"
)

print(
    f"Std Test EER: "
    f"{std_eer * 100:.2f}%"
)

print(
    f"Mean ROC-AUC: "
    f"{mean_auc:.4f}"
)

print(
    f"Std ROC-AUC: "
    f"{std_auc:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "experiment":
        "3-session rotating cross-validation",

    "method":
        "224-feature PCA + RBF SVM",

    "subjects":
        SUBJECTS,

    "recordings":
        RECORDINGS,

    "epochs_per_window":
        EPOCHS_PER_WINDOW,

    "input_features":
        224,

    "hyperparameter_search": {

        "pca_components":
            PCA_COMPONENTS,

        "C":
            SVM_C_VALUES,

        "gamma":
            SVM_GAMMA_VALUES
    },

    "folds":
        all_fold_results,

    "overall": {

        "mean_test_eer":
            float(mean_eer),

        "std_test_eer":
            float(std_eer),

        "mean_test_auc":
            float(mean_auc),

        "std_test_auc":
            float(std_auc)
    }
}


output_file = (
    OUTPUT_DIR /
    "cross_validation_results.json"
)


with open(
    output_file,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print("\n" + "=" * 70)

print(
    "RESULTS SAVED"
)

print(
    output_file
)

print("=" * 70)
