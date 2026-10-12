import json
from pathlib import Path

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "preprocessed_sessions"
OUTPUT_DIR = ROOT / "results" / "classifier_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"A{i:02d}" for i in range(1, 11)]
FS = 256
EPOCHS_PER_WINDOW = 10
RANDOM_STATE = 42

# Parameters selected by the existing SVM optimization experiment.
PCA_COMPONENTS = 100
SVM_C = 10.0
SVM_GAMMA = 0.001

BANDS = [
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 30.0),
]


def create_windows(epochs):
    n_windows = len(epochs) // EPOCHS_PER_WINDOW
    if n_windows == 0:
        raise ValueError("Not enough epochs to form one window.")

    usable = epochs[:n_windows * EPOCHS_PER_WINDOW]
    grouped = usable.reshape(
        n_windows,
        EPOCHS_PER_WINDOW,
        epochs.shape[1],
        epochs.shape[2],
    )
    return grouped.mean(axis=1)


def extract_features(windows):
    all_features = []

    for window in windows:
        features = []

        for channel in window:
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.max(channel),
                np.min(channel),
                np.ptp(channel),
                np.var(channel),
            ])

            freqs, psd = welch(
                channel,
                fs=FS,
                nperseg=min(256, len(channel)),
            )

            for low, high in BANDS:
                mask = (freqs >= low) & (freqs < high)
                features.append(
                    float(np.mean(psd[mask])) if np.any(mask) else 0.0
                )

            fft_values = np.abs(rfft(channel))
            fft_freqs = rfftfreq(len(channel), d=1 / FS)

            for low, high in BANDS:
                mask = (fft_freqs >= low) & (fft_freqs < high)
                features.append(
                    float(np.mean(fft_values[mask]))
                    if np.any(mask) else 0.0
                )

        all_features.append(features)

    return np.asarray(all_features, dtype=np.float64)


def load_recording(recording_number):
    X_parts = []
    y_parts = []

    for subject_index, subject in enumerate(SUBJECTS):
        path = DATA_DIR / subject / f"recording_{recording_number}_epochs.npy"

        if not path.exists():
            raise FileNotFoundError(f"Missing recording: {path}")

        epochs = np.load(path)
        windows = create_windows(epochs)
        features = extract_features(windows)

        if features.shape[1] != 224:
            raise ValueError(
                f"Expected 224 features, got {features.shape[1]} "
                f"for {subject} recording {recording_number}"
            )

        X_parts.append(features)
        y_parts.extend([subject_index] * len(features))

    return np.concatenate(X_parts), np.asarray(y_parts, dtype=int)


def build_authenticator(name):
    if name == "SVM":
        return SVC(
            kernel="rbf",
            C=SVM_C,
            gamma=SVM_GAMMA,
            class_weight="balanced",
        )

    if name == "Logistic Regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )

    if name == "LDA":
        return LinearDiscriminantAnalysis(
            priors=[0.5, 0.5],
        )

    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown classifier: {name}")


def score_model(model, X):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).reshape(-1)

    return np.asarray(model.predict_proba(X)[:, 1]).reshape(-1)


def collect_scores(models, X, y):
    genuine_scores = []
    impostor_scores = []

    for claimed_subject, model in enumerate(models):
        scores = score_model(model, X)
        genuine_mask = y == claimed_subject

        genuine_scores.extend(scores[genuine_mask])
        impostor_scores.extend(scores[~genuine_mask])

    return np.asarray(genuine_scores), np.asarray(impostor_scores)


def calculate_eer_threshold(genuine, impostor):
    labels = np.concatenate([
        np.ones(len(genuine), dtype=int),
        np.zeros(len(impostor), dtype=int),
    ])
    scores = np.concatenate([genuine, impostor])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))

    eer = float((fpr[index] + fnr[index]) / 2.0)
    threshold = float(thresholds[index])
    return eer, threshold


def rates_at_threshold(genuine, impostor, threshold):
    # Higher score means more likely to be the claimed subject.
    far = float(np.mean(impostor >= threshold))
    frr = float(np.mean(genuine < threshold))
    return far, frr


def main():
    print("=" * 70)
    print("CROSS-SESSION CLASSIFIER COMPARISON")
    print("=" * 70)
    print("Enrollment: Recording 1")
    print("Threshold calibration: Recording 2")
    print("Held-out evaluation: Recording 3")
    print("Features: 224 per window")
    print("PCA components:", PCA_COMPONENTS)

    print("\nLoading recordings...")
    X_train, y_train = load_recording(1)
    X_val, y_val = load_recording(2)
    X_test, y_test = load_recording(3)

    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("Test data:", X_test.shape)

    # Fit preprocessing exclusively on the enrollment recording.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=RANDOM_STATE,
    )
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    classifier_names = [
        "SVM",
        "Logistic Regression",
        "LDA",
        "Random Forest",
    ]

    results = {
        "protocol": {
            "enrollment_recording": 1,
            "threshold_calibration_recording": 2,
            "held_out_test_recording": 3,
            "subjects": SUBJECTS,
            "features_per_window": 224,
            "epochs_per_window": EPOCHS_PER_WINDOW,
            "pca_components": PCA_COMPONENTS,
            "scaler_and_pca_fit_on": "Recording 1 only",
            "threshold_selected_on": "Recording 2 only",
            "test_threshold_reused_without_adjustment": True,
        },
        "classifiers": {},
    }

    for name in classifier_names:
        print(f"\nEvaluating {name}...")

        models = []

        for subject_index in range(len(SUBJECTS)):
            binary_train_labels = (
                y_train == subject_index
            ).astype(int)

            model = build_authenticator(name)
            model.fit(X_train_pca, binary_train_labels)
            models.append(model)

        val_genuine, val_impostor = collect_scores(
            models, X_val_pca, y_val
        )
        test_genuine, test_impostor = collect_scores(
            models, X_test_pca, y_test
        )

        val_eer, val_threshold = calculate_eer_threshold(
            val_genuine, val_impostor
        )

        test_far, test_frr = rates_at_threshold(
            test_genuine, test_impostor, val_threshold
        )

        # Descriptive only: this test EER uses test labels to find
        # its own threshold. It is NOT the deployed test operating point.
        test_eer_descriptive, test_eer_threshold_descriptive = (
            calculate_eer_threshold(test_genuine, test_impostor)
        )

        entry = {
            "validation_eer": val_eer,
            "validation_threshold": val_threshold,
            "test_far_at_validation_threshold": test_far,
            "test_frr_at_validation_threshold": test_frr,
            "test_eer_descriptive_only": test_eer_descriptive,
            "test_eer_threshold_descriptive_only":
                test_eer_threshold_descriptive,
            "test_genuine_score_count": int(len(test_genuine)),
            "test_impostor_score_count": int(len(test_impostor)),
        }

        results["classifiers"][name] = entry

        print(f"  Validation EER: {val_eer * 100:.2f}%")
        print(f"  Test FAR at fixed threshold: {test_far * 100:.2f}%")
        print(f"  Test FRR at fixed threshold: {test_frr * 100:.2f}%")
        print(
            "  Descriptive test EER (test-selected threshold): "
            f"{test_eer_descriptive * 100:.2f}%"
        )

    output_path = OUTPUT_DIR / "classifier_comparison_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nComparison complete.")
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()
