import numpy as np
import json
from pathlib import Path
from itertools import permutations

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "preprocessed_sessions"
RESULTS_DIR = PROJECT_ROOT / "results" / "cross_session"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"A{i:02d}" for i in range(1, 11)]
RECORDINGS = [1, 2, 3]

# Number of epochs used to create one authentication sample.
# We average multiple P300 epochs to make the signal more stable.
EPOCHS_PER_WINDOW = 10

# ============================================================
# LOAD DATA
# ============================================================

def load_recording(subject, recording):

    file_path = (
        DATA_DIR
        / subject
        / f"recording_{recording}_epochs.npy"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find: {file_path}"
        )

    return np.load(file_path)


# ============================================================
# CREATE AUTHENTICATION WINDOWS
# ============================================================

def create_windows(epochs, window_size=10):

    windows = []

    n_complete = len(epochs) // window_size

    for i in range(n_complete):

        start = i * window_size
        end = start + window_size

        window = epochs[start:end].mean(axis=0)

        windows.append(window)

    return np.asarray(windows)


# ============================================================
# CREATE TEMPLATE
# ============================================================

def create_template(epochs):

    """
    Enrollment template = average EEG pattern
    from all enrollment epochs.
    """

    template = epochs.mean(axis=0)

    return template


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a_flat = a.flatten()
    b_flat = b.flatten()

    denominator = (
        np.linalg.norm(a_flat)
        * np.linalg.norm(b_flat)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a_flat, b_flat)
        / denominator
    )


# ============================================================
# CALCULATE EER
# ============================================================

def calculate_eer(genuine_scores, impostor_scores):

    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)

    all_scores = np.concatenate(
        [genuine_scores, impostor_scores]
    )

    thresholds = np.linspace(
        all_scores.min(),
        all_scores.max(),
        1000
    )

    fars = []
    frrs = []

    for threshold in thresholds:

        # Genuine user rejected
        frr = np.mean(
            genuine_scores < threshold
        )

        # Impostor accepted
        far = np.mean(
            impostor_scores >= threshold
        )

        fars.append(far)
        frrs.append(frr)

    fars = np.asarray(fars)
    frrs = np.asarray(frrs)

    idx = np.argmin(
        np.abs(fars - frrs)
    )

    eer = (fars[idx] + frrs[idx]) / 2

    return {
        "eer": float(eer),
        "threshold": float(thresholds[idx]),
        "far": float(fars[idx]),
        "frr": float(frrs[idx])
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print("=" * 70)
print("CROSS-SESSION EEG AUTHENTICATION")
print("=" * 70)

print("\nConfiguration:")
print("Subjects:", SUBJECTS)
print("Recordings:", RECORDINGS)
print("Epochs per authentication window:", EPOCHS_PER_WINDOW)

# ------------------------------------------------------------
# Load all recordings
# ------------------------------------------------------------

print("\nLoading data...")

dataset = {}

for subject in SUBJECTS:

    dataset[subject] = {}

    for recording in RECORDINGS:

        epochs = load_recording(
            subject,
            recording
        )

        dataset[subject][recording] = epochs

        print(
            f"{subject} | Recording {recording} | "
            f"{epochs.shape}"
        )


# ============================================================
# TEST ALL CROSS-SESSION COMBINATIONS
# ============================================================

all_results = []

all_genuine_scores = []
all_impostor_scores = []

session_combinations = list(
    permutations(RECORDINGS, 2)
)

for enrollment_recording, test_recording in session_combinations:

    print("\n" + "=" * 70)
    print(
        f"ENROLLMENT: Recording {enrollment_recording} "
        f"→ TEST: Recording {test_recording}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Create templates for every subject
    # --------------------------------------------------------

    templates = {}

    for subject in SUBJECTS:

        enrollment_epochs = (
            dataset[subject][enrollment_recording]
        )

        templates[subject] = create_template(
            enrollment_epochs
        )

    # --------------------------------------------------------
    # Test every subject
    # --------------------------------------------------------

    combination_genuine = []
    combination_impostor = []

    for test_subject in SUBJECTS:

        test_epochs = (
            dataset[test_subject][test_recording]
        )

        windows = create_windows(
            test_epochs,
            EPOCHS_PER_WINDOW
        )

        print(
            f"\nTesting {test_subject}: "
            f"{len(windows)} authentication windows"
        )

        for window in windows:

            # ================================================
            # GENUINE TEST
            # Same person's test signal
            # vs same person's enrolled template
            # ================================================

            genuine_score = cosine_similarity(
                window,
                templates[test_subject]
            )

            combination_genuine.append(
                genuine_score
            )

            # ================================================
            # IMPOSTOR TESTS
            # Same test signal vs everyone else's template
            # ================================================

            for enrolled_subject in SUBJECTS:

                if enrolled_subject == test_subject:
                    continue

                impostor_score = cosine_similarity(
                    window,
                    templates[enrolled_subject]
                )

                combination_impostor.append(
                    impostor_score
                )

    # --------------------------------------------------------
    # Calculate EER for this combination
    # --------------------------------------------------------

    metrics = calculate_eer(
        combination_genuine,
        combination_impostor
    )

    result = {
        "enrollment_recording": enrollment_recording,
        "test_recording": test_recording,
        "eer": metrics["eer"],
        "threshold": metrics["threshold"],
        "far": metrics["far"],
        "frr": metrics["frr"],
        "genuine_samples": len(combination_genuine),
        "impostor_samples": len(combination_impostor)
    }

    all_results.append(result)

    all_genuine_scores.extend(
        combination_genuine
    )

    all_impostor_scores.extend(
        combination_impostor
    )

    print("\nRESULT:")
    print(
        f"EER: {metrics['eer'] * 100:.2f}%"
    )
    print(
        f"Threshold: {metrics['threshold']:.4f}"
    )
    print(
        f"FAR: {metrics['far'] * 100:.2f}%"
    )
    print(
        f"FRR: {metrics['frr'] * 100:.2f}%"
    )


# ============================================================
# OVERALL CROSS-SESSION RESULT
# ============================================================

print("\n" + "=" * 70)
print("OVERALL CROSS-SESSION RESULT")
print("=" * 70)

overall_metrics = calculate_eer(
    all_genuine_scores,
    all_impostor_scores
)

print(
    f"\nOverall EER: "
    f"{overall_metrics['eer'] * 100:.2f}%"
)

print(
    f"Overall Threshold: "
    f"{overall_metrics['threshold']:.4f}"
)

print(
    f"Overall FAR: "
    f"{overall_metrics['far'] * 100:.2f}%"
)

print(
    f"Overall FRR: "
    f"{overall_metrics['frr'] * 100:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output = {
    "configuration": {
        "subjects": SUBJECTS,
        "recordings": RECORDINGS,
        "epochs_per_window": EPOCHS_PER_WINDOW,
        "authentication_method":
            "Cross-session cosine similarity"
    },
    "session_pair_results": all_results,
    "overall_result": overall_metrics,
    "total_genuine_scores": len(all_genuine_scores),
    "total_impostor_scores": len(all_impostor_scores)
}

output_file = (
    RESULTS_DIR
    / "cross_session_results.json"
)

with open(output_file, "w") as f:
    json.dump(
        output,
        f,
        indent=4
    )

# Save raw scores for later graphs
np.save(
    RESULTS_DIR / "genuine_scores.npy",
    np.asarray(all_genuine_scores)
)

np.save(
    RESULTS_DIR / "impostor_scores.npy",
    np.asarray(all_impostor_scores)
)

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print(f"\nResults saved to:")
print(output_file)

print("\nRaw scores saved to:")
print(RESULTS_DIR / "genuine_scores.npy")
print(RESULTS_DIR / "impostor_scores.npy")
