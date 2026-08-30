import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt
import json
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed_sessions"

SUBJECTS = [
    "A01", "A02", "A03", "A04", "A05",
    "A06", "A07", "A08", "A09", "A10"
]

FS = 256

LOWCUT = 0.1
HIGHCUT = 30.0

BASELINE_MS = 200
POST_STIMULUS_MS = 800

BASELINE_SAMPLES = int(round(BASELINE_MS / 1000 * FS))
POST_SAMPLES = int(round(POST_STIMULUS_MS / 1000 * FS))

EPOCH_LENGTH = BASELINE_SAMPLES + POST_SAMPLES


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(signal):
    nyq = 0.5 * FS

    low = LOWCUT / nyq
    high = HIGHCUT / nyq

    b, a = butter(
        4,
        [low, high],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        signal,
        padlen=0
    )


# ============================================================
# PROCESS ONE RECORDING
# ============================================================

def process_recording(
    recording,
    subject,
    recording_number,
    input_filename
):

    # --------------------------------------------------------
    # Extract fields
    # --------------------------------------------------------

    channel_names = [
        str(x[0])
        for x in recording[0][0]
    ]

    X = recording[1]

    y_stim = recording[2].flatten()

    y_target = recording[3].flatten()

    print("Raw EEG shape:", X.shape)

    print(
        "Stimulus labels:",
        y_stim.shape
    )

    print(
        "Target labels:",
        y_target.shape
    )

    # --------------------------------------------------------
    # Convert to channels × samples
    # --------------------------------------------------------

    eeg = X.T.astype(float)

    print(
        "EEG shape:",
        eeg.shape
    )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    print("Filtering EEG...")

    filtered_eeg = np.zeros_like(
        eeg,
        dtype=float
    )

    for ch in range(eeg.shape[0]):

        filtered_eeg[ch] = bandpass_filter(
            eeg[ch]
        )

    # --------------------------------------------------------
    # Find target events
    # --------------------------------------------------------

    target_indices = np.where(
        y_target == 1
    )[0]

    print(
        "Target events:",
        len(target_indices)
    )

    # --------------------------------------------------------
    # Epoch extraction
    # --------------------------------------------------------

    epochs = []

    for idx in target_indices:

        start = (
            idx
            - BASELINE_SAMPLES
        )

        end = (
            start
            + EPOCH_LENGTH
        )

        if (
            start >= 0
            and
            end <= filtered_eeg.shape[1]
        ):

            epoch = filtered_eeg[
                :,
                start:end
            ]

            if epoch.shape[1] == EPOCH_LENGTH:
                epochs.append(epoch)

    epochs = np.asarray(
        epochs,
        dtype=float
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    expected_shape = (
        len(epochs),
        len(channel_names),
        EPOCH_LENGTH
    )

    if epochs.ndim != 3:
        raise RuntimeError(
            f"Invalid epoch shape: {epochs.shape}"
        )

    print(
        "Epoch shape:",
        epochs.shape
    )

    # --------------------------------------------------------
    # Baseline correction
    # --------------------------------------------------------

    baseline_corrected = np.zeros_like(
        epochs,
        dtype=float
    )

    for i in range(
        epochs.shape[0]
    ):

        baseline = epochs[
            i,
            :,
            :BASELINE_SAMPLES
        ].mean(
            axis=1,
            keepdims=True
        )

        baseline_corrected[i] = (
            epochs[i]
            - baseline
        )

    print(
        "After baseline correction:",
        baseline_corrected.shape
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    subject_dir = (
        OUTPUT_DIR
        / subject
    )

    subject_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save epochs
    # --------------------------------------------------------

    epoch_file = (
        subject_dir
        / f"recording_{recording_number}_epochs.npy"
    )

    np.save(
        epoch_file,
        baseline_corrected
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "subject": subject,

        "recording": recording_number,

        "source_file": input_filename,

        "sampling_rate": FS,

        "n_epochs": int(
            baseline_corrected.shape[0]
        ),

        "n_channels": int(
            baseline_corrected.shape[1]
        ),

        "n_samples": int(
            baseline_corrected.shape[2]
        ),

        "channels": channel_names,

        "baseline_ms": BASELINE_MS,

        "post_stimulus_ms": POST_STIMULUS_MS,

        "baseline_samples": BASELINE_SAMPLES,

        "post_stimulus_samples": POST_SAMPLES,

        "epoch_samples": EPOCH_LENGTH,

        "filter": "0.1-30 Hz Butterworth bandpass"
    }

    metadata_file = (
        subject_dir
        / f"recording_{recording_number}_metadata.json"
    )

    with open(
        metadata_file,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        f"✓ Saved {epoch_file.name}"
    )

    print(
        f"✓ Saved {metadata_file.name}"
    )

    return baseline_corrected.shape


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("SESSION-AWARE P300 PREPROCESSING")
print("=" * 70)

print(
    f"Subjects: {len(SUBJECTS)}"
)

print(
    f"Sampling rate: {FS} Hz"
)

print(
    f"Epoch: approximately "
    f"-{BASELINE_MS} ms → +{POST_STIMULUS_MS} ms"
)

print(
    f"Epoch length: {EPOCH_LENGTH} samples"
)

print("=" * 70)


total_recordings = 0
total_epochs = 0

for subject in SUBJECTS:

    print("\n" + "=" * 70)
    print(f"SUBJECT {subject}")
    print("=" * 70)

    input_file = (
        RAW_DIR
        / f"{subject}S.mat"
    )

    if not input_file.exists():

        print(
            f"⚠ File not found: {input_file}"
        )

        continue

    try:

        mat = sio.loadmat(
            input_file
        )

        data = mat["data"]

        number_of_recordings = (
            data.shape[1]
        )

        print(
            "Number of recordings:",
            number_of_recordings
        )

        for recording_idx in range(
            number_of_recordings
        ):

            recording_number = (
                recording_idx + 1
            )

            print(
                "\n"
                + "-" * 60
            )

            print(
                f"{subject} | "
                f"Recording {recording_number}"
            )

            print(
                "-" * 60
            )

            recording = data[
                0,
                recording_idx
            ][0, 0]

            shape = process_recording(
                recording,
                subject,
                recording_number,
                input_file.name
            )

            total_recordings += 1
            total_epochs += shape[0]

    except Exception as e:

        print(
            f"✗ ERROR processing {subject}:"
        )

        print(
            repr(e)
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print(
    "Total recordings processed:",
    total_recordings
)

print(
    "Total target epochs:",
    total_epochs
)

print(
    "Expected recordings:",
    len(SUBJECTS) * 3
)

print(
    "Expected epochs approximately:",
    len(SUBJECTS) * 3 * 1536
)

print(
    "\nOutput:",
    OUTPUT_DIR
)

print("=" * 70)
