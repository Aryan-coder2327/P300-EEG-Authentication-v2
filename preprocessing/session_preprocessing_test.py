import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt
import json
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUBJECT = "A01"

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / f"{SUBJECT}S.mat"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessed_sessions"
    / SUBJECT
)

# Dataset sampling rate
FS = 256

# Bandpass filter
LOWCUT = 0.1
HIGHCUT = 30.0

# Epoch duration
BASELINE_MS = 200
POST_STIMULUS_MS = 800


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(signal):
    """
    Apply a 4th-order Butterworth bandpass filter.
    Frequency range: 0.1–30 Hz
    """

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
# LOAD MATLAB DATA
# ============================================================

print("=" * 60)
print(f"PROCESSING {SUBJECT}")
print("=" * 60)

print(f"\nInput: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

mat = sio.loadmat(INPUT_FILE)

if "data" not in mat:
    raise KeyError(
        "MAT file does not contain a 'data' variable."
    )

data = mat["data"]

print(f"Number of recordings: {data.shape[1]}")

# Create output directory
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PROCESS EACH RECORDING
# ============================================================

for recording_idx in range(data.shape[1]):

    recording_number = recording_idx + 1

    print("\n" + "=" * 60)
    print(f"RECORDING {recording_number}")
    print("=" * 60)

    # --------------------------------------------------------
    # Unwrap MATLAB structure
    # --------------------------------------------------------

    recording = data[
        0,
        recording_idx
    ][0, 0]

    # --------------------------------------------------------
    # Extract dataset fields
    # --------------------------------------------------------

    # Field 0 = channel names
    channel_names = [
        str(x[0])
        for x in recording[0][0]
    ]

    # Field 1 = EEG signal
    X = recording[1]

    # Field 2 = stimulus labels
    y_stim = recording[2].flatten()

    # Field 3 = target/non-target labels
    y_target = recording[3].flatten()

    print(
        "Raw EEG shape:",
        X.shape
    )

    print(
        "Stimulus labels:",
        y_stim.shape
    )

    print(
        "Target labels:",
        y_target.shape
    )

    print(
        "Sampling rate:",
        FS
    )

    print(
        "Channels:",
        channel_names
    )


    # ========================================================
    # CONVERT EEG TO (CHANNELS, SAMPLES)
    # ========================================================

    eeg = X.T.astype(float)

    print(
        "EEG shape (channels, samples):",
        eeg.shape
    )


    # ========================================================
    # BANDPASS FILTERING
    # ========================================================

    print("Filtering EEG...")

    filtered_eeg = np.zeros_like(
        eeg,
        dtype=float
    )

    for ch in range(eeg.shape[0]):

        filtered_eeg[ch] = bandpass_filter(
            eeg[ch]
        )


    # ========================================================
    # FIND TARGET EVENTS
    # ========================================================

    target_indices = np.where(
        y_target == 1
    )[0]

    print(
        "Target events found:",
        len(target_indices)
    )


    # ========================================================
    # EPOCH EXTRACTION
    #
    # Approximately:
    # -200 ms → +800 ms
    #
    # At 256 Hz:
    # 200 ms ≈ 51 samples
    # 800 ms ≈ 205 samples
    #
    # Total = 256 samples
    # ========================================================

    baseline_samples = int(
        round(
            BASELINE_MS / 1000 * FS
        )
    )

    post_samples = int(
        round(
            POST_STIMULUS_MS / 1000 * FS
        )
    )

    epoch_length = (
        baseline_samples
        + post_samples
    )

    print(
        "Baseline samples:",
        baseline_samples
    )

    print(
        "Post-stimulus samples:",
        post_samples
    )

    print(
        "Total epoch samples:",
        epoch_length
    )


    # --------------------------------------------------------
    # Extract epochs
    # --------------------------------------------------------

    epochs = []

    for idx in target_indices:

        start = (
            idx
            - baseline_samples
        )

        end = (
            start
            + epoch_length
        )

        # Make sure the complete epoch
        # exists inside the recording
        if (
            start >= 0
            and
            end <= filtered_eeg.shape[1]
        ):

            epoch = filtered_eeg[
                :,
                start:end
            ]

            # Safety check
            if (
                epoch.shape[1]
                == epoch_length
            ):

                epochs.append(epoch)


    epochs = np.asarray(
        epochs,
        dtype=float
    )

    print(
        "Epoch shape before baseline correction:",
        epochs.shape
    )

    # Safety check
    if epochs.ndim != 3:
        raise RuntimeError(
            f"Epoch extraction failed. "
            f"Expected 3D array but got "
            f"shape {epochs.shape}"
        )


    # ========================================================
    # BASELINE CORRECTION
    # ========================================================

    print(
        "Applying baseline correction..."
    )

    baseline_corrected = np.zeros_like(
        epochs,
        dtype=float
    )

    for i in range(
        epochs.shape[0]
    ):

        # Mean of the baseline period
        # for each channel
        baseline = epochs[
            i,
            :,
            :baseline_samples
        ].mean(
            axis=1,
            keepdims=True
        )

        # Subtract baseline from entire epoch
        baseline_corrected[i] = (
            epochs[i]
            - baseline
        )


    print(
        "Epoch shape after baseline correction:",
        baseline_corrected.shape
    )


    # ========================================================
    # SAVE PREPROCESSED EPOCHS
    # ========================================================

    epoch_file = (
        OUTPUT_DIR
        / f"recording_{recording_number}_epochs.npy"
    )

    metadata_file = (
        OUTPUT_DIR
        / f"recording_{recording_number}_metadata.json"
    )

    np.save(
        epoch_file,
        baseline_corrected
    )


    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {

        "subject": SUBJECT,

        "recording": recording_number,

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

        "baseline_samples": baseline_samples,

        "post_stimulus_samples": post_samples,

        "epoch_samples": epoch_length,

        "filter": "0.1-30 Hz Butterworth bandpass",

        "source_file": INPUT_FILE.name

    }


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
        f"✓ Saved: {epoch_file}"
    )

    print(
        f"✓ Saved: {metadata_file}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print(f"{SUBJECT} PROCESSING COMPLETE")
print("=" * 60)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)
