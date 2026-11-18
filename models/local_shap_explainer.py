"""On-demand SHAP explanations for the saved EEG authentication CNNs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

from models.cnn_authentication_early_stopping import (
    ROOT,
    SUBJECTS,
    DEVICE,
    EEG1DCNN,
    load_recording,
    standardize_windows,
)

BACKGROUND_SIZE = 16
SHAP_NSAMPLES = 50
SEED = 42

CHANNELS = [
    "Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8",
    "F3", "F4", "FCz", "C3", "C4", "CP3", "CPz", "CP4",
]

RESULTS_DIR = ROOT / "results" / "cnn_authentication_early_stopping"
MODEL_DIR = RESULTS_DIR / "saved_models"
OUTPUT_DIR = ROOT / "results" / "shap_explainability"
VALUES_DIR = OUTPUT_DIR / "shap_values"


class SHAPWrapper(torch.nn.Module):
    """Expose the CNN logit with a two-dimensional output for SHAP."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(1)


def load_subject_model(subject):
    """Load a saved subject-specific CNN; do not train it."""
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown subject {subject!r}. Choose from {SUBJECTS}.")

    checkpoint = MODEL_DIR / f"{subject}_cnn_state_dict.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint}")

    model = EEG1DCNN().to(DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_recordings():
    """Load and standardize all recordings using the existing project pipeline."""
    X1, y1 = load_recording(1)
    X2, y2 = load_recording(2)
    X3, y3 = load_recording(3)
    X1, X2, X3, mean, std = standardize_windows(X1, X2, X3)
    return (X1, y1), (X2, y2), (X3, y3)


def get_thresholds():
    path = RESULTS_DIR / "cnn_authentication_results.json"
    with path.open(encoding="utf-8") as f:
        results = json.load(f)
    return results["validation_thresholds_by_subject"]


def explain_window(subject, window_index, recording=3):
    """
    Explain one window from a recording loaded by the project's data pipeline.

    The index is zero-based. The decision uses the saved R2 validation threshold.
    """
    if recording not in (1, 2, 3):
        raise ValueError("recording must be 1, 2, or 3")

    recordings = get_recordings()
    X, y = recordings[recording - 1]

    if not isinstance(window_index, (int, np.integer)):
        raise TypeError("window_index must be an integer")
    if not 0 <= window_index < len(X):
        raise IndexError(
            f"Window index must be between 0 and {len(X) - 1} "
            f"for recording {recording}."
        )

    rng = np.random.default_rng(SEED)
    background_indices = rng.choice(
        len(recordings[0][0]),
        size=min(BACKGROUND_SIZE, len(recordings[0][0])),
        replace=False,
    )
    background = torch.as_tensor(
        recordings[0][0][background_indices],
        dtype=torch.float32,
        device=DEVICE,
    )

    window = X[window_index:window_index + 1]
    window_tensor = torch.as_tensor(
        window, dtype=torch.float32, device=DEVICE
    )

    model = load_subject_model(subject)
    with torch.no_grad():
        score = float(model(window_tensor).cpu().item())

    threshold = float(get_thresholds()[subject])
    decision = "ACCEPT" if score >= threshold else "REJECT"

    wrapped = SHAPWrapper(model)
    explainer = shap.GradientExplainer(wrapped, background)
    raw_values = explainer.shap_values(
        window_tensor, nsamples=SHAP_NSAMPLES
    )

    values = np.asarray(raw_values)
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[0])

    # SHAP versions can return (1, channels, time, 1) or
    # (1, channels, time); normalize to (channels, time).
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.shape != window.shape[1:]:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; "
            f"expected {window.shape[1:]}"
        )

    importance = np.abs(values).mean(axis=1)
    order = np.argsort(importance)[::-1]
    top_channels = [
        {
            "channel": CHANNELS[int(i)],
            "mean_absolute_shap": float(importance[i]),
            "mean_signed_shap": float(values[i].mean()),
        }
        for i in order[:5]
    ]

    # This wording describes the model decision, not biological causation.
    relation = "at or above" if decision == "ACCEPT" else "below"
    top_names = ", ".join(item["channel"] for item in top_channels[:3])
    plain_english = (
        f"The model gave this EEG window a score of {score:.3f}, "
        f"which is {relation} the saved authentication threshold "
        f"of {threshold:.3f}. The resulting decision is {decision}. "
        f"The strongest average absolute SHAP contributions were associated "
        f"with {top_names}. SHAP describes how the model's output is "
        f"attributed across the input; it does not prove that these channels "
        f"caused the decision or establish a person's identity by itself."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local_dir = OUTPUT_DIR / "on_demand"
    local_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{subject}_recording{recording}_window{window_index}"
    pd.DataFrame({
        "channel": CHANNELS,
        "mean_absolute_shap": importance,
        "mean_signed_shap": values.mean(axis=1),
    }).sort_values(
        "mean_absolute_shap", ascending=False
    ).to_csv(local_dir / f"{safe_name}_channels.csv", index=False)

    np.savez_compressed(
        local_dir / f"{safe_name}_explanation.npz",
        input_window=window[0],
        shap_values=values,
        channels=np.asarray(CHANNELS),
        score_logit=score,
        threshold=threshold,
        decision=decision,
        recording=recording,
        window_index=window_index,
    )

    return {
        "subject": subject,
        "recording": recording,
        "window_index": int(window_index),
        "dataset_label_is_subject": bool(y[window_index] == SUBJECTS.index(subject)),
        "score_logit": score,
        "validation_threshold": threshold,
        "decision": decision,
        "top_channels": top_channels,
        "plain_english": plain_english,
        "shap_values": values,
        "input_window": window[0],
        "channels": list(CHANNELS),
        "sampling_rate_hz": 256,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Explain one recorded EEG authentication window."
    )
    parser.add_argument("--subject", choices=SUBJECTS, required=True)
    parser.add_argument("--recording", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--window", type=int, required=True,
                        help="Zero-based window index")
    args = parser.parse_args()

    result = explain_window(args.subject, args.window, args.recording)

    print("\n=== EEG Authentication ===")
    print(f"Subject model: {result['subject']}")
    print(f"Recording/window: {result['recording']}/{result['window_index']}")
    print(f"Score (logit): {result['score_logit']:.4f}")
    print(f"Threshold: {result['validation_threshold']:.4f}")
    print(f"Decision: {result['decision']}")
    print("\nTop channels:")
    for item in result["top_channels"]:
        print(
            f"  {item['channel']}: "
            f"mean |SHAP|={item['mean_absolute_shap']:.6f}"
        )
    print("\nPlain-English explanation:")
    print(result["plain_english"])
    print(f"\nSaved detailed output under: {OUTPUT_DIR / 'on_demand'}")
