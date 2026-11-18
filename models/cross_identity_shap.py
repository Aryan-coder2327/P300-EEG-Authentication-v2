"""Explain a sample from one subject using another subject's saved CNN."""
import argparse
import json
from pathlib import Path

import numpy as np
import shap
import torch

from models.cnn_authentication_early_stopping import (
    SUBJECTS, DEVICE, EEG1DCNN, load_recording, standardize_windows
)

ROOT = Path.cwd()
RESULTS = ROOT / "results" / "cnn_authentication_early_stopping"
MODEL_DIR = RESULTS / "saved_models"
OUTPUT_DIR = ROOT / "results" / "shap_explainability" / "cross_identity"

CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "Pz", "Cz"
]
# Use the project's actual channel names if they are defined elsewhere.
SHAP_NSAMPLES = 50


class SHAPWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claimed", choices=SUBJECTS, required=True)
    parser.add_argument("--source", choices=SUBJECTS, required=True)
    parser.add_argument("--recording", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--window", type=int, required=True)
    args = parser.parse_args()

    X1, y1 = load_recording(1)
    X2, y2 = load_recording(2)
    X3, y3 = load_recording(3)
    X1, X2, X3, _, _ = standardize_windows(X1, X2, X3)
    data = {1: (X1, y1), 2: (X2, y2), 3: (X3, y3)}
    X, y = data[args.recording]

    windows_per_subject = len(X) // len(SUBJECTS)
    if not 0 <= args.window < windows_per_subject:
        raise ValueError(f"Window must be between 0 and {windows_per_subject - 1}.")

    global_idx = SUBJECTS.index(args.source) * windows_per_subject + args.window
    sample = X[global_idx:global_idx + 1]
    if len(sample) != 1:
        raise ValueError("Requested sample index is out of range.")

    checkpoint = MODEL_DIR / f"{args.claimed}_cnn_state_dict.pt"
    model = EEG1DCNN().to(DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    tensor = torch.as_tensor(sample, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        score = float(model(tensor).cpu().item())

    with (RESULTS / "cnn_authentication_results.json").open() as f:
        report = json.load(f)
    threshold = float(report["validation_thresholds_by_subject"][args.claimed])
    decision = "ACCEPT" if score >= threshold else "REJECT"

    # Background uses windows from the claimed subject in Recording 1.
    claimed_start = SUBJECTS.index(args.claimed) * windows_per_subject
    background = X1[claimed_start:claimed_start + 16]
    background_tensor = torch.as_tensor(
        background, dtype=torch.float32, device=DEVICE
    )

    explainer = shap.GradientExplainer(SHAPWrapper(model), background_tensor)
    raw = explainer.shap_values(tensor, nsamples=SHAP_NSAMPLES)
    values = np.asarray(raw)
    values = np.squeeze(values)

    # Normalize common SHAP output shapes to channels x time.
    if values.ndim == 3:
        values = values[..., 0] if values.shape[-1] == 1 else values[0]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP array shape: {np.asarray(raw).shape}")

    channel_names = CHANNELS if len(CHANNELS) == values.shape[0] else [
        f"Channel_{i + 1}" for i in range(values.shape[0])
    ]
    importance = np.mean(np.abs(values), axis=1)
    order = np.argsort(importance)[::-1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / (
        f"claimed_{args.claimed}_source_{args.source}_"
        f"recording{args.recording}_window{args.window}.npz"
    )
    np.savez_compressed(
        output, shap_values=values, score_logit=score,
        threshold=threshold, decision=decision,
        claimed_identity=args.claimed, source_subject=args.source,
        recording=args.recording, window=args.window
    )

    print("=== Cross-identity EEG SHAP ===")
    print(f"Claimed identity/model: {args.claimed}")
    print(f"Sample source: {args.source}")
    print(f"Recording/window: {args.recording}/{args.window}")
    print(f"Score (logit): {score:.6f}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Decision: {decision}")
    print(f"Dataset label matches claimed identity: {int(y[global_idx]) == SUBJECTS.index(args.claimed)}")
    print("\nTop channels by mean absolute SHAP:")
    for i in order[:5]:
        print(f"  {channel_names[i]}: {importance[i]:.6f}")
    print(f"\nSaved SHAP values to: {output}")


if __name__ == "__main__":
    main()
