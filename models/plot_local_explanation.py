"""Plot the EEG waveform and SHAP heatmap for one saved local explanation."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.local_shap_explainer import (
    CHANNELS,
    OUTPUT_DIR,
    explain_window,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="A01")
    parser.add_argument("--recording", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--window", type=int, default=32)
    args = parser.parse_args()

    result = explain_window(
        args.subject, args.window, args.recording
    )

    eeg = result["input_window"]
    shap_values = result["shap_values"]
    times = np.arange(eeg.shape[1]) / result["sampling_rate_hz"]

    out_dir = OUTPUT_DIR / "on_demand" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.subject}_recording{args.recording}_window{args.window}"
    )

    # Plot all 16 EEG channels with vertical offsets for readability.
    fig, ax = plt.subplots(figsize=(13, 7))
    offsets = np.arange(len(CHANNELS)) * 5.0
    for i, channel in enumerate(CHANNELS):
        signal = eeg[i]
        scale = np.std(signal)
        if scale > 0:
            signal = signal / scale
        ax.plot(times, signal + offsets[i], linewidth=0.8)
    ax.set_yticks(offsets)
    ax.set_yticklabels(CHANNELS)
    ax.set_xlabel("Time within EEG window (seconds)")
    ax.set_title(
        f"{args.subject} EEG — Recording {args.recording}, "
        f"Window {args.window}"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_waveform.png", dpi=200)
    plt.close(fig)

    # Signed SHAP: positive and negative values indicate opposite
    # directions of contribution to the model's output.
    limit = float(np.max(np.abs(shap_values)))
    if limit == 0:
        limit = 1e-12

    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(
        shap_values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[times[0], times[-1], -0.5, len(CHANNELS) - 0.5],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_yticks(np.arange(len(CHANNELS)))
    ax.set_yticklabels(CHANNELS)
    ax.set_xlabel("Time within EEG window (seconds)")
    ax.set_ylabel("EEG channel")
    ax.set_title(
        f"Local SHAP — {result['decision']} "
        f"(score {result['score_logit']:.3f}, "
        f"threshold {result['validation_threshold']:.3f})"
    )
    fig.colorbar(image, ax=ax, label="Signed SHAP contribution to CNN logit")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_shap_heatmap.png", dpi=200)
    plt.close(fig)

    print("\nDecision:", result["decision"])
    print("Score:", round(result["score_logit"], 4))
    print("Threshold:", round(result["validation_threshold"], 4))
    print("\nPlain-English explanation:")
    print(result["plain_english"])
    print("\nSaved plots:")
    print(out_dir / f"{stem}_waveform.png")
    print(out_dir / f"{stem}_shap_heatmap.png")


if __name__ == "__main__":
    main()
