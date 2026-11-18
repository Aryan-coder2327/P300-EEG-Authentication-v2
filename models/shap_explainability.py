
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.cnn_authentication_early_stopping import (
    ROOT, SUBJECTS, DEVICE, EEG1DCNN,
    load_recording, standardize_windows,
)

SEED = 42
BACKGROUND_SIZE = 16
SAMPLES_PER_CLASS = 10
SHAP_NSAMPLES = 50  # Smoke-test setting; increase to 50 after verifying runtime.

CHANNELS = [
    "Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8",
    "F3", "F4", "FCz", "C3", "C4", "CP3", "CPz", "CP4",
]

OUTPUT = ROOT / "results" / "shap_explainability"
VALUES_DIR = OUTPUT / "shap_values"
PLOTS_DIR = OUTPUT / "plots"
LOCAL_DIR = PLOTS_DIR / "local_examples"

for folder in (OUTPUT, VALUES_DIR, PLOTS_DIR, LOCAL_DIR):
    folder.mkdir(parents=True, exist_ok=True)


class SHAPWrapper(torch.nn.Module):
    """Give SHAP a 2D (batch, one-output) tensor."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(1)


def choose(rng, indices, count):
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= count:
        return np.sort(indices)
    return np.sort(rng.choice(indices, count, replace=False))


def load_model(subject):
    path = (
        ROOT / "results" / "cnn_authentication_early_stopping"
        / "saved_models" / f"{subject}_cnn_state_dict.pt"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing saved CNN weights: {path}")

    base = EEG1DCNN().to(DEVICE)
    base.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True)
    )
    base.eval()
    return base, SHAPWrapper(base).to(DEVICE).eval()


def normalize_shap(values, expected_samples):
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError(f"Expected one output, got {len(values)}")
        values = values[0]

    values = np.asarray(values)

    # Current tested SHAP output: (samples, channels, time, 1).
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]

    if values.ndim != 3:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")
    if values.shape[0] != expected_samples:
        raise ValueError(f"Unexpected sample dimension: {values.shape}")
    if values.shape[1] != len(CHANNELS):
        raise ValueError(f"Unexpected channel dimension: {values.shape}")

    return values.astype(np.float32)


def save_local_plots(subject, sample_type, decision, values):
    tag = f"{subject}_{sample_type}_{decision}".lower()
    channel_scores = np.abs(values).mean(axis=1)
    order = np.argsort(channel_scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.asarray(CHANNELS)[order], channel_scores[order])
    ax.set_xlabel("Mean absolute SHAP value (CNN logit)")
    ax.set_title(f"{subject}: {sample_type}, decision={decision}")
    fig.tight_layout()
    fig.savefig(LOCAL_DIR / f"{tag}_channels.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(values, aspect="auto", origin="lower")
    ax.set_yticks(np.arange(len(CHANNELS)))
    ax.set_yticklabels(CHANNELS)
    ax.set_xlabel("Time sample (256 Hz)")
    ax.set_ylabel("EEG channel")
    ax.set_title(f"{subject}: signed local SHAP values")
    fig.colorbar(im, ax=ax, label="SHAP value (CNN logit)")
    fig.tight_layout()
    fig.savefig(LOCAL_DIR / f"{tag}_heatmap.png", dpi=180)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    print("Device:", DEVICE)
    print("Loading recordings; no training will be performed.")

    X1, y1 = load_recording(1)
    X2, y2 = load_recording(2)
    X3, y3 = load_recording(3)
    X1, X2, X3, mean, std = standardize_windows(X1, X2, X3)

    result_path = (
        ROOT / "results" / "cnn_authentication_early_stopping"
        / "cnn_authentication_results.json"
    )
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)

    thresholds = result["validation_thresholds_by_subject"]
    bg_idx = choose(rng, np.arange(len(X1)), BACKGROUND_SIZE)
    background = torch.tensor(X1[bg_idx], dtype=torch.float32, device=DEVICE)

    all_values, all_types, all_subjects = [], [], []
    group_rows, subject_rows, local_rows = [], [], []
    subject_summaries = []

    for subject_idx, subject in enumerate(SUBJECTS):
        print(f"\nExplaining {subject}...")
        base_model, wrapped_model = load_model(subject)

        genuine = choose(rng, np.flatnonzero(y3 == subject_idx), SAMPLES_PER_CLASS)
        impostor = choose(rng, np.flatnonzero(y3 != subject_idx), SAMPLES_PER_CLASS)
        selected = np.concatenate([genuine, impostor])
        types = np.concatenate([
            np.ones(len(genuine), dtype=int),
            np.zeros(len(impostor), dtype=int),
        ])

        permutation = rng.permutation(len(selected))
        selected, types = selected[permutation], types[permutation]
        X = X3[selected]
        x_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)

        explainer = shap.GradientExplainer(wrapped_model, background)
        raw = explainer.shap_values(x_tensor, nsamples=SHAP_NSAMPLES)
        values = normalize_shap(raw, len(selected))

        np.savez_compressed(
            VALUES_DIR / f"{subject}_shap_values.npz",
            shap_values=values,
            inputs=X.astype(np.float32),
            genuine_labels=types,
            recording_indices=selected,
            channels=np.asarray(CHANNELS),
        )

        with torch.no_grad():
            scores = base_model(x_tensor).cpu().numpy()
        threshold = float(thresholds[subject])
        abs_values = np.abs(values)

        all_values.append(values)
        all_types.extend(types.tolist())
        all_subjects.extend([subject] * len(types))

        for group, mask in (
            ("all", np.ones(len(types), dtype=bool)),
            ("genuine", types == 1),
            ("impostor", types == 0),
        ):
            if not mask.any():
                continue
            for c, channel in enumerate(CHANNELS):
                group_rows.append({
                    "subject": subject,
                    "sample_group": group,
                    "channel": channel,
                    "mean_absolute_shap": float(abs_values[mask, c, :].mean()),
                    "mean_signed_shap": float(values[mask, c, :].mean()),
                    "sample_count": int(mask.sum()),
                })

        for c, channel in enumerate(CHANNELS):
            subject_rows.append({
                "subject": subject,
                "channel": channel,
                "mean_absolute_shap": float(abs_values[:, c, :].mean()),
                "mean_signed_shap": float(values[:, c, :].mean()),
            })

        for type_value, type_name in ((1, "genuine"), (0, "impostor")):
            candidates = np.flatnonzero(types == type_value)
            if not len(candidates):
                continue
            if type_value == 1:
                preferred = candidates[scores[candidates] >= threshold]
            else:
                preferred = candidates[scores[candidates] < threshold]
            i = int(preferred[0] if len(preferred) else candidates[0])
            decision = "ACCEPT" if scores[i] >= threshold else "REJECT"

            save_local_plots(subject, type_name, decision, values[i])
            local_rows.append({
                "subject": subject,
                "sample_type": type_name,
                "decision": decision,
                "score_logit": float(scores[i]),
                "validation_threshold": threshold,
                "recording": 3,
                "window_index": int(selected[i]),
                "mean_absolute_shap": float(abs_values[i].mean()),
            })

        subject_summaries.append({
            "subject": subject,
            "samples_explained": int(len(selected)),
            "genuine_samples": int((types == 1).sum()),
            "impostor_samples": int((types == 0).sum()),
            "mean_absolute_shap": float(abs_values.mean()),
            "validation_threshold": threshold,
        })

        print(f"  SHAP shape: {values.shape}; mean |SHAP|={abs_values.mean():.6g}")
        del explainer, x_tensor, wrapped_model, base_model

    combined = np.concatenate(all_values, axis=0)
    combined_types = np.asarray(all_types)
    abs_combined = np.abs(combined)

    channel_df = pd.DataFrame({
        "channel": CHANNELS,
        "mean_absolute_shap": abs_combined.mean(axis=(0, 2)),
        "mean_signed_shap": combined.mean(axis=(0, 2)),
    }).sort_values("mean_absolute_shap", ascending=False)
    channel_df.to_csv(OUTPUT / "channel_importance.csv", index=False)
    pd.DataFrame(group_rows).to_csv(
        OUTPUT / "genuine_vs_impostor_channel_importance.csv", index=False
    )
    pd.DataFrame(subject_rows).to_csv(
        OUTPUT / "subject_channel_importance.csv", index=False
    )
    pd.DataFrame(local_rows).to_csv(
        OUTPUT / "local_decision_examples.csv", index=False
    )

    time_importance = abs_combined.mean(axis=(0, 1))
    pd.DataFrame({
        "sample_index": np.arange(len(time_importance)),
        "time_seconds": np.arange(len(time_importance)) / 256.0,
        "mean_absolute_shap": time_importance,
    }).to_csv(OUTPUT / "time_importance.csv", index=False)

    # Global channel ranking.
    plot_df = channel_df.sort_values("mean_absolute_shap")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["channel"], plot_df["mean_absolute_shap"])
    ax.set_xlabel("Mean absolute SHAP value (CNN logit)")
    ax.set_title("Global EEG Channel Importance")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "global_channel_importance.png", dpi=200)
    plt.close(fig)

    # Global time importance.
    times = np.arange(len(time_importance)) / 256.0
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(times, time_importance)
    ax.set_xlabel("Time within window (seconds)")
    ax.set_ylabel("Mean absolute SHAP value")
    ax.set_title("Global Temporal Importance")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "global_time_importance.png", dpi=200)
    plt.close(fig)

    # Global channel-by-time heatmap.
    channel_time = abs_combined.mean(axis=0)
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(channel_time, aspect="auto", origin="lower")
    ax.set_yticks(np.arange(len(CHANNELS)))
    ax.set_yticklabels(CHANNELS)
    ax.set_xlabel("Time sample")
    ax.set_ylabel("EEG channel")
    ax.set_title("Global Channel-by-Time SHAP Magnitude")
    fig.colorbar(im, ax=ax, label="Mean absolute SHAP")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "channel_time_heatmap.png", dpi=200)
    plt.close(fig)

    # Compare genuine and impostor channel attributions.
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CHANNELS))
    width = 0.38
    for offset, (label, value) in enumerate((
        ("Genuine", 1), ("Impostor", 0)
    )):
        mask = combined_types == value
        importance = abs_combined[mask].mean(axis=(0, 2))
        ax.bar(x + (offset - 0.5) * width, importance, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS, rotation=45)
    ax.set_ylabel("Mean absolute SHAP")
    ax.set_title("Genuine vs Impostor Channel Importance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "genuine_vs_impostor_channels.png", dpi=200)
    plt.close(fig)

    summary = {
        "method": "SHAP GradientExplainer",
        "model": "Saved early-stopping subject-specific 1D-CNN",
        "output_explained": "CNN logit",
        "background_recording": 1,
        "explained_recording": 3,
        "background_size": int(len(bg_idx)),
        "samples_per_class_per_subject": SAMPLES_PER_CLASS,
        "shap_nsamples": SHAP_NSAMPLES,
        "total_explained_samples": int(len(combined)),
        "top_channels": channel_df.head(5).to_dict(orient="records"),
        "subject_summaries": subject_summaries,
        "limitations": [
            "Attributions are model explanations, not causal evidence.",
            "The output explained is a logit, not a calibrated probability.",
            "Recording 3 explanations are post-hoc and do not tune the model.",
            "Genuine/impostor groups and subjects may differ.",
        ],
    }
    with open(OUTPUT / "shap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSHAP analysis complete.")
    print(channel_df.head(10).to_string(index=False))
    print("Outputs:", OUTPUT)


if __name__ == "__main__":
    main()
