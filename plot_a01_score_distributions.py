import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.cnn_authentication_early_stopping import (
    EEG1DCNN,
    DEVICE,
    load_recording,
    standardize_windows,
    predict_scores,
)

ROOT = Path.cwd()
RESULTS = ROOT / "results" / "cnn_authentication_early_stopping"
SUBJECTS = [f"A{i:02d}" for i in range(1, 11)]
SUBJECT = "A01"
SUBJECT_INDEX = SUBJECTS.index(SUBJECT)

with (RESULTS / "cnn_authentication_results.json").open() as f:
    report = json.load(f)

threshold = float(report["validation_thresholds_by_subject"][SUBJECT])

# Load the same recordings and preprocessing used by the GUI.
X1, y1 = load_recording(1)
X2, y2 = load_recording(2)
X3, y3 = load_recording(3)
X1, X2, X3, mean, std = standardize_windows(X1, X2, X3)

model = EEG1DCNN().to(DEVICE)
checkpoint = RESULTS / "saved_models" / f"{SUBJECT}_cnn_state_dict.pt"
model.load_state_dict(
    torch.load(checkpoint, map_location=DEVICE, weights_only=True)
)
model.eval()

recordings = [
    ("Recording 2 — calibration", X2, y2),
    ("Recording 3 — held-out", X3, y3),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (title, X, y) in zip(axes, recordings):
    scores = predict_scores(model, X)
    genuine = scores[y == SUBJECT_INDEX]
    impostor = scores[y != SUBJECT_INDEX]

    ax.hist(genuine, bins=35, alpha=0.65, density=True,
            label=f"Genuine (n={len(genuine)})")
    ax.hist(impostor, bins=45, alpha=0.55, density=True,
            label=f"Impostor (n={len(impostor)})")
    ax.axvline(threshold, linestyle="--", linewidth=2,
               label=f"Frozen threshold = {threshold:.3f}")

    far = np.mean(impostor >= threshold)
    frr = np.mean(genuine < threshold)

    ax.set_title(f"{title}\nFAR={far:.2%} | FRR={frr:.2%}")
    ax.set_xlabel("CNN logit score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

fig.suptitle("A01 Authentication Score Distributions", fontsize=15)
fig.tight_layout()

output = ROOT / "a01_score_distributions.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(f"Plot saved to: {output}")
print(f"Frozen threshold: {threshold:.8f}")
plt.show()
