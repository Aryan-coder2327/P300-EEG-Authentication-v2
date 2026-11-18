import json
from pathlib import Path

import numpy as np
import torch

from models.cnn_authentication_early_stopping import (
    EEG1DCNN,
    SUBJECTS,
    DATA_DIR,
    EPOCHS_PER_WINDOW,
    DEVICE,
    predict_scores,
    load_recording,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "cnn_authentication_early_stopping"
MODEL_DIR = OUT / "saved_models"
REPORT = OUT / "cnn_authentication_results.json"
STATS = OUT / "normalization_stats.npz"

# Load the frozen report and normalization statistics.
saved = json.loads(REPORT.read_text())
thresholds = saved["validation_thresholds_by_subject"]

with np.load(STATS) as z:
    mean = z["mean"]
    std = z["std"]

# Load only held-out Recording 3. No training or threshold fitting.
X_test, y_test = load_recording(3)
X_test = ((X_test - mean) / np.where(std < 1e-8, 1.0, std)).astype(np.float32)

print("Held-out Recording 3:", X_test.shape)
print("Device:", DEVICE)
print("\nPer-subject results using frozen thresholds")
print("-" * 76)

all_genuine_rejects = []
all_impostor_accepts = []
all_genuine_count = 0
all_impostor_count = 0

for i, subject in enumerate(SUBJECTS):
    checkpoint = MODEL_DIR / f"{subject}_cnn_state_dict.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = EEG1DCNN(n_channels=16).to(DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)

    scores = predict_scores(model, X_test)
    genuine = scores[y_test == i]
    impostor = scores[y_test != i]
    threshold = float(thresholds[subject])

    genuine_rejects = int(np.sum(genuine < threshold))
    impostor_accepts = int(np.sum(impostor >= threshold))

    frr = genuine_rejects / len(genuine)
    far = impostor_accepts / len(impostor)

    all_genuine_rejects.append(genuine_rejects)
    all_impostor_accepts.append(impostor_accepts)
    all_genuine_count += len(genuine)
    all_impostor_count += len(impostor)

    print(
        f"{subject}: threshold={threshold: .6f} | "
        f"genuine rejected={genuine_rejects}/{len(genuine)} "
        f"(FRR {frr*100:.3f}%) | "
        f"impostor accepted={impostor_accepts}/{len(impostor)} "
        f"(FAR {far*100:.3f}%)"
    )

far_total = sum(all_impostor_accepts) / all_impostor_count
frr_total = sum(all_genuine_rejects) / all_genuine_count

print("\n" + "=" * 76)
print(f"Total genuine rejects: {sum(all_genuine_rejects)}/{all_genuine_count}")
print(f"Total impostor accepts: {sum(all_impostor_accepts)}/{all_impostor_count}")
print(f"Recomputed FAR: {far_total*100:.6f}%")
print(f"Saved FAR:      {saved['test_far_at_subject_validation_thresholds']*100:.6f}%")
print(f"Recomputed FRR: {frr_total*100:.6f}%")
print(f"Saved FRR:      {saved['test_frr_at_subject_validation_thresholds']*100:.6f}%")
print("Thresholds loaded from saved report; none were recalibrated.")
