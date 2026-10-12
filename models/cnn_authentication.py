import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "preprocessed_sessions"
OUTPUT_DIR = ROOT / "results" / "cnn_authentication"
MODEL_DIR = OUTPUT_DIR / "saved_models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"A{i:02d}" for i in range(1, 11)]

EPOCHS_PER_WINDOW = 10
SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_recording(recording_number):
    all_windows = []
    all_labels = []

    for subject_index, subject in enumerate(SUBJECTS):
        path = (
            DATA_DIR
            / subject
            / f"recording_{recording_number}_epochs.npy"
        )

        if not path.exists():
            raise FileNotFoundError(f"Missing recording: {path}")

        epochs = np.load(path)

        n_windows = len(epochs) // EPOCHS_PER_WINDOW
        epochs = epochs[:n_windows * EPOCHS_PER_WINDOW]

        windows = epochs.reshape(
            n_windows,
            EPOCHS_PER_WINDOW,
            epochs.shape[1],
            epochs.shape[2],
        ).mean(axis=1)

        all_windows.append(windows.astype(np.float32))
        all_labels.extend([subject_index] * len(windows))

    return (
        np.concatenate(all_windows),
        np.asarray(all_labels, dtype=int),
    )


def standardize_windows(train, validation, test):
    # Fit normalization statistics using Recording 1 only.
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    train_scaled = (train - mean) / std
    validation_scaled = (validation - mean) / std
    test_scaled = (test - mean) / std

    return train_scaled, validation_scaled, test_scaled, mean, std


# --------------------------------------------------
# CNN ARCHITECTURE
# --------------------------------------------------

class EEG1DCNN(nn.Module):
    def __init__(self, n_channels=16):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.features(x).squeeze(1)


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def calculate_eer_threshold(genuine, impostor):
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)

    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("Both genuine and impostor scores are required.")

    labels = np.concatenate([
        np.ones(len(genuine), dtype=int),
        np.zeros(len(impostor), dtype=int),
    ])

    scores = np.concatenate([genuine, impostor])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    # Ignore infinite thresholds when choosing a usable operating point.
    finite = np.isfinite(thresholds)

    if not np.any(finite):
        raise ValueError("No finite ROC thresholds were generated.")

    finite_indices = np.flatnonzero(finite)
    index = finite_indices[
        np.argmin(np.abs(fpr[finite] - fnr[finite]))
    ]

    eer = float((fpr[index] + fnr[index]) / 2.0)
    threshold = float(thresholds[index])

    return eer, threshold


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

def train_subject_model(X_train, y_train):
    model = EEG1DCNN().to(DEVICE)

    positives = max(int(y_train.sum()), 1)
    negatives = max(int(len(y_train) - positives), 1)

    pos_weight = torch.tensor(
        [negatives / positives],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)

        if epoch in (0, 4, 9, 14, 19):
            print(
                f"    Epoch {epoch + 1}/{EPOCHS}, "
                f"loss={total_loss / len(dataset):.4f}"
            )

    return model


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_scores(model, X):
    model.eval()

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32)
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )

    outputs = []

    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(DEVICE))
            outputs.extend(logits.cpu().numpy().tolist())

    return np.asarray(outputs, dtype=float)


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():
    print("=" * 70)
    print("1D-CNN EEG AUTHENTICATION")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Training: Recording 1")
    print("Threshold calibration: Recording 2")
    print("Held-out testing: Recording 3")
    print("Input: raw EEG windows (16 channels x 256 samples)")
    print("Thresholds: calibrated separately for each subject")

    X_train, y_train = load_recording(1)
    X_val, y_val = load_recording(2)
    X_test, y_test = load_recording(3)

    X_train, X_val, X_test, norm_mean, norm_std = (
        standardize_windows(X_train, X_val, X_test)
    )

    print("Training windows:", X_train.shape)
    print("Validation windows:", X_val.shape)
    print("Test windows:", X_test.shape)

    # Save preprocessing statistics for future inference.
    np.savez(
        OUTPUT_DIR / "normalization_stats.npz",
        mean=norm_mean,
        std=norm_std,
    )

    models = []

    for subject_index, subject in enumerate(SUBJECTS):
        print(f"\nTraining CNN authenticator for {subject}...")

        binary_labels = (
            y_train == subject_index
        ).astype(np.float32)

        model = train_subject_model(X_train, binary_labels)
        models.append(model)

        # Save each trained model.
        torch.save(
            model.state_dict(),
            MODEL_DIR / f"{subject}_cnn_state_dict.pt",
        )

    print(f"\nSaved {len(models)} CNN models to: {MODEL_DIR}")

    # --------------------------------------------------
    # VALIDATION-BASED THRESHOLD CALIBRATION
    # --------------------------------------------------

    validation_thresholds = {}
    validation_eers = []
    per_subject_results = []

    test_genuine_decisions = []
    test_impostor_decisions = []
    test_eers_descriptive = []

    for subject_index, (subject, model) in enumerate(
        zip(SUBJECTS, models)
    ):
        val_scores = predict_scores(model, X_val)
        test_scores = predict_scores(model, X_test)

        val_mask = y_val == subject_index
        test_mask = y_test == subject_index

        val_genuine = val_scores[val_mask]
        val_impostor = val_scores[~val_mask]

        test_genuine = test_scores[test_mask]
        test_impostor = test_scores[~test_mask]

        # Choose this subject's threshold using Recording 2 only.
        val_eer, threshold = calculate_eer_threshold(
            val_genuine,
            val_impostor,
        )

        validation_thresholds[subject] = threshold
        validation_eers.append(val_eer)

        # Apply the frozen threshold to Recording 3.
        subject_far = float(
            np.mean(test_impostor >= threshold)
        )
        subject_frr = float(
            np.mean(test_genuine < threshold)
        )

        test_impostor_decisions.extend(
            (test_impostor >= threshold).tolist()
        )
        test_genuine_decisions.extend(
            (test_genuine < threshold).tolist()
        )

        # Descriptive only: test labels select this diagnostic EER.
        subject_test_eer, _ = calculate_eer_threshold(
            test_genuine,
            test_impostor,
        )

        test_eers_descriptive.append(subject_test_eer)

        per_subject_results.append({
            "subject": subject,
            "validation_eer": val_eer,
            "validation_threshold": threshold,
            "test_far_at_validation_threshold": subject_far,
            "test_frr_at_validation_threshold": subject_frr,
            "test_eer_descriptive_only": subject_test_eer,
            "test_genuine_count": int(len(test_genuine)),
            "test_impostor_count": int(len(test_impostor)),
        })

        print(
            f"{subject}: validation EER={val_eer * 100:.2f}%, "
            f"threshold={threshold:.4f}, "
            f"test FAR={subject_far * 100:.2f}%, "
            f"test FRR={subject_frr * 100:.2f}%"
        )

    # Average validation EER across the 10 subject-specific models.
    validation_eer = float(np.mean(validation_eers))

    # Aggregate error rates across all genuine and impostor decisions.
    test_far = float(np.mean(test_impostor_decisions))
    test_frr = float(np.mean(test_genuine_decisions))

    # Mean of subject-specific test EERs; diagnostic only.
    test_eer_descriptive = float(
        np.mean(test_eers_descriptive)
    )

    total_test_genuine = len(test_genuine_decisions)
    total_test_impostor = len(test_impostor_decisions)

    results = {
        "method": "1D-CNN on averaged raw EEG windows",
        "input_shape_per_window": [16, 256],
        "subjects": SUBJECTS,
        "enrollment_recording": 1,
        "threshold_calibration_recording": 2,
        "held_out_test_recording": 3,
        "epochs_per_window": EPOCHS_PER_WINDOW,
        "training_epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "device": str(DEVICE),
        "validation_eer_mean_across_subjects": validation_eer,
        "validation_thresholds_by_subject": validation_thresholds,
        "test_far_at_subject_validation_thresholds": test_far,
        "test_frr_at_subject_validation_thresholds": test_frr,
        "test_eer_mean_descriptive_only": test_eer_descriptive,
        "test_genuine_score_count": total_test_genuine,
        "test_impostor_score_count": total_test_impostor,
        "threshold_protocol": (
            "Each subject's threshold is calibrated using Recording 2 "
            "and frozen for Recording 3."
        ),
        "test_eer_note": (
            "Descriptive only. Test labels are used to calculate "
            "test EER and do not determine operating thresholds."
        ),
        "preprocessing": (
            "Channel-wise normalization statistics fitted on "
            "Recording 1 only; applied unchanged to Recordings 2 and 3."
        ),
        "saved_model_directory": str(MODEL_DIR),
        "per_subject_results": per_subject_results,
    }

    output_path = OUTPUT_DIR / "cnn_authentication_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("CNN RESULTS")
    print("=" * 70)

    print(f"Mean validation EER: {validation_eer * 100:.2f}%")
    print(
        "Test FAR at subject-specific validation thresholds: "
        f"{test_far * 100:.2f}%"
    )
    print(
        "Test FRR at subject-specific validation thresholds: "
        f"{test_frr * 100:.2f}%"
    )
    print(
        "Mean descriptive test EER (test-selected thresholds): "
        f"{test_eer_descriptive * 100:.2f}%"
    )
    print("Test genuine decisions:", total_test_genuine)
    print("Test impostor decisions:", total_test_impostor)
    print("Results saved to:", output_path)
    print("Normalization statistics saved to:",
          OUTPUT_DIR / "normalization_stats.npz")


if __name__ == "__main__":
    main()
