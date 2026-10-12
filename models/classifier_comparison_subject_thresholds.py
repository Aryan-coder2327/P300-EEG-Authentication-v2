import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from classifier_comparison import (
    SUBJECTS,
    PCA_COMPONENTS,
    RANDOM_STATE,
    build_authenticator,
    calculate_eer_threshold,
    load_recording,
    score_model,
    rates_at_threshold,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "results" / "classifier_comparison_subject_thresholds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("SUBJECT-SPECIFIC THRESHOLD CLASSIFIER COMPARISON")
    print("Enrollment: Recording 1")
    print("Threshold calibration: Recording 2")
    print("Held-out test: Recording 3")

    X_train, y_train = load_recording(1)
    X_val, y_val = load_recording(2)
    X_test, y_test = load_recording(3)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    classifier_names = [
        "SVM",
        "Logistic Regression",
        "LDA",
        "Random Forest",
    ]

    results = {
        "protocol": {
            "enrollment_recording": 1,
            "threshold_calibration_recording": 2,
            "held_out_test_recording": 3,
            "thresholds": "One independently calibrated threshold per claimed subject",
            "scaler_and_pca_fit_on": "Recording 1 only",
            "test_selected_thresholds_used_for_operating_metrics": False,
            "note": (
                "Classifiers use the same 224 engineered features and PCA "
                "pipeline. Hyperparameter tuning is not equally exhaustive "
                "across classifiers."
            ),
        },
        "classifiers": {},
    }

    for name in classifier_names:
        print(f"\nEvaluating {name}...")
        models = []

        for subject_index in range(len(SUBJECTS)):
            labels = (y_train == subject_index).astype(int)
            model = build_authenticator(name)
            model.fit(X_train, labels)
            models.append(model)

        subject_results = []
        all_impostor_accepts = []
        all_genuine_rejects = []
        validation_eers = []
        diagnostic_test_eers = []

        for subject_index, (subject, model) in enumerate(zip(SUBJECTS, models)):
            val_scores = score_model(model, X_val)
            test_scores = score_model(model, X_test)

            val_genuine = val_scores[y_val == subject_index]
            val_impostor = val_scores[y_val != subject_index]
            test_genuine = test_scores[y_test == subject_index]
            test_impostor = test_scores[y_test != subject_index]

            val_eer, threshold = calculate_eer_threshold(
                val_genuine, val_impostor
            )
            test_far, test_frr = rates_at_threshold(
                test_genuine, test_impostor, threshold
            )

            # These decisions use only the frozen Recording 2 threshold.
            all_impostor_accepts.extend(
                (test_impostor >= threshold).tolist()
            )
            all_genuine_rejects.extend(
                (test_genuine < threshold).tolist()
            )

            # Test-selected EER is diagnostic only, never an operating metric.
            diagnostic_test_eer, _ = calculate_eer_threshold(
                test_genuine, test_impostor
            )

            validation_eers.append(val_eer)
            diagnostic_test_eers.append(diagnostic_test_eer)

            subject_results.append({
                "subject": subject,
                "validation_eer": val_eer,
                "validation_threshold": threshold,
                "test_far_at_validation_threshold": test_far,
                "test_frr_at_validation_threshold": test_frr,
                "test_eer_descriptive_only": diagnostic_test_eer,
                "test_genuine_count": int(len(test_genuine)),
                "test_impostor_count": int(len(test_impostor)),
            })

            print(
                f"  {subject}: val EER={val_eer * 100:.2f}%, "
                f"test FAR={test_far * 100:.2f}%, "
                f"test FRR={test_frr * 100:.2f}%"
            )

        results["classifiers"][name] = {
            "validation_eer_mean_across_subjects": float(
                np.mean(validation_eers)
            ),
            "test_far_at_subject_validation_thresholds": float(
                np.mean(all_impostor_accepts)
            ),
            "test_frr_at_subject_validation_thresholds": float(
                np.mean(all_genuine_rejects)
            ),
            "test_eer_mean_descriptive_only": float(
                np.mean(diagnostic_test_eers)
            ),
            "per_subject": subject_results,
        }

        summary = results["classifiers"][name]
        print(
            f"  Mean validation EER: "
            f"{summary['validation_eer_mean_across_subjects'] * 100:.2f}%"
        )
        print(
            f"  Test FAR: "
            f"{summary['test_far_at_subject_validation_thresholds'] * 100:.2f}%"
        )
        print(
            f"  Test FRR: "
            f"{summary['test_frr_at_subject_validation_thresholds'] * 100:.2f}%"
        )

    output_path = OUTPUT_DIR / "subject_threshold_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved corrected evaluation to: {output_path}")


if __name__ == "__main__":
    main()
