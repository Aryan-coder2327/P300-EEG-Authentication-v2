# NEURO/LOCK — P300 EEG Authentication GUI

A minimalist, browser-based interface for the existing P300 EEG authentication project. It is deliberately **not a Tkinter desktop window**: FastAPI serves a responsive HTML/CSS/JavaScript application in the browser.

## Features

- **Overview:** checks all 30 preprocessed recording files, ten early-stopping CNN checkpoints and the saved subject thresholds.
- **Enrol identity:** stores a named reference cohort (Recording 1 window range) in `results/gui/enrolled_templates.json`.
- **Authenticate:** evaluates a selected recorded window against the enrolled subject's existing CNN and frozen threshold.
- **Imposter lab:** tests a different source subject against the claimed identity; displays both the model decision and expected outcome, including false accepts.
- **Explainability:** calls the existing `models.local_shap_explainer.explain_window` pipeline and displays per-window channel/time SHAP attributions, ranked channels and compact signal previews.
- **Stream simulator:** evaluates consecutive recorded windows and shows each decision as a playback sequence.
- **Audit trail:** session event log with JSON export.

## Important model semantics

The GUI uses only `results/cnn_authentication_early_stopping/saved_models/{subject}_cnn_state_dict.pt` and thresholds in `results/cnn_authentication_early_stopping/cnn_authentication_results.json`. It does **not** route to the older PCA/SVM experiment.

Enrolment is intentionally represented honestly: it saves an identity record and selected reference-window indices. It does not retrain the CNN, create a new learned embedding, or recalibrate thresholds. Authentication is still the existing subject-specific CNN with its frozen Recording-2-calibrated threshold. The GUI refuses to authenticate until that subject has an enrolment record.

All sample indices are local to a subject, `0–152`. Recording 3 is the held-out test recording; Recording 2 is available as a calibration/demo option and must not be represented as held-out evaluation. Scores are CNN **logits**, not probabilities.

## Install and run

From the **root of your existing repository** (the directory containing `models/`, `data/`, and `results/`):

```bash
# Activate your existing virtual environment first.
pip install -r requirements-gui.txt
uvicorn gui_app:app --reload
```

Open `http://127.0.0.1:8000`. API docs are at `http://127.0.0.1:8000/api/docs`.

Copy these files into your existing repository:

```text
gui_app.py
requirements-gui.txt
README_GUI.md
gui_static/
  index.html
  style.css
  app.js
```

The project must already contain:

```text
models/cnn_authentication_early_stopping.py
models/local_shap_explainer.py
data/preprocessed_sessions/A01/recording_1_epochs.npy  # through A10, recordings 1–3
results/cnn_authentication_early_stopping/cnn_authentication_results.json
results/cnn_authentication_early_stopping/saved_models/A01_cnn_state_dict.pt  # through A10
```

The SHAP dependency (`shap`) and model/data dependencies (`torch`, `numpy`, `scikit-learn`, `pandas`, etc.) should be installed in the existing project environment as required by the original project. `requirements-gui.txt` adds only the web server dependencies; it intentionally does not pin or replace the project's ML stack.

## Demo sequence

1. Start the server and check **System overview → Pipeline readiness**. All 30 `.npy` files, all 10 `.pt` checkpoints and the CNN result JSON should be present.
2. Open **Enrol identity**; choose A01 and save a small R1 reference cohort.
3. Open **Authenticate**; choose A01 for both claimed identity and sample source. Use R3 and a window index such as 32 for a genuine-sample case.
4. Open **Imposter lab**; claim A01 but choose A02 as the sample source. A `REJECT` is the expected outcome; the tool shows if the model instead falsely accepts.
5. Open **Explainability** and explain the same subject/source/recording/window combination. SHAP is computed on demand and can take a little while on CPU.
6. Open **Stream simulator** to process a short sequence, then **Audit trail** to export the session event log.

## Limitations

- This is an offline recorded-data demo, not live headset acquisition.
- The current enrolment is a reference-cohort record, not a production-grade dynamic template lifecycle.
- No physical lock, liveness detection, replay defence, secure user store, access-control hardening or network authentication is implemented.
- The session audit view is not a tamper-proof security log.
- SHAP attributes the model logit; it does not prove causality or biological identity.
