# EEG-Based P300 Authentication with Explainable AI and IoT Integration

A research prototype exploring EEG-based biometric authentication using P300 event-related potentials (ERPs), subject-specific CNN models, Explainable AI (SHAP), and an MQTT-connected IoT smart-lock simulation.

The project uses publicly available EEG recordings to evaluate authentication decisions without requiring a physical EEG headset. Recorded EEG windows are replayed through a simulated authentication workflow, and the resulting ACCEPT/REJECT decision can be sent to a simulated smart lock.

---

## Overview

Traditional authentication systems rely on passwords, fingerprints, or facial recognition. This project investigates EEG-based biometric authentication using P300 responses, which are associated with stimulus-related brain activity.

The system combines:

- P300 EEG data preprocessing
- Subject-specific CNN authentication models
- Enrollment and verification workflows
- Threshold-based authentication decisions
- Impostor testing
- Explainable AI using SHAP
- Recorded EEG stream simulation
- MQTT-based IoT smart-lock integration
- Authentication event logging

> **Important:** This is an academic research prototype. It does not acquire live EEG signals and is not a production-ready security system.

---

## Features

### EEG Authentication
- Subject-specific CNN models for A01–A10
- Genuine and impostor authentication testing
- Threshold-based ACCEPT/REJECT decisions
- Recording-based authentication experiments

### Browser-Based GUI
- System dashboard
- Identity enrollment
- Authentication
- Impostor testing
- Explainability Lab
- Stream Simulator
- Audit Trail

### Explainable AI
- SHAP-based analysis of authentication decisions
- EEG channel importance
- Time-region importance
- Local explanations for individual authentication windows

### IoT Integration
- MQTT-based communication
- ESP32 smart-lock simulation
- ACCEPT → Unlock
- REJECT → Lock

---

## System Architecture

```text
             Public EEG Dataset
                     |
                     v
          EEG Preprocessing Pipeline
                     |
                     v
       Preprocessed EEG Recordings
                     |
                     v
          Subject-Specific CNN Models
                     |
                     v
       Threshold-Based Authentication
                     |
          +----------+-----------+
          |                      |
          v                      v
    ACCEPT / REJECT         SHAP Analysis
          |
          v
       MQTT Communication
          |
          v
       ESP32 Simulation
          |
          v
       Smart Lock
```

---

## Repository Structure

```text
.
├── data/
│   ├── raw/                         # Raw EEG files (not included)
│   ├── preprocessed_sessions/         # Preprocessed recordings (not included)
│   └── README.md
│
├── preprocessing/
│   ├── P300_preprocessing.ipynb
│   ├── session_preprocessing_all.py
│   ├── session_preprocessing_test.py
│   └── README.md
│
├── models/
│   ├── authentication.py
│   ├── cnn_authentication.py
│   ├── cnn_authentication_experiment.py
│   ├── cnn_authentication_early_stopping.py
│   ├── evaluation_metrics.py
│   ├── feature_extractor.py
│   ├── ml_authenticators.py
│   ├── cross_session_authentication.py
│   ├── cross_identity_shap.py
│   ├── local_shap_explainer.py
│   ├── shap_explainability.py
│   └── other research and experimental models
│
├── gui_static/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── smart_lock/
│   ├── sketch.ino
│   ├── diagram.json
│   ├── libraries.txt
│   └── wokwi-project.txt
│
├── results/
│   ├── cnn_authentication/
│   ├── cnn_authentication_experiment/
│   ├── cnn_authentication_early_stopping/
│   └── shap_explainability/
│
├── tests/
│
├── gui_app.py
├── iot_simulator.py
├── demo.py
├── requirements.txt
├── README.md
└── README_GUI.md
```

---

# Dataset

## BNCI Visual P300 Speller Dataset

This project uses the BNCI Visual P300 Speller EEG dataset.

Dataset source:

https://bnci-horizon-2020.eu/database/data-sets

The preprocessing pipeline expects the recordings for subjects A01–A10.

### Required raw files

Place the downloaded files inside:

```text
data/raw/
```

The filenames expected by the preprocessing script are:

```text
data/raw/
├── A01S.mat
├── A02S.mat
├── A03S.mat
├── A04S.mat
├── A05S.mat
├── A06S.mat
├── A07S.mat
├── A08S.mat
├── A09S.mat
└── A10S.mat
```

Raw EEG data is not included in this repository.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Aryan-coder2327/P300-EEG-Authentication-v2.git
```

Navigate into the project:

```bash
cd P300-EEG-Authentication-v2
```

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
```

Activate:

```powershell
venv\Scripts\activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

---

# Data Preprocessing

## 1. Download the raw EEG data

Download the required recordings from the BNCI database and place the files in:

```text
data/raw/
```

Ensure the files are named:

```text
A01S.mat
A02S.mat
...
A10S.mat
```

## 2. Run preprocessing

From the project root:

```bash
python preprocessing/session_preprocessing_all.py
```

The script processes the recordings for subjects A01–A10.

---

## Preprocessing Pipeline

The preprocessing pipeline performs:

1. Loading EEG recordings from MATLAB `.mat` files
2. EEG channel identification
3. Band-pass filtering
4. Flash/stimulus onset detection
5. EEG epoch extraction
6. Baseline correction
7. P300 ERP processing

### Signal Processing Configuration

| Parameter | Value |
|---|---|
| Sampling frequency | 256 Hz |
| Band-pass filter | 0.1–30 Hz |
| Filter | 4th-order Butterworth |
| Baseline interval | −200 ms |
| Post-stimulus interval | 800 ms |
| Epoch duration | 1 second |

---

## Preprocessed Data Structure

Preprocessed data is stored in:

```text
data/preprocessed_sessions/
```

The directory structure generated by the preprocessing pipeline is:

```text
data/preprocessed_sessions/
├── A01/
│   ├── recording_1_epochs.npy
│   ├── recording_1_metadata.json
│   ├── recording_2_epochs.npy
│   ├── recording_2_metadata.json
│   ├── recording_3_epochs.npy
│   ├── recording_3_metadata.json
├── A02/
│   └── ...
└── A10/
```

Each recording contains its preprocessed EEG epochs and associated metadata.

The GUI uses the preprocessed recordings for authentication, explainability, and stream simulation.

---

# Authentication System

The project contains multiple machine-learning experiments, including CNN-based authentication and classical machine-learning approaches.

The browser-based authentication interface specifically uses the **CNN Early Stopping model pipeline**.

## CNN Authentication

The GUI uses:

```text
models/cnn_authentication_early_stopping.py
```

Model checkpoints:

```text
results/cnn_authentication_early_stopping/saved_models/
```

Authentication results and thresholds:

```text
results/cnn_authentication_early_stopping/
    cnn_authentication_results.json
```

The system evaluates an EEG window using the CNN associated with the claimed identity.

The model generates a score, which is evaluated against the subject's authentication threshold.

```text
EEG Window
     |
     v
Subject-Specific CNN
     |
     v
Authentication Score (Logit)
     |
     v
Frozen Threshold
     |
     +------ Score meets threshold ------> ACCEPT
     |
     +------ Score below threshold -----> REJECT
```

### Important Model Details

- The system uses subject-specific models.
- Authentication scores are CNN logits, not probabilities.
- Thresholds are calibrated using the validation/calibration workflow.
- The GUI does not retrain the CNN during enrollment.
- Enrollment stores an identity record and selected reference-window indices.
- Enrollment does not create a new neural-network embedding or recalibrate the authentication threshold.

---

# GUI

The project includes a browser-based interface powered by FastAPI.

The GUI is not a Tkinter desktop application. The backend serves the web interface and authentication APIs.

## Start the GUI

From the repository root, with the virtual environment activated:

```bash
python -m uvicorn gui_app:app --reload
```

Open the application:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/api/docs

---

## GUI Modules

### Dashboard
Displays system readiness and checks the availability of required datasets, CNN checkpoints, and authentication result files.

### Enrol Identity
Creates an identity enrollment record using selected reference windows.

Enrollment is a reference-cohort registration process. It does not retrain the CNN or generate a new model.

### Authenticate
Tests a selected EEG recording window against the claimed identity's CNN and frozen threshold.

### Impostor Lab
Tests EEG samples originating from another subject while claiming a different identity.

The interface displays the authentication decision and allows false accepts to be observed.

### Explainability Lab
Uses the local SHAP explanation pipeline to analyze an authentication window.

It provides insights into:
- Channel importance
- Time-region importance
- Window-level feature contributions

SHAP explanations are computed on demand and may take time on CPU.

### Stream Simulator
Replays consecutive recorded EEG windows to simulate an authentication stream.

This is recorded-data replay—not live EEG acquisition.

### Audit Trail
Displays authentication events generated during the session and supports exporting the event log.

The audit trail is a demonstration log, not a tamper-proof security logging system.

---

# IoT Smart Lock

The authentication system can communicate with an ESP32 smart-lock simulation using MQTT.

## Communication Flow

```text
CNN Authentication
       |
       v
 ACCEPT / REJECT
       |
       v
 MQTT Message
       |
       v
 ESP32
       |
       v
 Servo Lock Simulation
```

### Lock Behavior

| Authentication Result | Lock Behavior |
|---|---|
| ACCEPT | Unlock – servo moves to 90° |
| REJECT | Lock – servo moves to 0° |

## MQTT Configuration

The current prototype uses:

```text
Broker: broker.hivemq.com
Port: 1883
Topic: aryan/p300eeg/smartlock/control
```

Authentication results are published as MQTT messages:

```text
ACCEPT
```

or

```text
REJECT
```

The ESP32 simulation receives the message and controls the servo accordingly.

> The current MQTT setup uses a public broker and an unsecured topic. It is intended only for demonstration purposes.

---

## Wokwi Smart Lock Simulation

The ESP32 smart-lock simulation is available in Wokwi:

https://wokwi.com/projects/481039949638141953

The project source is also included locally in:

```text
smart_lock/
```

Contents:

```text
smart_lock/
├── sketch.ino
├── diagram.json
├── libraries.txt
└── wokwi-project.txt
```

This allows the circuit and firmware to be examined and reproduced independently of the online simulation.

---

# Explainable AI

Explainability is incorporated into the authentication pipeline to examine what contributes to CNN authentication decisions.

The project uses SHAP-based explainability.

The implementation includes:

- Global channel importance
- Subject-level channel importance
- Temporal importance
- Local authentication-window explanations
- Cross-identity explainability experiments
- Genuine-versus-impostor analysis

Relevant implementation files include:

```text
models/
├── shap_explainability.py
├── local_shap_explainer.py
├── cross_identity_shap.py
└── plot_local_explanation.py
```

Previously generated explainability outputs are available under:

```text
results/shap_explainability/
```

SHAP explanations are intended to help interpret model behavior. They do not establish causality or independently prove biological identity.

---

# Stream Simulation

The Stream Simulator processes sequential EEG windows from the recorded dataset.

It can demonstrate:

- Genuine authentication streams
- Impostor streams
- Consecutive authentication decisions
- ACCEPT/REJECT sequences
- False acceptance behavior

The simulated stream replays recorded dataset windows.

It does not generate new EEG signals or receive data from a physical EEG headset.

---

# Evaluation

The authentication system is evaluated using standard biometric authentication metrics.

| Metric | Description |
|---|---|
| FAR | False Acceptance Rate |
| FRR | False Rejection Rate |
| Accuracy | Overall correctness of authentication decisions |
| Balanced Accuracy | Average of genuine-class and impostor-class performance |
| EER | Equal Error Rate where FAR and FRR intersect |

## Early-Stopping CNN Results

The reported held-out Recording 3 results for the early-stopping CNN authentication pipeline are:

| Metric | Result |
|---|---:|
| Accuracy | 95.97% |
| Balanced Accuracy | 94.42% |
| FAR | 3.64% |
| FRR | 7.52% |
| Mean validation EER | 3.97% |
| Held-out descriptive EER | 4.86% |

### Interpretation

- FAR measures how frequently impostor samples are incorrectly accepted.
- FRR measures how frequently genuine samples are incorrectly rejected.
- Accuracy measures overall classification performance.
- Balanced accuracy accounts for the performance of both classes.
- EER provides a threshold-independent comparison of authentication performance.

The held-out descriptive EER is reported separately from threshold calibration and is not used to select the authentication thresholds.

### Evaluation Caveat

The reported evaluations are window-level results.

EEG windows from the same recordings are correlated and should not be treated as independent users or independent recording sessions. These results do not establish production-grade biometric security or performance against real-world attackers.

---

# Research Artifacts

The repository retains multiple model experiments for reproducibility and comparison.

```text
results/
├── cnn_authentication/
│   └── saved_models/
├── cnn_authentication_experiment/
│   └── saved_models/
├── cnn_authentication_early_stopping/
│   └── saved_models/
└── shap_explainability/
```

The saved CNN model directories contain trained checkpoints for subjects A01–A10.

The early-stopping models are used by the current GUI authentication system.

Other model sets are retained for research comparison and reproducibility.

---

# Testing

IoT simulator tests are located in:

```text
tests/
```

The repository includes:

```text
tests/test_iot_simulator.py
```

---

# Limitations

This project is an academic prototype, and several limitations remain.

- No physical EEG headset is currently connected.
- Authentication uses recorded EEG data.
- The stream simulator replays recorded windows.
- Enrollment does not retrain models or create production-grade biometric templates.
- No production-grade liveness detection is implemented.
- Replay attack protection is not implemented.
- MQTT communication is not secured with authentication or encryption.
- The public MQTT broker and topic are not suitable for real access-control deployment.
- The audit trail is not tamper-proof.
- SHAP provides model attribution, not causal proof.
- Window-level results should not be interpreted as independent-user security guarantees.

The system should not be used to control real locks or protect sensitive physical assets.

---

# Technologies Used

### Programming
- Python

### Machine Learning
- PyTorch
- Scikit-learn
- NumPy
- SciPy
- Pandas

### Explainable AI
- SHAP

### Visualization
- Matplotlib

### Web Application
- FastAPI
- Uvicorn
- HTML
- CSS
- JavaScript

### IoT
- ESP32
- MQTT
- PubSubClient
- Servo
- Wokwi

---

# Troubleshooting

## GUI reports missing recordings

Verify that the preprocessing pipeline has been executed and that the expected files exist under:

```text
data/preprocessed_sessions/
```

## GUI reports missing models

Verify that the early-stopping checkpoints exist under:

```text
results/cnn_authentication_early_stopping/saved_models/
```

## Authentication is unavailable for a subject

Ensure that:
- The subject's preprocessed recordings exist.
- The subject's CNN checkpoint exists.
- The authentication results JSON is available.
- The subject has been enrolled through the GUI.

## SHAP takes a long time

SHAP explanations can be computationally expensive. Local explanations are generated on demand and may take time, especially on CPU.

## MQTT smart lock does not respond

Check:
- Wi-Fi connection in the Wokwi simulation
- MQTT broker connection
- Topic configuration
- ESP32 serial output
- Whether the published message is `ACCEPT` or `REJECT`

---

# Getting Started: Quick Commands

```bash
# Clone
git clone https://github.com/Aryan-coder2327/P300-EEG-Authentication-v2.git

# Enter project
cd P300-EEG-Authentication-v2

# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# After placing raw EEG files in data/raw/
python preprocessing/session_preprocessing_all.py

# Start the GUI
python -m uvicorn gui_app:app --reload
```

Then open:

http://127.0.0.1:8000

---

# Future Work

Potential future improvements include:

- Integration with a physical EEG headset
- Live EEG acquisition
- Improved enrollment and template management
- Robust cross-session authentication
- Replay and spoofing defenses
- MQTT authentication and encryption
- Secure IoT communication
- Hardware-based lock integration
- Improved user-level evaluation
- Security testing against realistic attacks

---

# Dataset Acknowledgment

This project uses publicly available EEG data from the BNCI Horizon 2020 database.

Dataset source:

https://bnci-horizon-2020.eu/database/data-sets

Researchers should consult the dataset's original documentation and citation requirements when using the data.

---

# License

See the repository's `LICENSE` file for licensing information.

---

# Project Status

**Status:** Academic research prototype / active development

The project demonstrates a complete experimental pipeline connecting recorded P300 EEG data, CNN-based authentication, Explainable AI, and an MQTT-based IoT smart-lock simulation.
