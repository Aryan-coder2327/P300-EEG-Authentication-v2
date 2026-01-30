EEG-Based P300 Authentication with Explainable AI and IoT Integration
**Overview**
This project develops a secure biometric authentication system based on electroencephalogram (EEG) signals, focusing on the P300 event-related potential (ERP). Unlike conventional authentication methods such as passwords, fingerprints, or facial recognition, EEG-based biometrics rely on internally generated brain responses that are difficult to forge or replay.
The system uses publicly available P300 EEG datasets to simulate real-time acquisition, extract P300 features, and authenticate users through machine-learning models. Explainable Artificial Intelligence (XAI) techniques are applied to interpret authentication decisions and visualize which EEG channels and time windows contribute most strongly to each prediction. The final goal is to demonstrate how EEG-based authentication can be integrated with an IoT-enabled smart lock for secure physical access control.
** Project Objectives**
Build a P300-based EEG authentication pipeline using public datasets
Simulate real-time EEG streaming without physical hardware
Design enrollment and verification workflows with biometric templates
Apply Explainable AI for transparency and debugging
Analyze security threats such as replay and impostor attacks
Integrate authentication output with an IoT smart-lock prototype
Evaluate system performance using FAR, FRR, and accuracy
 System Architecture
High-level pipeline:
Public EEG Dataset → Streaming Simulator → Preprocessing → P300 Extraction → Template Database → Authentication Engine → Decision Module → XAI Analysis → IoT Controller → Smart Lock
(Refer to /architecture/system_arch.png for the detailed diagram.)
** Methodology**
Dataset Acquisition
Public P300 EEG datasets such as BNCI Horizon and BCI Competition III are used.
Streaming Simulation
Recorded EEG is replayed chunk-by-chunk to mimic live headset input.
Preprocessing
Band-pass filtering, notch filtering, baseline correction, and epoch segmentation.
Feature Extraction
ERP averaging, temporal features, spectral features, or deep-learning embeddings.
Template Enrollment
User-specific templates are generated and stored in a database.
Authentication
Incoming EEG samples are matched against stored templates using similarity metrics or classifiers.
Explainable AI
SHAP and Grad-CAM are used to explain channel and time-region importance.
IoT Integration
Authentication results are transmitted to an IoT device (ESP32/NodeMCU) that controls a relay-based lock.
** Security Considerations**
Impostor attack simulation using EEG from other subjects
Replay protection through stimulus randomization
Threshold-based acceptance decisions
Template embedding storage instead of raw EEG
FAR and FRR monitoring
**Repository Structure**
/data               # EEG datasets (not included)
/preprocessing      # Filtering & epoching scripts
/models             # Authentication models
/simulation         # Real-time EEG replay
/xai                # Explainability modules
/iot                # IoT communication prototypes
/architecture       # Diagrams
/docs               # Reports and notes
/ppt                # Review presentations
