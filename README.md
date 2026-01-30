# EEG-Based P300 Authentication with Explainable AI and IoT Integration

## Overview

This project develops a secure biometric authentication system based on electroencephalogram (EEG) signals, focusing on the P300 event-related potential (ERP). Unlike conventional authentication methods such as passwords, fingerprints, or facial recognition, EEG-based biometrics rely on internally generated brain responses that are difficult to forge or replay.

The system uses publicly available P300 EEG datasets to simulate real-time acquisition, extract P300 features, and authenticate users through machine-learning models. Explainable Artificial Intelligence (XAI) techniques are applied to interpret authentication decisions and visualize which EEG channels and time windows contribute most strongly to each prediction. The final goal is to demonstrate how EEG-based authentication can be integrated with an IoT-enabled smart lock for secure physical access control.

## Project Objectives

- Build a P300-based EEG authentication pipeline using public datasets
- Simulate real-time EEG streaming without physical hardware
- Design enrollment and verification workflows with biometric templates
- Apply Explainable AI for transparency and debugging
- Analyze security threats such as replay and impostor attacks
- Integrate authentication output with an IoT smart-lock prototype
- Evaluate system performance using FAR, FRR, and accuracy

## System Architecture

**High-level pipeline:**

```
Public EEG Dataset → Streaming Simulator → Preprocessing → P300 Extraction → 
Template Database → Authentication Engine → Decision Module → XAI Analysis → 
IoT Controller → Smart Lock
```

Refer to `/architecture/system_arch.png` for the detailed diagram.

## Methodology

### Dataset Acquisition

Public P300 EEG datasets such as BNCI Horizon and BCI Competition III are used.

### Data Handling
Raw EEG datasets are not stored in this repository.
Instructions to download and place the data locally
are provided in `data/README.md`.


### Streaming Simulation

Recorded EEG is replayed chunk-by-chunk to mimic live headset input.

### Preprocessing

Band-pass filtering, notch filtering, baseline correction, and epoch segmentation are applied to raw EEG signals.

### Feature Extraction

ERP averaging, temporal features, spectral features, or deep-learning embeddings are computed from preprocessed signals.

### Template Enrollment

User-specific templates are generated and stored in a database for later comparison.

### Authentication

Incoming EEG samples are matched against stored templates using similarity metrics or classifiers.

### Explainable AI

SHAP and Grad-CAM are used to explain channel and time-region importance in authentication decisions.

### IoT Integration

Authentication results are transmitted to an IoT device (ESP32/NodeMCU) that controls a relay-based lock.

## Security Considerations

- **Impostor attack simulation:** EEG from other subjects is used to test authentication robustness
- **Replay protection:** Stimulus randomization prevents replay attacks
- **Threshold-based acceptance:** Configurable decision thresholds balance security and usability
- **Template embedding storage:** Encrypted embeddings instead of raw EEG data
- **Performance monitoring:** FAR and FRR metrics are continuously tracked

## Repository Structure

```
.
├── data/                    # EEG datasets (not included)
├── preprocessing/           # Filtering & epoching scripts
├── models/                  # Authentication models
├── simulation/              # Real-time EEG replay
├── xai/                     # Explainability modules
├── iot/                     # IoT communication prototypes
├── architecture/            # Diagrams and architecture documentation
├── docs/                    # Reports and notes
├── ppt/                     # Review presentations
└── README.md                # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download public datasets:**
   ```bash
   python scripts/download_datasets.py
   ```

3. **Run preprocessing pipeline:**
   ```bash
   python preprocessing/preprocess_eeg.py
   ```

4. **Train authentication model:**
   ```bash
   python models/train_authenticator.py
   ```

5. **Run simulation and authentication:**
   ```bash
   python simulation/run_authentication_demo.py
   ```

6. **Visualize explainability results:**
   ```bash
   python xai/generate_explanations.py
   ```

## Key Features

- **Non-invasive biometric authentication:** Leverages natural brain responses rather than external biometrics
- **Real-time processing:** Streaming simulation enables demonstration without specialized hardware
- **Transparency:** XAI techniques provide interpretable insights into authentication decisions
- **IoT integration:** Direct connection to smart locks for practical access control
- **Comprehensive evaluation:** FAR, FRR, and accuracy metrics for security assessment

## Performance Metrics

The system is evaluated using standard biometric authentication metrics:

- **False Acceptance Rate (FAR):** Percentage of unauthorized attempts incorrectly accepted
- **False Rejection Rate (FRR):** Percentage of authorized attempts incorrectly rejected
- **Equal Error Rate (EER):** Threshold where FAR equals FRR
- **Accuracy:** Overall correctness of authentication decisions

## Security Analysis

This project includes simulations of common attacks:

- **Impostor attacks:** Unauthorized users attempting to authenticate with their own EEG
- **Replay attacks:** Attackers using previously captured EEG signals
- **Presentation attacks:** Mimicking or spoofing EEG patterns

Countermeasures such as stimulus randomization and template variation are implemented and evaluated.

## IoT Integration

Authentication decisions are transmitted to IoT devices via MQTT or HTTP protocols. The system controls relay-based locks through:

- ESP32/NodeMCU microcontrollers
- Wi-Fi or Bluetooth connectivity
- Real-time decision feedback

## Dataset Sources

Public datasets used in this project:

- **BNCI Horizon:** Multiple P300-based BCI datasets
- **BCI Competition III:** Standard benchmark dataset for BCI research

All datasets are used in accordance with their respective licenses and citation requirements.

## Technologies Used

- **Signal Processing:** NumPy, SciPy, MNE-Python
- **Machine Learning:** Scikit-learn, TensorFlow/PyTorch
- **Explainability:** SHAP, Grad-CAM, Captum
- **Visualization:** Matplotlib, Plotly, Seaborn
- **IoT:** MQTT, Arduino IDE, MicroPython
- **Data Processing:** Pandas, Jupyter Notebooks

## Contributing

Contributions are welcome! Please follow standard GitHub workflow:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Create a Pull Request

## Future Work

- Integration with commercial EEG headsets (OpenBCI, Emotiv)
- Mobile application for real-time authentication
- Multi-modal biometric fusion (EEG + face/fingerprint)
- Deep learning models for improved classification
- Security audit and penetration testing
- User experience optimization and enrollment time reduction

## Citation

If you use this project in your research, please cite:

```
@misc{eeg_p300_auth,
  title={EEG-Based P300 Authentication with Explainable AI and IoT Integration},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/repo}},
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- BNCI Horizon consortium for public EEG datasets
- BCI Competition organizers for standardized benchmarks
- Research community for foundational work on P300 BCI systems
- SHAP and Grad-CAM developers for explainability tools

## Contact

For questions or inquiries about this project, please contact:

- **Email:** your.email@example.com
- **Issues:** Please use the GitHub Issues page for bug reports and feature requests

---

**Last Updated:** January 2026

**Status:** Active Development
