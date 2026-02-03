import numpy as np
import json
from pathlib import Path

def load_preprocessed_eeg(data_path, subjects=None):
    data_path = Path(data_path)
    
    if subjects is None:
        npy_files = list(data_path.glob('*_epochs.npy'))
        subjects = [f.stem.replace('_epochs', '') for f in npy_files]
        subjects = sorted(subjects)
    
    all_epochs = []
    all_metadata = []
    
    for subject in subjects:
        epochs_file = data_path / f'{subject}_epochs.npy'
        if not epochs_file.exists():
            print(f"Warning: {epochs_file} not found")
            continue
        
        epochs = np.load(epochs_file)
        all_epochs.append(epochs)
        
        metadata_file = data_path / f'{subject}_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                meta = json.load(f)
                all_metadata.append(meta)
    
    eeg = np.vstack(all_epochs)
    
    metadata = {
        'subjects': subjects,
        'n_total_epochs': eeg.shape[0],
        'n_channels': eeg.shape[1],
        'n_samples': eeg.shape[2],
        'sampling_rate': all_metadata[0]['sampling_rate'] if all_metadata else 240,
        'channels': all_metadata[0]['channels'] if all_metadata else ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']
    }
    
    return eeg, metadata

def split_by_subject(eeg, subjects, test_ratio=0.3):
    unique_subjects = np.unique(subjects)
    n_test = max(1, int(len(unique_subjects) * test_ratio))
    test_subject_ids = np.random.choice(unique_subjects, size=n_test, replace=False)
    test_mask = np.isin(subjects, test_subject_ids)
    train_mask = ~test_mask
    return (eeg[train_mask], eeg[test_mask], subjects[train_mask], subjects[test_mask])

def normalize_eeg(eeg):
    eeg_norm = np.zeros_like(eeg, dtype=float)
    for epoch_idx in range(eeg.shape[0]):
        for ch in range(eeg.shape[1]):
            signal = eeg[epoch_idx, ch, :].astype(float)
            mean = signal.mean()
            std = signal.std()
            eeg_norm[epoch_idx, ch, :] = (signal - mean) / (std + 1e-8)
    return eeg_norm

def flatten_eeg(eeg):
    n_epochs = eeg.shape[0]
    return eeg.reshape(n_epochs, -1)
