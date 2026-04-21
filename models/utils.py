import numpy as np
import json
import os
from pathlib import Path

def load_preprocessed_eeg(data_dir):
    eeg_list = []
    npy_files = sorted(Path(data_dir).glob('*_epochs.npy'))
    for npy_file in npy_files:
        eeg = np.load(npy_file)
        eeg_list.append(eeg)
    eeg_all = np.vstack(eeg_list)
    return eeg_all, {'subjects': list(range(len(eeg_list)))}

def split_by_subject(eeg, subjects, test_subject=9):
    mask_train = subjects != test_subject
    mask_test = subjects == test_subject
    eeg_train = eeg[mask_train]
    eeg_test = eeg[mask_test]
    subjects_train = subjects[mask_train]
    subjects_test = subjects[mask_test]
    return eeg_train, eeg_test, subjects_train, subjects_test
