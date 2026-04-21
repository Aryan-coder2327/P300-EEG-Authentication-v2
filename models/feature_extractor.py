import numpy as np
from scipy.fftpack import fft
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FeatureExtractor:
    def __init__(self, sampling_rate=256):
        self.fs = sampling_rate
    
    def extract_temporal_features(self, eeg):
        features = []
        for trial in eeg:
            trial_features = []
            for ch in trial:
                trial_features.extend([
                    np.mean(ch),
                    np.std(ch),
                    np.max(ch),
                    np.min(ch),
                    np.ptp(ch),
                    np.var(ch)
                ])
            features.append(trial_features)
        return np.array(features)
    
    def extract_spectral_features(self, eeg):
        features = []
        bands = [(0.5, 4), (4, 8), (8, 12), (12, 30)]
        
        for trial in eeg:
            trial_features = []
            for ch in trial:
                f, psd = welch(ch, self.fs, nperseg=min(256, len(ch)))
                for low, high in bands:
                    mask = (f >= low) & (f < high)
                    trial_features.append(np.mean(psd[mask]))
            features.append(trial_features)
        return np.array(features)
    
    def extract_wavelet_features(self, eeg):
        features = []
        scales = [1, 2, 4, 8]
        
        for trial in eeg:
            trial_features = []
            for ch in trial:
                for scale in scales:
                    f, psd = welch(ch, self.fs, nperseg=min(256, len(ch)))
                    fft_vals = np.abs(fft(ch))[:len(ch)//2]
                    trial_features.append(np.mean(fft_vals[::scale]))
            features.append(trial_features)
        return np.array(features)
    
    def extract_all_features(self, eeg, feature_types=['temporal', 'spectral', 'wavelet']):
        features = []
        if 'temporal' in feature_types:
            features.append(self.extract_temporal_features(eeg))
        if 'spectral' in feature_types:
            features.append(self.extract_spectral_features(eeg))
        if 'wavelet' in feature_types:
            features.append(self.extract_wavelet_features(eeg))
        return np.hstack(features)
    
    @staticmethod
    def normalize_features(X_train, X_test):
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)
        return X_train_norm, X_test_norm, scaler
    
    @staticmethod
    def apply_pca(X_train, X_test, n_components=50, verbose=False):
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        if verbose:
            print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        return X_train_pca, X_test_pca, pca
