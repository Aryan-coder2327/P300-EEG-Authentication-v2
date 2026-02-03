import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TemplateAuthenticator:
    def __init__(self, similarity_metric='cosine', threshold=0.70):
        if similarity_metric not in ['cosine', 'euclidean', 'correlation']:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        
        self.templates = {}
        self.similarity_metric = similarity_metric
        self.threshold = threshold
    
    def enroll(self, eeg_trials, subject_id):
        template = eeg_trials.mean(axis=0)
        template = self._normalize_signal(template)
        self.templates[subject_id] = template
        print(f"✓ Enrolled subject {subject_id}")
        print(f"  - Input trials: {eeg_trials.shape[0]}")
        print(f"  - Template shape: {template.shape}")
        return template
    
    def authenticate(self, eeg_trial, subject_id, return_score=False):
        if subject_id not in self.templates:
            raise ValueError(f"Subject {subject_id} not enrolled.")
        
        template = self.templates[subject_id]
        trial_norm = self._normalize_signal(eeg_trial)
        score = self._similarity(trial_norm, template)
        decision = score >= self.threshold
        
        if return_score:
            return decision, score
        return decision
    
    def authenticate_batch(self, eeg_trials, subject_ids, return_scores=False):
        decisions = []
        scores = []
        
        for trial, subject_id in zip(eeg_trials, subject_ids):
            decision, score = self.authenticate(trial, subject_id, return_score=True)
            decisions.append(decision)
            scores.append(score)
        
        decisions = np.array(decisions)
        scores = np.array(scores)
        
        if return_scores:
            return scores
        return decisions
    
    def _similarity(self, trial, template):
        if self.similarity_metric == 'cosine':
            trial_flat = trial.flatten().reshape(1, -1)
            template_flat = template.flatten().reshape(1, -1)
            cos_sim = cosine_similarity(trial_flat, template_flat)[0, 0]
            return (cos_sim + 1) / 2
        
        elif self.similarity_metric == 'euclidean':
            trial_flat = trial.flatten()
            template_flat = template.flatten()
            dist = np.linalg.norm(trial_flat - template_flat)
            return 1 / (1 + dist)
        
        elif self.similarity_metric == 'correlation':
            trial_flat = trial.flatten()
            template_flat = template.flatten()
            corr = np.corrcoef(trial_flat, template_flat)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            return (corr + 1) / 2
    
    @staticmethod
    def _normalize_signal(signal):
        signal_norm = np.zeros_like(signal, dtype=float)
        
        if signal.ndim == 1:
            mean = signal.mean()
            std = signal.std()
            signal_norm = (signal - mean) / (std + 1e-8)
        
        elif signal.ndim == 2:
            for ch in range(signal.shape[0]):
                mean = signal[ch, :].mean()
                std = signal[ch, :].std()
                signal_norm[ch, :] = (signal[ch, :] - mean) / (std + 1e-8)
        
        return signal_norm
