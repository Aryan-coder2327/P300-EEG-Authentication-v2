import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

class AuthenticationMetrics:
    @staticmethod
    def compute_eer(genuine_scores, impostor_scores, n_thresholds=1000):
        thresholds = np.linspace(0, 1, n_thresholds)
        min_diff = float('inf')
        best_threshold = 0
        best_eer = 0
        
        for threshold in thresholds:
            fa = np.sum(impostor_scores >= threshold) / len(impostor_scores)
            fr = np.sum(genuine_scores < threshold) / len(genuine_scores)
            
            if abs(fa - fr) < min_diff:
                min_diff = abs(fa - fr)
                best_eer = (fa + fr) / 2
                best_threshold = threshold
        
        return best_eer, best_threshold
    
    @staticmethod
    def compute_far_frr(genuine_scores, impostor_scores, threshold):
        fa = np.sum(impostor_scores >= threshold) / len(impostor_scores)
        fr = np.sum(genuine_scores < threshold) / len(genuine_scores)
        return fa, fr
