import numpy as np
import matplotlib.pyplot as plt

class AuthenticationMetrics:
    @staticmethod
    def compute_far_frr(scores_genuine, scores_impostor, threshold):
        far = np.sum(scores_impostor >= threshold) / len(scores_impostor)
        frr = np.sum(scores_genuine < threshold) / len(scores_genuine)
        return far, frr
    
    @staticmethod
    def compute_eer(scores_genuine, scores_impostor, n_thresholds=1000):
        thresholds = np.linspace(0, 1, n_thresholds)
        
        min_diff = float('inf')
        best_threshold = 0.5
        eer = 1.0
        
        for threshold in thresholds:
            far, frr = AuthenticationMetrics.compute_far_frr(
                scores_genuine, scores_impostor, threshold
            )
            
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                best_threshold = threshold
                eer = (far + frr) / 2
        
        return eer, best_threshold
    
    @staticmethod
    def compute_accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def plot_score_distribution(scores_genuine, scores_impostor, threshold=None, title="Score Distribution"):
        plt.figure(figsize=(10, 6))
        plt.hist(scores_genuine, bins=30, alpha=0.6, label='Genuine', color='green')
        plt.hist(scores_impostor, bins=30, alpha=0.6, label='Impostor', color='red')
        if threshold is not None:
            plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_roc_curve(scores_genuine, scores_impostor, title="FAR vs FRR"):
        thresholds = np.linspace(0, 1, 100)
        fars = []
        frrs = []
        
        for threshold in thresholds:
            far, frr = AuthenticationMetrics.compute_far_frr(
                scores_genuine, scores_impostor, threshold
            )
            fars.append(far)
            frrs.append(frr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fars, frrs, linewidth=2.5, color='purple')
        plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
        plt.ylabel('False Rejection Rate (FRR)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
