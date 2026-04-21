import pandas as pd
import matplotlib.pyplot as plt

class PaperComparison:
    BENCHMARK_PAPERS = {
        'Arias-Cabarcos 2021': {'eer': 0.08, 'method': 'ML on low-cost EEG'},
        'Bernal et al. 2023': {'eer': 0.00, 'method': 'RF/SVM/CNN'},
        'Zhao et al. 2025': {'eer': 0.12, 'method': 'ERP comparison'},
        'Tao et al. 2022': {'eer': 0.10, 'method': '3D-CNN'},
        'Wang et al. 2022': {'eer': 0.15, 'method': 'Privacy-preserving'}
    }
    
    @staticmethod
    def print_comparison_table(your_results, verbose=True):
        data = []
        for paper, info in PaperComparison.BENCHMARK_PAPERS.items():
            data.append({'Paper': paper, 'EER': f"{info['eer']*100:.1f}%", 'Method': info['method']})
        
        for method, results in your_results.items():
            data.append({'Paper': f'YOUR {method.upper()}', 'EER': f"{results['eer']*100:.1f}%", 'Method': 'ML on healthy subjects'})
        
        df = pd.DataFrame(data)
        if verbose:
            print(df.to_string(index=False))
        return df
    
    @staticmethod
    def plot_comparison(your_results, save_path=None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        papers = list(PaperComparison.BENCHMARK_PAPERS.keys())
        eers = [PaperComparison.BENCHMARK_PAPERS[p]['eer']*100 for p in papers]
        
        your_eers = [your_results[m]['eer']*100 for m in ['lda', 'svm', 'rf']]
        your_methods = ['YOUR LDA', 'YOUR SVM', 'YOUR RF']
        
        all_papers = papers + your_methods
        all_eers = eers + your_eers
        
        ax.bar(range(len(all_papers)), all_eers, color=['blue']*len(papers) + ['green']*len(your_methods))
        ax.set_xticks(range(len(all_papers)))
        ax.set_xticklabels(all_papers, rotation=45, ha='right')
        ax.set_ylabel('EER (%)')
        ax.set_title('P300 Authentication EER Comparison')
        ax.grid(axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
