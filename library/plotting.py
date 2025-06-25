import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_features(D, L, output_path):
    """Plot feature distributions and correlations"""
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    # Feature distribution plot
    plt.figure(figsize=(12, 8))
    for i in range(D.shape[0]):
        plt.subplot(2, 4, i+1)
        plt.hist(D[i, L==0], bins=50, alpha=0.5, label='Class 0')
        plt.hist(D[i, L==1], bins=50, alpha=0.5, label='Class 1')
        plt.title(f'Feature {i+1}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}_distributions.png")
    plt.close()
    
    # Correlation matrix plot
    plt.figure(figsize=(10, 8))
    corr = np.corrcoef(D)
    plt.imshow(corr, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Feature Correlation Matrix')
    plt.savefig(f"{output_path}_correlations.png")
    plt.close()

def plot_minDCF_lr(lambdas, minDCF_dict, output_path, title):
    """Plot minDCF for logistic regression"""
    plt.figure()
    for prior, minDCF in minDCF_dict.items():
        plt.plot(lambdas, minDCF, label=f'π_t = {prior}')
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def bayes_error_plot(p_values, minDCF, actDCF, output_path, title):
    """Plot Bayes error comparison"""
    plt.figure()
    plt.plot(p_values, minDCF, label='minDCF')
    plt.plot(p_values, actDCF, label='actDCF')
    plt.xlabel('Prior log-odds')
    plt.ylabel('DCF')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()