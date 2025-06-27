"""
Utility functions for HTRU2 pulsar detection project.

This module provides visualization functions, statistical tests, and other
utility functions to support the pulsar detection analysis.

Author: Taha Khamessi
Date: 2025-06-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, roc_auc_score
)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VisualizationUtils:
    """Comprehensive visualization utilities for pulsar detection analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize visualization utilities.
        
        Args:
            figsize: Default figure size
            dpi: DPI for high-quality plots
        """
        self.figsize = figsize
        self.dpi = dpi
        self.feature_names = [
            "Mean_Profile", "Std_Profile", "Excess_kurtosis_Profile", "Skewness_Profile",
            "Mean_DM", "Std_DM", "Excess_kurtosis_DM", "Skewness_DM"
        ]
    
    def plot_data_overview(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create comprehensive data overview plots.
        
        Args:
            df: Input DataFrame
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        fig.suptitle('HTRU2 Dataset Overview', fontsize=16, fontweight='bold')
        
        # Class distribution
        ax1 = axes[0, 0]
        class_counts = df['Target'].value_counts()
        colors = ['lightcoral', 'skyblue']
        wedges, texts, autotexts = ax1.pie(class_counts.values, 
                                          labels=['Non-Pulsar', 'Pulsar'],
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        ax1.set_title('Class Distribution')
        
        # Missing values heatmap
        ax2 = axes[0, 1]
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            sns.heatmap(missing_data.to_frame().T, annot=True, cmap='Reds', ax=ax2)
            ax2.set_title('Missing Values')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                    fontsize=14, transform=ax2.transAxes)
            ax2.set_title('Missing Values Check')
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        # Dataset statistics
        ax3 = axes[1, 0]
        stats_text = f"""
        Dataset Shape: {df.shape}
        Features: {df.shape[1] - 1}
        Samples: {df.shape[0]:,}
        
        Class Balance:
        Non-Pulsar: {class_counts[0]:,} ({class_counts[0]/len(df)*100:.1f}%)
        Pulsar: {class_counts[1]:,} ({class_counts[1]/len(df)*100:.1f}%)
        
        Imbalance Ratio: {class_counts[0]/class_counts[1]:.2f}:1
        """
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Dataset Statistics')
        
        # Feature correlation heatmap
        ax4 = axes[1, 1]
        corr_matrix = df[self.feature_names].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax4, fmt='.2f', cbar_kws={"shrink": .8})
        ax4.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Data overview plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_distributions(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot feature distributions with class separation.
        
        Args:
            df: Input DataFrame
            save_path: Path to save the plot
        """
        n_features = len(self.feature_names)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows), dpi=self.dpi)
        fig.suptitle('Feature Distributions by Class', fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, feature in enumerate(self.feature_names):
            ax = axes[i]
            
            # Plot distributions for each class
            for class_val, label, color in [(0, 'Non-Pulsar', 'lightcoral'), 
                                          (1, 'Pulsar', 'skyblue')]:
                data = df[df['Target'] == class_val][feature]
                ax.hist(data, bins=50, alpha=0.7, label=label, color=color, density=True)
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {feature}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Feature distributions plot saved to {save_path}")
        
        plt.show()
    
    def plot_pca_analysis(self, X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None):
        """
        Create PCA analysis plots.
        
        Args:
            X: Feature matrix
            y: Target vector
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=self.dpi)
        fig.suptitle('Principal Component Analysis', fontsize=16, fontweight='bold')
        
        # Fit PCA
        pca_full = PCA()
        X_pca_full = pca_full.fit_transform(X)
        
        # Explained variance ratio
        ax1 = axes[0]
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        ax1.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', linewidth=2, markersize=8)
        ax1.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Cumulative Explained Variance')
        ax1.set_title('Explained Variance Ratio')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # PCA projection (2D)
        ax2 = axes[1]
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X)
        
        scatter = ax2.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, 
                            cmap='coolwarm', alpha=0.6, s=20)
        ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2f})')
        ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2f})')
        ax2.set_title('2D PCA Projection')
        plt.colorbar(scatter, ax=ax2, label='Class')
        
        # Feature contributions to first two PCs
        ax3 = axes[2]
        components_df = pd.DataFrame(
            pca_2d.components_.T,
            columns=['PC1', 'PC2'],
            index=self.feature_names
        )
        
        components_df.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Feature Contributions to PC1 & PC2')
        ax3.set_ylabel('Component Loading')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"PCA analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create comprehensive model comparison plots.
        
        Args:
            results_df: DataFrame with model results
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # ROC AUC comparison
        ax1 = axes[0, 0]
        results_df_sorted = results_df.sort_values('roc_auc', ascending=True)
        bars1 = ax1.barh(results_df_sorted['Model'], results_df_sorted['roc_auc'], 
                        color='skyblue', alpha=0.8)
        ax1.set_xlabel('ROC AUC Score')
        ax1.set_title('ROC AUC Comparison')
        ax1.set_xlim(0, 1)
        
        # Add value labels on bars
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # F1 Score comparison
        ax2 = axes[0, 1]
        f1_sorted = results_df.sort_values('f1_score', ascending=True)
        bars2 = ax2.barh(f1_sorted['Model'], f1_sorted['f1_score'], 
                        color='lightcoral', alpha=0.8)
        ax2.set_xlabel('F1 Score')
        ax2.set_title('F1 Score Comparison')
        ax2.set_xlim(0, 1)
        
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # Precision vs Recall scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(results_df['recall'], results_df['precision'], 
                            c=results_df['f1_score'], cmap='viridis', 
                            s=100, alpha=0.7)
        
        for i, model in enumerate(results_df['Model']):
            ax3.annotate(model, (results_df.iloc[i]['recall'], results_df.iloc[i]['precision']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision vs Recall')
        plt.colorbar(scatter, ax=ax3, label='F1 Score')
        
        # Multi-metric radar chart
        ax4 = axes[1, 1]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Select top 3 models for radar chart
        top_models = results_df.head(3)
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        for idx, (_, model_data) in enumerate(top_models.iterrows()):
            values = [model_data[metric] for metric in metrics]
            values += [values[0]]  # Complete the circle
            
            ax4.plot(angles, values, 'o-', linewidth=2, 
                    label=model_data['Model'], alpha=0.8)
            ax4.fill(angles, values, alpha=0.25)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics)
        ax4.set_ylim(0, 1)
        ax4.set_title('Top 3 Models - Multi-Metric Comparison')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, models_dict: Dict[str, Dict], 
                              y_true: np.ndarray, save_path: Optional[str] = None):
        """
        Plot confusion matrices for multiple models.
        
        Args:
            models_dict: Dictionary of model results
            y_true: True labels
            save_path: Path to save the plot
        """
        n_models = len(models_dict)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows), dpi=self.dpi)
        fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(models_dict.items()):
            ax = axes[idx]
            
            y_pred = results['predictions']
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Non-Pulsar', 'Pulsar'],
                       yticklabels=['Non-Pulsar', 'Pulsar'])
            
            ax.set_title(f'{model_name}\nAUC: {results["metrics"]["roc_auc"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confusion matrices plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, models_dict: Dict[str, Dict], 
                       y_true: np.ndarray, save_path: Optional[str] = None):
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_dict: Dictionary of model results
            y_true: True labels
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        for model_name, results in models_dict.items():
            y_proba = results['probabilities']
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, alpha=0.8,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ROC curves plot saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, models_dict: Dict[str, Dict], 
                                   y_true: np.ndarray, save_path: Optional[str] = None):
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            models_dict: Dictionary of model results
            y_true: True labels
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        
        for model_name, results in models_dict.items():
            y_proba = results['probabilities']
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, linewidth=2, alpha=0.8,
                    label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.8,
                   label=f'Random (AUC = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"PR curves plot saved to {save_path}")
            
        plt.show()
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                              model_name: str, save_path: Optional[str] = None):
        """
        Plot feature importance for a given model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model
            save_path: Path to save the plot
        """
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning(f"Feature importance not available for {model_name}")
            return
        
        # Create DataFrame for easier plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        bars = plt.barh(importance_df['feature'], importance_df['importance'], 
                       color='skyblue', alpha=0.8)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()

class StatisticalUtils:
    """Statistical analysis utilities."""
    
    @staticmethod
    def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.
        
        Args:
            y: Target vector
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        return dict(zip(classes, weights))
    
    @staticmethod
    def perform_statistical_tests(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """
        Perform statistical tests to compare feature distributions between classes.
        
        Args:
            df: Input DataFrame
            feature_names: List of feature names
            
        Returns:
            DataFrame with test results
        """
        from scipy.stats import mannwhitneyu, ks_2samp
        
        results = []
        
        class_0 = df[df['Target'] == 0]
        class_1 = df[df['Target'] == 1]
        
        for feature in feature_names:
            data_0 = class_0[feature]
            data_1 = class_1[feature]
            
            # Mann-Whitney U test (non-parametric)
            mw_stat, mw_pval = mannwhitneyu(data_0, data_1, alternative='two-sided')
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = ks_2samp(data_0, data_1)
            
            results.append({
                'Feature': feature,
                'MannWhitney_Statistic': mw_stat,
                'MannWhitney_p_value': mw_pval,
                'KS_Statistic': ks_stat,
                'KS_p_value': ks_pval,
                'Significant_MW': mw_pval < 0.05,
                'Significant_KS': ks_pval < 0.05
            })
        
        return pd.DataFrame(results)

def save_experiment_config(config: Dict[str, Any], filepath: str):
    """
    Save experiment configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the configuration
    """
    import json
    from datetime import datetime
    
    config['timestamp'] = datetime.now().isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Experiment configuration saved to {filepath}")

def create_results_summary(results_df: pd.DataFrame) -> str:
    """
    Create a formatted summary of model results.
    
    Args:
        results_df: DataFrame with model results
        
    Returns:
        Formatted string summary
    """
    summary = f"""
=== MODEL PERFORMANCE SUMMARY ===

Best Model: {results_df.iloc[0]['Model']}
Best ROC AUC: {results_df.iloc[0]['roc_auc']:.4f}

Top 3 Models:
"""
    
    for i in range(min(3, len(results_df))):
        model = results_df.iloc[i]
        summary += f"""
{i+1}. {model['Model']}:
   - ROC AUC: {model['roc_auc']:.4f}
   - F1 Score: {model['f1_score']:.4f}
   - Precision: {model['precision']:.4f}
   - Recall: {model['recall']:.4f}
"""
    
    return summary

if __name__ == "__main__":
    # Example usage
    from preprocess import HTRUPreprocessor
    
    # Load data
    preprocessor = HTRUPreprocessor()
    df = preprocessor.load_data()
    
    # Create visualizations
    viz = VisualizationUtils()
    viz.plot_data_overview(df)
    viz.plot_feature_distributions(df)
    
    # Prepare data for PCA
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data()
    viz.plot_pca_analysis(X_train, y_train)