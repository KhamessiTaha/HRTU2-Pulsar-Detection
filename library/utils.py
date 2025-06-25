import numpy as np
import scipy.stats
import scipy.linalg
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Union, Dict, Any
from pathlib import Path
import warnings

# Import or define LogisticRegression here
try:
    from library import LogisticRegression
except ImportError:
    # Placeholder for LogisticRegression if not available
    class LogisticRegression:
        def trainClassifier(self, *args, **kwargs):
            raise NotImplementedError("LogisticRegression class is not implemented or imported.")
        def computeLLR(self, *args, **kwargs):
            raise NotImplementedError("LogisticRegression class is not implemented or imported.")


class MLUtils:
    """Machine Learning utilities for classification tasks."""
    
    @staticmethod
    def vrow(v: np.ndarray) -> np.ndarray:
        """Convert vector to row vector."""
        return v.reshape(1, v.size)
    
    @staticmethod
    def vcol(v: np.ndarray) -> np.ndarray:
        """Convert vector to column vector."""
        return v.reshape(v.size, 1)
    
    @staticmethod
    def empirical_mean(X: np.ndarray) -> np.ndarray:
        """Compute empirical mean of dataset."""
        return MLUtils.vcol(X.mean(axis=1))
    
    @staticmethod
    def empirical_covariance(X: np.ndarray) -> np.ndarray:
        """Compute empirical covariance matrix."""
        mu = MLUtils.empirical_mean(X)
        centered = X - mu
        return np.dot(centered, centered.T) / X.shape[1]


class DataLoader:
    """Data loading and preprocessing utilities."""
    
    @staticmethod
    def load_dataset_shuffle(
        train_file: Union[str, Path], 
        test_file: Union[str, Path], 
        n_features: int,
        standardize: bool = True,
        random_seed: int = 0
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load and preprocess training and test datasets.
        
        Args:
            train_file: Path to training data file
            test_file: Path to test data file
            n_features: Number of features to use
            standardize: Whether to standardize features
            random_seed: Random seed for reproducibility
            
        Returns:
            ((DTR, LTR), (DTE, LTE)): Training and test data/labels
        """
        def _load_file(filename: Union[str, Path]) -> Tuple[List[np.ndarray], List[int]]:
            data_list, label_list = [], []
            
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < n_features + 1:
                        continue
                        
                    features = np.array(parts[:n_features], dtype=np.float32)
                    label = int(parts[-1])
                    
                    data_list.append(MLUtils.vcol(features))
                    label_list.append(label)
            
            return data_list, label_list
        
        # Load training data
        train_data_list, train_label_list = _load_file(train_file)
        DTR = np.hstack(train_data_list)
        LTR = np.array(train_label_list, dtype=np.int32)
        
        # Compute standardization parameters on training data
        if standardize:
            train_mean = MLUtils.empirical_mean(DTR)
            train_std = MLUtils.vcol(np.std(DTR, axis=1))
            train_std = np.where(train_std == 0, 1, train_std)  # Avoid division by zero
            DTR = (DTR - train_mean) / train_std
        
        # Load and standardize test data
        test_data_list, test_label_list = _load_file(test_file)
        DTE = np.hstack(test_data_list)
        LTE = np.array(test_label_list, dtype=np.int32)
        
        if standardize:
            DTE = (DTE - train_mean) / train_std
        
        # Shuffle datasets
        np.random.seed(random_seed)
        train_idx = np.random.permutation(DTR.shape[1])
        test_idx = np.random.permutation(DTE.shape[1])
        
        return (DTR[:, train_idx], LTR[train_idx]), (DTE[:, test_idx], LTE[test_idx])
    
    @staticmethod
    def split_dataset(
        D: np.ndarray, 
        L: np.ndarray, 
        train_ratio: float = 2/3,
        random_seed: int = 0
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Split dataset into training and test sets."""
        n_train = int(D.shape[1] * train_ratio)
        
        np.random.seed(random_seed)
        indices = np.random.permutation(D.shape[1])
        
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        return (D[:, train_idx], L[train_idx]), (D[:, test_idx], L[test_idx])


class FeatureProcessor:
    """Feature processing utilities."""
    
    @staticmethod
    def gaussianization(
        DTR: np.ndarray, 
        DTE: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply Gaussian rank transformation to features."""
        # Compute ranks for training data
        rank_DTR = np.zeros_like(DTR)
        for i in range(DTR.shape[0]):
            for j in range(DTR.shape[1]):
                rank_DTR[i, j] = (DTR[i] < DTR[i, j]).sum()
        
        rank_DTR = (rank_DTR + 1) / (DTR.shape[1] + 2)
        gaussianized_DTR = scipy.stats.norm.ppf(rank_DTR)
        
        if DTE is None:
            return gaussianized_DTR
        
        # Compute ranks for test data using training distribution
        rank_DTE = np.zeros_like(DTE)
        for i in range(DTE.shape[0]):
            for j in range(DTE.shape[1]):
                rank_DTE[i, j] = (DTR[i] <= DTE[i, j]).sum() 
        
        rank_DTE /= (DTR.shape[1] + 2)
        gaussianized_DTE = scipy.stats.norm.ppf(rank_DTE)
        
        return gaussianized_DTR, gaussianized_DTE
    
    @staticmethod
    def PCA(D: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
        """Principal Component Analysis with projection matrix return"""
        mu = MLUtils.empirical_mean(D)  # Fixed: was missing MLUtils prefix
        C = MLUtils.empirical_covariance(D - mu)  # Fixed: added MLUtils prefix
        _, U = np.linalg.eigh(C)
        P = U[:, ::-1][:, 0:m]
        return np.dot(P.T, D), P  # Return transformed data AND projection matrix
        
    @staticmethod
    def LDA(D: np.ndarray, L: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
        """Linear Discriminant Analysis dimensionality reduction."""
        SW = CovarianceUtils.within_class_covariance(D, L)
        SB = CovarianceUtils.between_class_covariance(D, L)
        
        eigenvalues, eigenvectors = scipy.linalg.eigh(SB, SW)
        
        # Sort by eigenvalue magnitude (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        projection_matrix = eigenvectors[:, sorted_indices[:m]]
        
        return np.dot(projection_matrix.T, D), projection_matrix


class CovarianceUtils:
    """Covariance matrix computation utilities."""
    
    @staticmethod
    def within_class_covariance(D: np.ndarray, L: np.ndarray) -> np.ndarray:
        """Compute within-class covariance matrix."""
        SW = np.zeros((D.shape[0], D.shape[0]))
        
        for class_label in np.unique(L):
            class_data = D[:, L == class_label]
            class_cov = MLUtils.empirical_covariance(class_data)
            SW += class_data.shape[1] * class_cov
        
        return SW / D.shape[1]
    
    @staticmethod
    def between_class_covariance(D: np.ndarray, L: np.ndarray) -> np.ndarray:
        """Compute between-class covariance matrix."""
        SB = np.zeros((D.shape[0], D.shape[0]))
        global_mean = MLUtils.empirical_mean(D)
        
        for class_label in np.unique(L):
            class_data = D[:, L == class_label]
            class_mean = MLUtils.empirical_mean(class_data)
            mean_diff = class_mean - global_mean
            
            SB += class_data.shape[1] * np.dot(mean_diff, mean_diff.T)
        
        return SB / D.shape[1]


class ClassificationMetrics:
    """Classification evaluation metrics."""
    
    @staticmethod
    def assign_labels(
        scores: np.ndarray, 
        pi: float, 
        Cfn: float, 
        Cfp: float, 
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """Assign binary labels based on scores and costs."""
        if threshold is None:
            threshold = -np.log(pi * Cfn) + np.log((1 - pi) * Cfp)
        
        return (scores > threshold).astype(np.int32)
    
    @staticmethod
    def confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute 2x2 confusion matrix."""
        conf_matrix = np.zeros((2, 2), dtype=int)
        
        conf_matrix[0, 0] = np.sum((predictions == 0) & (labels == 0))  # TN
        conf_matrix[0, 1] = np.sum((predictions == 0) & (labels == 1))  # FN
        conf_matrix[1, 0] = np.sum((predictions == 1) & (labels == 0))  # FP
        conf_matrix[1, 1] = np.sum((predictions == 1) & (labels == 1))  # TP
        
        return conf_matrix
    
    @staticmethod
    def DCF_unnormalized(conf_matrix: np.ndarray, pi: float, Cfn: float, Cfp: float) -> float:
        """Compute unnormalized Detection Cost Function."""
        FNR = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1]) if (conf_matrix[0, 1] + conf_matrix[1, 1]) > 0 else 0
        FPR = conf_matrix[1, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0]) if (conf_matrix[0, 0] + conf_matrix[1, 0]) > 0 else 0
        
        return pi * Cfn * FNR + (1 - pi) * Cfp * FPR
    
    @staticmethod
    def DCF_normalized(conf_matrix: np.ndarray, pi: float, Cfn: float, Cfp: float) -> float:
        """Compute normalized Detection Cost Function."""
        dcf_unnorm = ClassificationMetrics.DCF_unnormalized(conf_matrix, pi, Cfn, Cfp)
        dcf_dummy = min(pi * Cfn, (1 - pi) * Cfp)
        
        return dcf_unnorm / dcf_dummy if dcf_dummy > 0 else float('inf')
    
    @staticmethod
    def min_DCF(scores: np.ndarray, labels: np.ndarray, pi: float, Cfn: float, Cfp: float) -> float:
        """Compute minimum Detection Cost Function over all thresholds."""
        thresholds = np.sort(scores)
        thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])
        
        dcf_values = []
        for threshold in thresholds:
            predictions = ClassificationMetrics.assign_labels(scores, pi, Cfn, Cfp, threshold)
            conf_matrix = ClassificationMetrics.confusion_matrix(predictions, labels)
            dcf = ClassificationMetrics.DCF_normalized(conf_matrix, pi, Cfn, Cfp)
            dcf_values.append(dcf)
        
        return np.min(dcf_values)
    
    @staticmethod
    def actual_DCF(
        scores: np.ndarray, 
        labels: np.ndarray, 
        pi: float, 
        Cfn: float, 
        Cfp: float, 
        threshold: Optional[float] = None
    ) -> float:
        """Compute actual Detection Cost Function for given threshold."""
        predictions = ClassificationMetrics.assign_labels(scores, pi, Cfn, Cfp, threshold)
        conf_matrix = ClassificationMetrics.confusion_matrix(predictions, labels)
        return ClassificationMetrics.DCF_normalized(conf_matrix, pi, Cfn, Cfp)
    
    @staticmethod
    def compute_ROC_DET_curves(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute ROC and DET curve points."""
        thresholds = np.sort(scores)
        thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])
        
        FPR_list, FNR_list = [], []
        
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            conf_matrix = ClassificationMetrics.confusion_matrix(predictions, labels)
            
            # Handle edge cases
            if conf_matrix[0, 0] + conf_matrix[1, 0] > 0:
                FPR = conf_matrix[1, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
            else:
                FPR = 0.0
                
            if conf_matrix[0, 1] + conf_matrix[1, 1] > 0:
                FNR = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1])
            else:
                FNR = 0.0
            
            FPR_list.append(FPR)
            FNR_list.append(FNR)
        
        FPR = np.array(FPR_list)
        FNR = np.array(FNR_list)
        TPR = 1 - FNR
        TNR = 1 - FPR
        
        return FPR, FNR, TNR, TPR


class ModelEvaluation:
    """Model evaluation utilities."""
    
    @staticmethod
    def compute_calibrated_scores(scores, L, prior):
        """Calibrate scores using logistic regression"""
        # Create calibrator (use simple LR without quadratic terms)
        calibrator = LogisticRegression()
        # Train on scores and labels
        calibrator.trainClassifier(MLUtils.vrow(scores), L, l=1e-4, pi=prior, type='linear')
        # Return calibrated log-likelihood ratios
        return calibrator.computeLLR(MLUtils.vrow(scores))
    
    @staticmethod
    def k_fold_validation(
        D: np.ndarray,
        L: np.ndarray,
        pi: float,
        model: Any,  # This is now a model INSTANCE, not a class
        model_args: Tuple,
        k_folds: int = 5,
        calibrated: bool = False,
        Cfn: float = 1.0,
        Cfp: float = 1.0,
        random_seed: int = 0
    ) -> Tuple[float, float]:
        """Perform k-fold cross-validation."""
        np.random.seed(random_seed)
        indices = np.random.permutation(D.shape[1])
        
        fold_size = D.shape[1] // k_folds
        all_scores = []
        all_labels = []
        
        for fold in range(k_folds):
            # Define test fold indices
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < k_folds - 1 else D.shape[1]
            test_indices = indices[start_idx:end_idx]
            
            # Define training fold indices
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            # Split data
            DTR_fold = D[:, train_indices]
            LTR_fold = L[train_indices]
            DTE_fold = D[:, test_indices]
            LTE_fold = L[test_indices]
            
            # Train model and compute scores - USE THE MODEL INSTANCE DIRECTLY
            trained_model = model.trainClassifier(DTR_fold, LTR_fold, *model_args)
            
            if calibrated:
                # Use the new calibration function
                scores_fold = ModelEvaluation.compute_calibrated_scores(
                    trained_model.computeLLR(DTE_fold), 
                    LTR_fold,
                    pi,
                    model.__class__  # You may need to pass the calibrator class here
                )
            else:
                scores_fold = trained_model.computeLLR(DTE_fold)
            
            all_scores.extend(scores_fold)
            all_labels.extend(LTE_fold)
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        min_dcf = ClassificationMetrics.min_DCF(all_scores, all_labels, pi, Cfn, Cfp)
        act_dcf = ClassificationMetrics.actual_DCF(all_scores, all_labels, pi, Cfn, Cfp)
        
        return min_dcf, act_dcf

class Visualizer:
    """Plotting and visualization utilities."""
    
    def __init__(self, output_dir: Union[str, Path] = ''):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_feature_distributions(
        self, 
        DTR: np.ndarray, 
        LTR: np.ndarray, 
        name: str,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """Plot feature distributions for each class."""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(DTR.shape[0])]
        
        D0 = DTR[:, LTR == 0]
        D1 = DTR[:, LTR == 1]
        
        for i in range(DTR.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(D0[i, :], bins=50, density=True, alpha=0.7, 
                   color='orange', label='Class 0', edgecolor='darkorange')
            ax.hist(D1[i, :], bins=50, density=True, alpha=0.7, 
                   color='cornflowerblue', label='Class 1', edgecolor='royalblue')
            
            ax.set_title(feature_names[i])
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{name}_feature_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_correlation_heatmap(self, DTR: np.ndarray, LTR: np.ndarray) -> None:
        """Plot correlation heatmaps for different class splits."""
        datasets = {
            'whole_dataset': DTR,
            'class_0': DTR[:, LTR == 0],
            'class_1': DTR[:, LTR == 1]
        }
        
        for name, data in datasets.items():
            corr_matrix = np.abs(np.corrcoef(data))
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr_matrix, cmap='viridis', vmin=0, vmax=1)
            
            ax.set_title(f'Correlation Matrix - {name.replace("_", " ").title()}')
            plt.colorbar(im)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'correlation_{name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_ROC_curve(
        self, 
        results: List[Tuple[np.ndarray, str, str]], 
        labels: np.ndarray,
        filename: str,
        title: str = "ROC Curve"
    ) -> None:
        """Plot ROC curves for multiple models."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for scores, label, color in results:
            FPR, _, _, TPR = ClassificationMetrics.compute_ROC_DET_curves(scores, labels)
            ax.plot(FPR, TPR, label=label, color=color, linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'roc_{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_DET_curve(
        self, 
        results: List[Tuple[np.ndarray, str, str]], 
        labels: np.ndarray,
        filename: str,
        title: str = "DET Curve"
    ) -> None:
        """Plot DET curves for multiple models."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for scores, label, color in results:
            FPR, FNR, _, _ = ClassificationMetrics.compute_ROC_DET_curves(scores, labels)
            ax.plot(FPR, FNR, label=label, color=color, linewidth=2)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('False Negative Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'det_{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()


# Backward compatibility aliases
vrow = MLUtils.vrow
vcol = MLUtils.vcol
empirical_mean = MLUtils.empirical_mean
empirical_covariance = MLUtils.empirical_covariance
load_dataset_shuffle = DataLoader.load_dataset_shuffle
split_db_2to1 = DataLoader.split_dataset
features_gaussianization = FeatureProcessor.gaussianization
PCA = FeatureProcessor.PCA
LDA = FeatureProcessor.LDA
empirical_withinclass_cov = CovarianceUtils.within_class_covariance
empirical_betweenclass_cov = CovarianceUtils.between_class_covariance
assign_labels = ClassificationMetrics.assign_labels
conf_matrix = ClassificationMetrics.confusion_matrix
DCFu = ClassificationMetrics.DCF_unnormalized
DCF = ClassificationMetrics.DCF_normalized
minDCF = ClassificationMetrics.min_DCF
actDCF = ClassificationMetrics.actual_DCF
compute_rates_values = ClassificationMetrics.compute_ROC_DET_curves
kfolds = ModelEvaluation.k_fold_validation