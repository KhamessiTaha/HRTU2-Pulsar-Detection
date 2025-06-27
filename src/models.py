"""
Machine learning models module for HTRU2 pulsar detection.

This module provides a comprehensive framework for training, evaluating, and
comparing multiple machine learning models for pulsar detection tasks.

Author: Taha Khamessi
Date: 2025-06-27
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score, matthews_corrcoef,
    log_loss, roc_curve
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import joblib
import os
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PulsarModelFramework:
    """
    Comprehensive framework for pulsar detection model training and evaluation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model framework.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
        # Initialize model configurations
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models with their configurations."""
        self.models = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(random_state=self.random_state, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'CatBoost': {
                'model': CatBoostClassifier(random_state=self.random_state, verbose=False),
                'params': {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 7],
                    'l2_leaf_reg': [1, 3, 5]
                }
            },
            'SVM': {
                'model': SVC(kernel='linear', probability=True),
                'params': {'C': [0.1, 1]},
                'cv': 3  # SVM is typically slower, so i'm using fewer folds (really bad pc :<)
            },
            'MLP': {
                'model': MLPClassifier(random_state=self.random_state, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
                    'activation': ['relu', 'tanh', 'logistic'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            },
            'KNeighbors': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'GaussianNB': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                }
            }
        }
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_proba),
                'pr_auc': average_precision_score(y_true, y_proba),
                'mcc': matthews_corrcoef(y_true, y_pred),
                'log_loss': log_loss(y_true, y_proba)
            }
            
            # Calculate specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Calculate balanced accuracy
            metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray, 
                          use_hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            use_hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary containing trained model and results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in available models")
        
        logger.info(f"Training {model_name}...")
        
        model_config = self.models[model_name]
        model = model_config['model']
        
        if use_hyperparameter_tuning and len(model_config['params']) > 0:
            # Perform hyperparameter tuning
            logger.info(f"Performing hyperparameter tuning for {model_name}...")
            
            # Use RandomizedSearchCV for faster tuning
            search = RandomizedSearchCV(
                model, 
                model_config['params'],
                n_iter=50,  # Reduced for faster execution
                cv=5,
                scoring='roc_auc',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            
            logger.info(f"Best parameters for {model_name}: {best_params}")
        else:
            # Train with default parameters
            best_model = model
            best_model.fit(X_train, y_train)
            best_params = {}
        
        # Make predictions
        y_pred = best_model.predict(X_val)
        y_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(y_val, y_pred, y_proba)
        
        # Store results
        result = {
            'model': best_model,
            'best_params': best_params,
            'predictions': y_pred,
            'probabilities': y_proba,
            'metrics': metrics
        }
        
        self.trained_models[model_name] = result
        
        logger.info(f"{model_name} - ROC AUC: {metrics.get('roc_auc', 0):.4f}, "
                   f"F1: {metrics.get('f1_score', 0):.4f}")
        
        return result
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        use_hyperparameter_tuning: bool = True,
                        models_to_train: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Train all available models.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data  
            use_hyperparameter_tuning: Whether to perform hyperparameter tuning
            models_to_train: Specific models to train (if None, train all)
            
        Returns:
            Dictionary of all training results
        """
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        logger.info(f"Training {len(models_to_train)} models...")
        
        results = {}
        for model_name in models_to_train:
            try:
                result = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val, use_hyperparameter_tuning
                )
                results[model_name] = result
                
                # Track best model
                current_score = result['metrics'].get('roc_auc', 0)
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_model = model_name
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.results = results
        logger.info(f"Training completed. Best model: {self.best_model} "
                   f"(ROC AUC: {self.best_score:.4f})")
        
        return results
    
    def create_results_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame with all model results for easy comparison.
        
        Returns:
            DataFrame with model comparison results
        """
        if not self.results:
            logger.warning("No results available. Train models first.")
            return pd.DataFrame()
        
        results_list = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            row = {'Model': model_name}
            row.update(metrics)
            results_list.append(row)
        
        df = pd.DataFrame(results_list)
        df = df.sort_values('roc_auc', ascending=False).reset_index(drop=True)
        
        return df
    
    def cross_validate_models(self, X: np.ndarray, y: np.ndarray,
                            cv: int = 5, scoring: str = 'roc_auc') -> pd.DataFrame:
        """
        Perform cross-validation on all models.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            DataFrame with CV results
        """
        cv_results = []
        
        for model_name, model_config in self.models.items():
            try:
                logger.info(f"Cross-validating {model_name}...")
                model = model_config['model']
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                
                cv_results.append({
                    'Model': model_name,
                    'CV_Mean': scores.mean(),
                    'CV_Std': scores.std(),
                    'CV_Min': scores.min(),
                    'CV_Max': scores.max()
                })
                
            except Exception as e:
                logger.error(f"Error in CV for {model_name}: {str(e)}")
                continue
        
        df = pd.DataFrame(cv_results)
        df = df.sort_values('CV_Mean', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_feature_importance(self, model_name: str) -> Optional[np.ndarray]:
        """
        Get feature importance from a trained model.
        
        Args:
            model_name: Name of the trained model
            
        Returns:
            Feature importance array or None if not available
        """
        if model_name not in self.trained_models:
            logger.warning(f"Model {model_name} not found in trained models")
            return None
        
        model = self.trained_models[model_name]['model']
        
        # Different models have different ways to access feature importance
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            logger.warning(f"Feature importance not available for {model_name}")
            return None
    
    def save_model(self, model_name: str, filepath: str):
        """Save a trained model to disk."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model_data = {
            'model': self.trained_models[model_name]['model'],
            'best_params': self.trained_models[model_name]['best_params'],
            'metrics': self.trained_models[model_name]['metrics'],
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"{model_name} saved to {filepath}")
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """Load a saved model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        
        model_data = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        return model_data
    
    def save_results(self, filepath: str):
        """Save all results to a JSON file."""
        results_to_save = {}
        
        for model_name, result in self.results.items():
            results_to_save[model_name] = {
                'best_params': result['best_params'],
                'metrics': result['metrics']
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")

def train_baseline_models(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         random_state: int = 42) -> Dict[str, Dict]:
    """
    Convenience function to train baseline models quickly.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with baseline model results
    """
    framework = PulsarModelFramework(random_state=random_state)
    
    # Train only fast baseline models
    baseline_models = ['LogisticRegression', 'RandomForest', 'GaussianNB', 'KNeighbors']
    
    return framework.train_all_models(
        X_train, y_train, X_val, y_val,
        use_hyperparameter_tuning=False,
        models_to_train=baseline_models
    )

if __name__ == "__main__":
    # Example usage
    from preprocess import prepare_data
    
    # Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data()
    
    # Initialize framework
    framework = PulsarModelFramework()
    
    # Train all models
    results = framework.train_all_models(X_train, y_train, X_val, y_val)
    
    # Create results DataFrame
    results_df = framework.create_results_dataframe()
    print("\nModel Comparison Results:")
    print(results_df.round(4))
    
    # Save best model
    if framework.best_model:
        framework.save_model(framework.best_model, f"../models/{framework.best_model}_best.pkl")