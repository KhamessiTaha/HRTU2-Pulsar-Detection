"""
Data preprocessing module for HTRU2 pulsar detection.

This module provides comprehensive data preprocessing functionality including
data loading, train/validation/test splitting, feature scaling, and class balancing
using SMOTE for the HTRU2 pulsar dataset.

Author: Taha Khamessi
Date: 2025-06-27
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HTRUPreprocessor:
    """
    Comprehensive preprocessing pipeline for HTRU2 pulsar detection dataset.
    
    This class encapsulates all preprocessing steps including data loading,
    splitting, scaling, and balancing with various configurable options.
    """
    
    def __init__(self, 
                 scaler_type: str = 'standard',
                 balancing_method: str = 'smote',
                 random_state: int = 42):
        """
        Initialize the preprocessor with configuration options.
        
        Args:
            scaler_type: Type of scaler ('standard', 'robust')
            balancing_method: Method for class balancing ('smote', 'borderline', 
                            'adasyn', 'smote_tomek', 'undersample', 'none')
            random_state: Random state for reproducibility
        """
        self.scaler_type = scaler_type
        self.balancing_method = balancing_method
        self.random_state = random_state
        self.scaler = None
        self.feature_names = [
            "Mean_Profile", "Std_Profile", "Excess_kurtosis_Profile", "Skewness_Profile",
            "Mean_DM", "Std_DM", "Excess_kurtosis_DM", "Skewness_DM"
        ]
        
    def load_data(self, path: str = "../data/HTRU_2.csv") -> pd.DataFrame:
        """
        Load HTRU2 dataset with proper column names and validation.
        
        Args:
            path: Path to the CSV file
            
        Returns:
            DataFrame with properly named columns
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data has incorrect shape or missing values
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")
            
        try:
            df = pd.read_csv(path, header=None)
            logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Validate data shape
            if df.shape[1] != 9:
                raise ValueError(f"Expected 9 columns, got {df.shape[1]}")
                
            # Assign column names
            df.columns = self.feature_names + ["Target"]
            
            # Validate target values
            unique_targets = df['Target'].unique()
            if not set(unique_targets).issubset({0, 1}):
                raise ValueError(f"Target should contain only 0 and 1, found: {unique_targets}")
                
            # Check for missing values
            if df.isnull().sum().sum() > 0:
                logger.warning("Missing values detected in dataset")
                
            # Log class distribution
            class_dist = df['Target'].value_counts()
            logger.info(f"Class distribution: {dict(class_dist)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_stratified_splits(self, 
                               df: pd.DataFrame,
                               test_size: float = 0.2,
                               val_size: float = 0.1) -> Tuple[pd.DataFrame, ...]:
        """
        Create stratified train/validation/test splits.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X = df[self.feature_names]
        y = df["Target"]
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y, 
            random_state=self.random_state
        )
        
        # Second split: separate train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, 
                      X_train: pd.DataFrame, 
                      X_val: pd.DataFrame, 
                      X_test: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        """
        Scale features using the specified scaler.
        
        Args:
            X_train, X_val, X_test: Feature matrices
            
        Returns:
            Tuple of scaled arrays and fitted scaler
        """
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Features scaled using {self.scaler_type} scaler")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def balance_data(self, X: np.ndarray, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance the training data using the specified method.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of balanced (X, y)
        """
        if self.balancing_method == 'none':
            return X, y.values
        
        # Initialize balancing method
        if self.balancing_method == 'smote':
            balancer = SMOTE(random_state=self.random_state)
        elif self.balancing_method == 'borderline':
            balancer = BorderlineSMOTE(random_state=self.random_state)
        elif self.balancing_method == 'adasyn':
            balancer = ADASYN(random_state=self.random_state)
        elif self.balancing_method == 'smote_tomek':
            balancer = SMOTETomek(random_state=self.random_state)
        elif self.balancing_method == 'undersample':
            balancer = RandomUnderSampler(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown balancing method: {self.balancing_method}")
        
        try:
            X_balanced, y_balanced = balancer.fit_resample(X, y)
            logger.info(f"Data balanced using {self.balancing_method}")
            logger.info(f"New class distribution: {np.bincount(y_balanced)}")
            return X_balanced, y_balanced
        except Exception as e:
            logger.error(f"Error in data balancing: {str(e)}")
            return X, y.values
    
    def get_cross_validation_splits(self, 
                                  X: np.ndarray, 
                                  y: np.ndarray, 
                                  n_splits: int = 5) -> StratifiedKFold:
        """
        Create stratified cross-validation splits.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of CV folds
            
        Returns:
            StratifiedKFold object
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        logger.info(f"Created {n_splits}-fold stratified CV")
        return cv
    
    def save_preprocessor(self, filepath: str):
        """Save the fitted scaler for later use."""
        if self.scaler is not None:
            joblib.dump(self.scaler, filepath)
            logger.info(f"Scaler saved to {filepath}")
        else:
            logger.warning("No fitted scaler to save")
    
    def load_preprocessor(self, filepath: str):
        """Load a previously fitted scaler."""
        if os.path.exists(filepath):
            self.scaler = joblib.load(filepath)
            logger.info(f"Scaler loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Scaler file not found at {filepath}")
    
    def prepare_data(self, 
                    data_path: str = "../data/HTRU_2.csv",
                    test_size: float = 0.2,
                    val_size: float = 0.1) -> Tuple[np.ndarray, ...]:
        """
        Complete preprocessing pipeline.
        
        Args:
            data_path: Path to the data file
            test_size: Test set proportion
            val_size: Validation set proportion
            
        Returns:
            Tuple of processed data arrays
        """
        logger.info("Starting data preprocessing pipeline...")
        
        # Load and split data
        df = self.load_data(data_path)
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_stratified_splits(
            df, test_size, val_size
        )
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )
        
        # Balance training data
        X_train_balanced, y_train_balanced = self.balance_data(X_train_scaled, y_train)
        
        logger.info("Preprocessing pipeline completed successfully")
        
        return (X_train_balanced, X_val_scaled, X_test_scaled, 
                y_train_balanced, y_val.values, y_test.values)

# Convenience function for backward compatibility
def prepare_data(data_path: str = "../data/HTRU_2.csv",
                test_size: float = 0.2,
                val_size: float = 0.1,
                scaler_type: str = 'standard',
                balancing_method: str = 'smote',
                random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Convenience function for data preparation with default settings.
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    preprocessor = HTRUPreprocessor(
        scaler_type=scaler_type,
        balancing_method=balancing_method,
        random_state=random_state
    )
    
    results = preprocessor.prepare_data(data_path, test_size, val_size)
    return results + (preprocessor.scaler,)

if __name__ == "__main__":
    # Example usage
    preprocessor = HTRUPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data()
    print(f"Data prepared successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")