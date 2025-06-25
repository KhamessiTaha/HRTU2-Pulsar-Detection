import numpy as np
import matplotlib.pyplot as plt
import importlib
import sys
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from library import utils, GaussianClassifier as GC, LogisticRegression as LR, SVM, GMM
import library.plotting as plotting


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    name: str
    model_class: Any
    params: Dict[str, Any]
    description: str


@dataclass
class ExperimentConfig:
    """Configuration class for experiments"""
    priors: List[float]
    pca_components: List[int]
    lambda_range: np.ndarray
    c_range: np.ndarray
    gmm_components: List[int]


class MLPipeline:
    """Main machine learning pipeline class"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.models = self._initialize_models()
        
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize model configurations"""
        return {
            'gaussian_full': ModelConfig(
                'Gaussian Full-Cov',
                GC.GaussianClassifier,
                {'cov_type': 'full', 'tied': False},
                'Full covariance Gaussian classifier'
            ),
            'gaussian_diag': ModelConfig(
                'Gaussian Diag-Cov',
                GC.GaussianClassifier,
                {'cov_type': 'diag', 'tied': False},
                'Diagonal covariance Gaussian classifier'
            ),
            'gaussian_tied_full': ModelConfig(
                'Gaussian Tied Full-Cov',
                GC.GaussianClassifier,
                {'cov_type': 'full', 'tied': True},
                'Tied full covariance Gaussian classifier'
            ),
            'gaussian_tied_diag': ModelConfig(
                'Gaussian Tied Diag-Cov',
                GC.GaussianClassifier,
                {'cov_type': 'diag', 'tied': True},
                'Tied diagonal covariance Gaussian classifier'
            ),
            'logistic_regression': ModelConfig(
                'Logistic Regression',
                LR.LogisticRegression,
                {'lambda': 1e-5, 'pi_t': 0.5},
                'Logistic regression classifier'
            ),
            'linear_svm': ModelConfig(
                'Linear SVM',
                SVM.SVM,
                {'kernel': 'linear', 'C': 1e-2, 'balanced': False},
                'Linear Support Vector Machine'
            ),
            'rbf_svm': ModelConfig(
                'RBF SVM',
                SVM.SVM,
                {'kernel': 'RBF', 'C': 1e-1, 'gamma': 1e-3},
                'RBF kernel Support Vector Machine'
            ),
            'poly_svm': ModelConfig(
                'Polynomial SVM',
                SVM.SVM,
                {'kernel': 'poly', 'C': 1e-3, 'degree': 2, 'coef0': 1},
                'Polynomial kernel Support Vector Machine'
            ),
            'gmm_full': ModelConfig(
                'GMM Full-Cov',
                GMM.GMM,
                {'n_components': 8, 'cov_type': 'full'},
                'Gaussian Mixture Model with full covariance'
            ),
            'gmm_diag': ModelConfig(
                'GMM Diag-Cov',
                GMM.GMM,
                {'n_components': 16, 'cov_type': 'diag'},
                'Gaussian Mixture Model with diagonal covariance'
            ),
            'gmm_tied': ModelConfig(
                'GMM Tied-Cov',
                GMM.GMM,
                {'n_components': 32, 'cov_type': 'tied'},
                'Gaussian Mixture Model with tied covariance'
            )
        }
    
    def load_data(self, train_path: str = 'Train.txt', test_path: str = 'Test.txt', 
                  n_features: int = 8) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load and return training and test data"""
        logger.info('Loading data...')
        try:
            (DTR, LTR), (DTE, LTE) = utils.load_dataset_shuffle(train_path, test_path, n_features)
            logger.info(f'Data loaded successfully. Train: {DTR.shape}, Test: {DTE.shape}')
            return (DTR, LTR), (DTE, LTE)
        except Exception as e:
            logger.error(f'Error loading data: {e}')
            raise
    
    def plot_features(self, DTR: np.ndarray, LTR: np.ndarray, output_dir: str = 'plots') -> None:
        """Generate feature plots"""
        logger.info('Plotting features...')
        try:
            Path(output_dir).mkdir(exist_ok=True)
            plotting.plot_features(DTR, LTR, f'{output_dir}/plot_features')
            logger.info('Feature plots generated successfully')
        except Exception as e:
            logger.error(f'Error plotting features: {e}')
    
    def apply_pca(self, DTR: np.ndarray, DTE: np.ndarray = None, 
                n_components: int = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply PCA transformation with proper projection matrix handling"""
        if n_components is None or n_components >= DTR.shape[0]:
            return DTR if DTE is None else (DTR, DTE)
        
        # Apply PCA and get projection matrix
        DTR_pca, P = utils.PCA(DTR, n_components)
        
        if DTE is not None:
            DTE_pca = np.dot(P.T, DTE)  # Use same projection matrix for test data
            return DTR_pca, DTE_pca
        
        return DTR_pca
    
    def evaluate_model(self, DTR: np.ndarray, LTR: np.ndarray, model_name: str,
                    validation_method: str = 'kfold', n_folds: int = 5) -> Dict[str, float]:
        """Evaluate a single model with given parameters"""
        model_config = self.models[model_name]
        # CREATE MODEL INSTANCE HERE
        model = model_config.model_class()
        results = {}
        
        for prior in self.config.priors:
            if validation_method == 'kfold':
                # PASS MODEL INSTANCE DIRECTLY
                min_dcf, act_dcf = utils.kfolds(
                    DTR, LTR, prior, model, 
                    tuple(model_config.params.values()), 
                    n_folds
                )
            else:  # single split
                min_dcf, act_dcf = utils.single_split(
                    DTR, LTR, prior, model, 
                    model_config.params
                )
            
            results[f'prior_{prior}_minDCF'] = min_dcf
            results[f'prior_{prior}_actDCF'] = act_dcf
        
        return results
    
    def hyperparameter_search(self, DTR: np.ndarray, LTR: np.ndarray, 
                             model_name: str, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Perform hyperparameter search for a model"""
        logger.info(f'Starting hyperparameter search for {model_name}')
        
        best_params = {}
        best_score = float('inf')
        results = []
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            try:
                # Update model config with current parameters
                temp_config = self.models[model_name]
                temp_config.params.update(params)
                
                # Evaluate model
                model_results = self.evaluate_model(DTR, LTR, model_name)
                
                # Use primary prior (0.5) for model selection
                current_score = model_results.get('prior_0.5_minDCF', float('inf'))
                
                if current_score < best_score:
                    best_score = current_score
                    best_params = params.copy()
                
                results.append({
                    'params': params,
                    'score': current_score,
                    'all_results': model_results
                })
                
            except Exception as e:
                logger.warning(f'Error evaluating params {params}: {e}')
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all combinations of parameters"""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = []
        
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def comprehensive_evaluation(self, DTR: np.ndarray, LTR: np.ndarray) -> Dict[str, Any]:
        """Run comprehensive evaluation across all models and PCA settings"""
        logger.info('Starting comprehensive evaluation...')
        
        results = {}
        
        for pca_components in self.config.pca_components:
            pca_key = f'pca_{pca_components}' if pca_components < DTR.shape[0] else 'raw'
            results[pca_key] = {}
            
            # Apply PCA if needed
            DTR_transformed = self.apply_pca(DTR, n_components=pca_components)
            
            # Evaluate each model
            for model_name in self.models.keys():
                try:
                    model_results = self.evaluate_model(DTR_transformed, LTR, model_name)
                    results[pca_key][model_name] = model_results
                    
                    logger.info(f'Completed {model_name} with {pca_key}')
                    
                except Exception as e:
                    logger.error(f'Error evaluating {model_name} with {pca_key}: {e}')
                    results[pca_key][model_name] = {'error': str(e)}
        
        return results
    
    def plot_validation_curves(self, DTR: np.ndarray, LTR: np.ndarray, 
                              output_dir: str = 'plots') -> None:
        """Generate validation curves for hyperparameter tuning"""
        logger.info('Generating validation curves...')
        Path(output_dir).mkdir(exist_ok=True)
        
        # Logistic Regression lambda curves
        self._plot_lr_validation_curves(DTR, LTR, output_dir)
        
        # SVM C curves
        self._plot_svm_validation_curves(DTR, LTR, output_dir)
        
        # GMM component curves
        self._plot_gmm_validation_curves(DTR, LTR, output_dir)
    
    def _plot_lr_validation_curves(self, DTR: np.ndarray, LTR: np.ndarray, 
                                   output_dir: str) -> None:
        """Plot logistic regression validation curves"""
        lambda_values = np.logspace(-5, 1, 20)
        
        for pca_comp in [None, 7, 6]:
            DTR_pca = self.apply_pca(DTR, n_components=pca_comp)
            
            prior_results = {prior: [] for prior in self.config.priors}
            
            for lambda_val in lambda_values:
                model = LR.LogisticRegression()
                for prior in self.config.priors:
                    min_dcf, _ = utils.kfolds(DTR_pca, LTR, prior, model, (lambda_val, 0.5))
                    prior_results[prior].append(min_dcf)
            
            # Plot results
            title = f'raw' if pca_comp is None else f'pca{pca_comp}'
            plotting.plot_minDCF_lr(lambda_values, prior_results, 
                        f'{output_dir}/lr_{title}.png', 
                        f'Logistic Regression / {title}')
    
    def _plot_svm_validation_curves(self, DTR: np.ndarray, LTR: np.ndarray, 
                                    output_dir: str) -> None:
        """Plot SVM validation curves"""
        C_values = np.logspace(-4, 2, 15)
        
        for kernel in ['linear', 'RBF', 'poly']:
            DTR_pca = self.apply_pca(DTR, n_components=7)
            
            prior_results = {prior: [] for prior in self.config.priors}
            
            for C_val in C_values:
                model = SVM.SVM()
                for prior in self.config.priors:
                    if kernel == 'linear':
                        params = ('linear', 0.5, False, 1, C_val)
                    elif kernel == 'RBF':
                        params = ('RBF', 0.5, False, 1, C_val, 0, 0, 1e-3)
                    else:  # poly
                        params = ('poly', 0.5, False, 1, C_val, 1, 2, 0)
                    
                    min_dcf, _ = utils.kfolds(DTR_pca, LTR, prior, model, params)
                    prior_results[prior].append(min_dcf)
            
            utils.plot_minDCF_svm(C_values, *prior_results.values(),
                                 f'{output_dir}/svm_{kernel}', f'SVM {kernel} / PCA=7')
    
    def _plot_gmm_validation_curves(self, DTR: np.ndarray, LTR: np.ndarray, 
                                    output_dir: str) -> None:
        """Plot GMM validation curves"""
        components = [2, 4, 8, 16, 32]
        
        for cov_type in ['full', 'diag', 'tied']:
            for pca_comp in [None, 7]:
                DTR_pca = self.apply_pca(DTR, n_components=pca_comp)
                
                prior_results = {prior: [] for prior in self.config.priors}
                
                for n_comp in components:
                    model = GMM.GMM()
                    for prior in self.config.priors:
                        min_dcf, _ = utils.kfolds(DTR_pca, LTR, prior, model, (n_comp, cov_type))
                        prior_results[prior].append(min_dcf)
                
                title = f'raw' if pca_comp is None else f'pca{pca_comp}'
                utils.plot_minDCF_gmm(components, *prior_results.values(),
                                     f'{output_dir}/gmm_{cov_type}_{title}',
                                     f'GMM {cov_type} / {title}')
    
    def score_calibration_analysis(self, DTR: np.ndarray, LTR: np.ndarray,
                                  output_dir: str = 'plots') -> Dict[str, Any]:
        """Perform score calibration analysis"""
        logger.info('Performing score calibration analysis...')
        Path(output_dir).mkdir(exist_ok=True)
        
        # Best models from evaluation
        best_models = [
            ('gaussian_tied_full', 'Tied Full-Cov'),
            ('logistic_regression', 'Logistic Regression'),
            ('linear_svm', 'Linear SVM'),
            ('gmm_full', 'GMM Full-Cov')
        ]
        
        DTR_pca = self.apply_pca(DTR, n_components=7)
        
        calibration_results = {}
        
        for model_key, model_name in best_models:
            logger.info(f'Calibrating {model_name}...')
            
            # Generate Bayes error plots
            p_range = np.linspace(-3, 3, 15)
            min_dcf_vals = []
            act_dcf_vals = []
            
            model_config = self.models[model_key]
            model = model_config.model_class()
            
            for p_val in p_range:
                prior_val = 1.0 / (1.0 + np.exp(-p_val))
                
                # Without calibration
                min_dcf, act_dcf = utils.kfolds(DTR_pca, LTR, prior_val, model, 
                                               model_config.params)
                min_dcf_vals.append(min_dcf)
                act_dcf_vals.append(act_dcf)
            
            # Plot Bayes error
            plotting.bayes_error_plot(p_range, min_dcf_vals, act_dcf_vals,
                         f'{output_dir}/bayes_{model_key}.png', 
                         f'{model_name} / PCA=7')
            
            # With calibration
            min_dcf_cal = []
            act_dcf_cal = []
            
            for p_val in p_range:
                prior_val = 1.0 / (1.0 + np.exp(-p_val))
                min_dcf, act_dcf = utils.kfolds(DTR_pca, LTR, prior_val, model,
                                               model_config.params, calibrated=True)
                min_dcf_cal.append(min_dcf)
                act_dcf_cal.append(act_dcf)
            
            plotting.bayes_error_plot(p_range, min_dcf_cal, act_dcf_cal,
                         f'{output_dir}/bayes_cal_{model_key}.png',
                         f'Calibrated {model_name} / PCA=7')
            
            calibration_results[model_key] = {
                'uncalibrated': {'min_dcf': min_dcf_vals, 'act_dcf': act_dcf_vals},
                'calibrated': {'min_dcf': min_dcf_cal, 'act_dcf': act_dcf_cal}
            }
        
        return calibration_results
    
    def final_evaluation(self, DTR: np.ndarray, LTR: np.ndarray, 
                        DTE: np.ndarray, LTE: np.ndarray,
                        output_dir: str = 'plots') -> Dict[str, Any]:
        """Perform final evaluation on test set"""
        logger.info('Performing final evaluation on test set...')
        Path(output_dir).mkdir(exist_ok=True)
        
        # Apply PCA
        DTR_pca, DTE_pca = self.apply_pca(DTR, DTE, n_components=7)
        
        # Best models
        best_models = [
            ('gaussian_tied_full', 'Tied Full-Cov'),
            ('logistic_regression', 'Logistic Regression'),
            ('linear_svm', 'Linear SVM'),
            ('gmm_full', 'GMM Full-Cov')
        ]
        
        evaluation_results = {}
        calibrated_scores = []
        model_names = []
        colors = ['r', 'b', 'g', 'darkorange']
        
        for (model_key, model_name), color in zip(best_models, colors):
            logger.info(f'Evaluating {model_name} on test set...')
            
            model_config = self.models[model_key]
            model = model_config.model_class()
            
            # Train model
            trained_model = model.trainClassifier(DTR_pca, LTR, *model_config.params)
            
            # Compute calibrated scores
            alpha, beta = utils.compute_calibrated_scores_param(
                trained_model.computeLLR(DTR_pca), LTR)
            
            scores = (alpha * trained_model.computeLLR(DTE_pca) + beta - 
                     np.log(0.5 / (1 - 0.5)))
            
            # Evaluate for each prior
            model_results = {}
            for prior in self.config.priors:
                min_dcf = utils.minDCF(scores, LTE, prior, 1, 1)
                act_dcf = utils.actDCF(scores, LTE, prior, 1, 1)
                
                model_results[f'prior_{prior}'] = {
                    'minDCF': min_dcf,
                    'actDCF': act_dcf
                }
            
            evaluation_results[model_key] = {
                'name': model_name,
                'results': model_results,
                'scores': scores
            }
            
            calibrated_scores.append(scores)
            model_names.append(model_name)
        
        # Generate ROC and DET curves
        roc_data = list(zip(calibrated_scores, model_names, colors))
        utils.plot_ROC(roc_data, LTE, f'{output_dir}/roc_final', 'Final Evaluation / PCA=7')
        utils.plot_DET(roc_data, LTE, f'{output_dir}/det_final', 'Final Evaluation / PCA=7')
        
        return evaluation_results
    
    def generate_report(self, results: Dict[str, Any], output_file: str = 'results_report.txt') -> None:
        """Generate a comprehensive report of all results"""
        logger.info(f'Generating report: {output_file}')
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MACHINE LEARNING PIPELINE RESULTS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary of best models
            f.write("BEST PERFORMING MODELS:\n")
            f.write("-" * 40 + "\n")
            
            for model_key, model_data in results.items():
                if isinstance(model_data, dict) and 'results' in model_data:
                    f.write(f"{model_data['name']}:\n")
                    for prior_key, metrics in model_data['results'].items():
                        f.write(f"  {prior_key}: minDCF={metrics['minDCF']:.3f}, "
                               f"actDCF={metrics['actDCF']:.3f}\n")
                    f.write("\n")
        
        logger.info(f'Report generated: {output_file}')
    
    def run_complete_pipeline(self, train_path: str = 'Train.txt', 
                             test_path: str = 'Test.txt') -> Dict[str, Any]:
        """Run the complete machine learning pipeline"""
        logger.info('Starting complete ML pipeline...')
        
        try:
            # Load data
            (DTR, LTR), (DTE, LTE) = self.load_data(train_path, test_path)
            
            # Feature analysis
            self.plot_features(DTR, LTR)
            
            # Comprehensive evaluation
            eval_results = self.comprehensive_evaluation(DTR, LTR)
            
            # Validation curves
            self.plot_validation_curves(DTR, LTR)
            
            # Score calibration
            calibration_results = self.score_calibration_analysis(DTR, LTR)
            
            # Final evaluation
            final_results = self.final_evaluation(DTR, LTR, DTE, LTE)
            
            # Generate report
            self.generate_report(final_results)
            
            logger.info('Pipeline completed successfully!')
            
            return {
                'evaluation': eval_results,
                'calibration': calibration_results,
                'final': final_results
            }
            
        except Exception as e:
            logger.error(f'Pipeline failed: {e}')
            raise


def main():
    """Main execution function"""
    # Reload modules for development
    importlib.reload(utils)
    importlib.reload(GC)
    importlib.reload(LR)
    importlib.reload(SVM)
    importlib.reload(GMM)
    
    # Configuration
    config = ExperimentConfig(
        priors=[0.5, 0.1, 0.9],
        pca_components=[8, 7, 6, 5],  # 8 means raw (no PCA)
        lambda_range=np.logspace(-5, 1, 20),
        c_range=np.logspace(-4, 2, 15),
        gmm_components=[2, 4, 8, 16, 32]
    )
    
    # Initialize pipeline
    pipeline = MLPipeline(config)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    return results


if __name__ == '__main__':
    main()