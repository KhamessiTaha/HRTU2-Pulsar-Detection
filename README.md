# HTRU2 Pulsar Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)

![Pulsar Visualization](results/figures/pca_analysis.png)

A comprehensive machine learning pipeline for detecting pulsars in the HTRU2 dataset using various classification algorithms. This project implements state-of-the-art techniques for astronomical signal processing and classification, addressing the class imbalance challenge inherent in pulsar detection.

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## About the Project

Pulsars are rapidly rotating neutron stars that emit beams of electromagnetic radiation. This project focuses on automating the detection of pulsar candidates from the High Time Resolution Universe Survey (HTRU2) dataset using machine learning techniques.

### Objectives

- Develop robust classification models for pulsar detection
- Handle severe class imbalance (91% non-pulsars vs 9% pulsars)
- Implement feature engineering and selection techniques
- Provide interpretable results using feature importance analysis
- Compare performance across multiple algorithms

### Scientific Context

Pulsar detection is crucial for:
- Understanding neutron star physics
- Gravitational wave detection
- Tests of general relativity
- Galactic structure studies

## Dataset

The HTRU2 dataset contains 17,898 pulsar candidates described by 8 continuous variables:

**Integrated Profile Statistics:**
1. Mean of integrated profile
2. Standard deviation of integrated profile
3. Excess kurtosis of integrated profile
4. Skewness of integrated profile

**DM-SNR Curve Statistics:**
5. Mean of DM-SNR curve
6. Standard deviation of DM-SNR curve
7. Excess kurtosis of DM-SNR curve
8. Skewness of DM-SNR curve

**Target Variable:**
- Class: 0 (non-pulsar) or 1 (pulsar)

**Data Characteristics:**
- Total samples: 17,898
- Pulsars: 1,639 (9.16%)
- Non-pulsars: 16,259 (90.84%)
- Missing values: None

## Project Structure

```
HTRU2-Pulsar-Detection/
├── data/                           # Dataset files
│   ├── HTRU_2.csv                  # Original HTRU2 data                    
├── notebooks/                      # Jupyter notebooks
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_Modeling.ipynb           # Model training and evaluation
│   └── 03_Interpretability.ipynb   # Feature importance and interpretability
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── models.py                  # Model implementations
│   ├── preprocess.py              # Preprocessing methods and functions
│   └── utils.py                   # Utility functions
├── models/                        # Trained model artifacts
│   ├── SVM_best.pkl
│   └── scaler.pkl
├── results/                       # Analysis outputs
│   ├── figures/                   # Visualizations
│   │   ├── confusion_matrices
│   │   ├── correlation_matrix
│   │   ├── data_overview
│   │   ├── error_analysis
│   │   ├── feature_boxplots
│   │   ├── feature_distributions
│   │   ├── partial_dependence_svm
│   │   ├── pr_curves
│   │   ├── roc_curves
│   │   ├── shap_force_pulsar_svm
│   │   ├── shap_summary_svm
│   │   ├── SVM_feature_importance
│   │   ├── threshold_optimization
│   │   └── pca_analysis               
│   └── metrics/                   # Performance metrics
├── paper/                         # Research paper
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── .gitignore                     # Git ignore file
└── README.md                      # This file
```

## Key Findings

### Best Performing Model
- **Algorithm**: Support Vector Machine (SVM) with RBF kernel
- **Validation ROC AUC**: 0.9843
- **Test ROC AUC**: 0.9708
- **Test Precision**: 0.8287
- **Test Recall**: 0.9146
- **Test F1-Score**: 0.8696

### Feature Importance (SVM Analysis)
1. **Excess kurtosis of integrated profile** (1.7413 importance)
2. **Skewness of DM-SNR curve** (0.5286 importance)  
3. **Standard deviation of DM-SNR curve** (0.4870 importance)
4. **Mean of integrated profile** (0.4768 importance)
5. **Excess kurtosis of DM-SNR curve** (0.4661 importance)

### Model Comparison (Validation Performance)
| Model | ROC AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| SVM | 0.9843 | 0.8287 | 0.9146 | 0.8696 |
| LogisticRegression | 0.9837 | 0.7906 | 0.9207 | 0.8507 |
| LightGBM | 0.9729 | 0.8506 | 0.9024 | 0.8757 |
| XGBoost | 0.9720 | 0.8315 | 0.9024 | 0.8655 |
| CatBoost | 0.9714 | 0.8287 | 0.9146 | 0.8696 |

### Cross-Validation Results (ROC AUC)
| Model | CV Mean | CV Std | CV Min | CV Max |
|-------|---------|--------|--------|--------|
| RandomForest | 0.997064 | 0.000780 | 0.996188 | 0.998177 |
| XGBoost | 0.996502 | 0.000754 | 0.995705 | 0.997846 |
| CatBoost | 0.996096 | 0.000678 | 0.995362 | 0.997289 |
| LightGBM | 0.995919 | 0.000786 | 0.994984 | 0.997214 |
| MLP | 0.991762 | 0.001428 | 0.989590 | 0.993915 |

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Jupyter Lab/Notebook

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KhamessiTaha/HTRU2-Pulsar-Detection.git
   cd HTRU2-Pulsar-Detection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or using conda:
   ```bash
   conda env create -f environment.yml
   conda activate htru2-pulsar
   ```

### Quick Start

1. **Download the HTRU2 dataset:**
   ```bash
   # Dataset will be automatically downloaded when running the first notebook
   # Or manually download from: https://archive.ics.uci.edu/ml/datasets/HTRU2
   ```

2. **Run the analysis pipeline:**
   ```bash
   jupyter lab
   ```
   
   Execute notebooks in order:
   - `01_EDA.ipynb` → Exploratory Data Analysis
   - `02_Modeling.ipynb` → Model Training and Evaluation  
   - `03_Interpretability.ipynb` → Feature Importance Analysis and Interpretation

## Usage

### Running Individual Components

**Data Preprocessing:**
```python
from src.preprocess import preprocess_data
X_train, X_test, y_train, y_test = preprocess_data('data/HTRU_2.csv')
```

**Model Training:**
```python
from src.models import train_svm_model
model = train_svm_model(X_train, y_train)
```

**Evaluation:**
```python
from src.utils import evaluate_model
metrics = evaluate_model(model, X_test, y_test)
```

## Methodology

### Data Preprocessing
- **Scaling**: RobustScaler for feature normalization (robust to outliers)
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling Technique)
- **Train/Validation/Test Split**: 70%/10%/20% stratified split

### Model Selection
- **Cross-Validation**: 5-fold stratified cross-validation
- **Hyperparameter Tuning**: GridSearchCV with ROC AUC optimization
- **Multiple Algorithms**: Comparison of 10 different classifiers

### Evaluation Metrics
- **Primary**: ROC AUC (handles class imbalance well)
- **Secondary**: Precision, Recall, F1-Score, MCC
- **Specialized**: PR AUC, Balanced Accuracy, Specificity

## Results

### Performance Summary
The SVM model achieved exceptional performance with a validation ROC AUC of 0.9843 and test ROC AUC of 0.9708, demonstrating excellent discrimination between pulsars and non-pulsars. The model shows:

- **High Test Precision**: 82.87% of predicted pulsars are actual pulsars
- **High Test Recall**: 91.46% of actual pulsars are correctly identified
- **Balanced Performance**: Test F1-score of 86.96% indicates good balance
- **Strong Generalization**: Minimal overfitting between validation and test performance

### Feature Insights
- **Excess kurtosis of integrated profile** is the most discriminative feature
- **DM-SNR curve statistics** (skewness and standard deviation) provide significant classification power
- **Integrated profile statistics** complement DM-SNR features effectively
- Combined features achieve substantially better performance than individual metrics

### Data Processing Pipeline
- **Balanced Training Set**: SMOTE increased training samples from 12,528 to 22,762 (50%/50% class distribution)
- **Robust Scaling**: Applied to handle outliers in astronomical data
- **Stratified Sampling**: Maintains class proportions across splits

### Astronomical Implications
The high performance suggests that machine learning can reliably automate pulsar detection, potentially:
- Reducing manual review time by 90%+
- Discovering new pulsars in large-scale surveys
- Enabling real-time pulsar candidate classification
- Supporting next-generation radio telescope surveys

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- Deep learning model implementations
- Additional feature engineering techniques
- Real-time classification pipeline
- Web interface for model deployment
- Extended dataset integration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **HTRU2 Dataset**: R. J. Lyon et al. (University of Manchester)
- **Original Paper**: Lyon, R. J., et al. "Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach." Monthly Notices of the Royal Astronomical Society 459.1 (2016): 1104-1123.
- **Scikit-learn Community**: For excellent machine learning tools
- **Python Data Science Stack**: NumPy, Pandas, Matplotlib, Seaborn

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{htru2_pulsar_detection,
  author = {Taha Khamessi},
  title = {HTRU2 Pulsar Detection: A Machine Learning Approach},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/KhamessiTaha/HTRU2-Pulsar-Detection}}
}
```

---

**Contact**: taha.khamessi@gmail.com

**Project Link**: [https://github.com/KhamessiTaha/HTRU2-Pulsar-Detection](https://github.com/KhamessiTaha/HTRU2-Pulsar-Detection)
