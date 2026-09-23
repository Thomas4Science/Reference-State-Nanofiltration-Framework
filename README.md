# Reference-State-Nanofiltration-Framework

## Overview
This repository contains the dataset and complete machine learning pipeline for the paper: **"A reference-state framework for competitive cation transport in mixed-salt nanofiltration"**. 

Predicting ion rejection in mixed-salt nanofiltration remains challenging because the transport of a target ion is simultaneously influenced by membrane properties and interactions with coexisting ions. Here, we develop a hierarchical, physics-guided machine-learning framework that treats single-salt cation rejection as a transport reference state and quantifies the mixture-associated deviation introduced by cation competition. Symbolic regression further compresses these relationships into explicit low-dimensional surrogates. This reference-state framework provides a mechanistically organized description of competitive cation transport and offers a basis for interpreting ion-selective nanofiltration in multicomponent solutions.

## Data Availability
The complete dataset used in this study is provided in the repository:
- `Supplementary_Data_Raw_and_Processed_Datasets.xlsx`

This comprehensive Excel file contains the original raw dataset alongside the staged datasets and intermediate outputs generated at key step of the machine learning pipeline, from raw data collection to the final physical feature sets.

## Prerequisites and Dependencies
The code is tested on Python 3.8+. To ensure full reproducibility of the machine learning pipeline and symbolic regression models, please install the following required packages:

- `pandas>=1.3.0`
- `numpy>=1.20.0`
- `openpyxl>=3.0.0`
- `scikit-learn>=1.0.2`
- `xgboost>=1.5.0`
- `lightgbm>=3.3.0`
- `pysr>=0.16.0`
- `shap>=0.40.0`
- `matplotlib>=3.4.0`
- `seaborn>=0.11.0`
- `plotly>=5.0.0`

### Quick Installation
You can instantly install all the necessary dependencies using `pip` by running the following command in your terminal:

```bash
pip install pandas>=1.3.0 numpy>=1.20.0 openpyxl>=3.0.0 scikit-learn>=1.0.2 xgboost>=1.5.0 lightgbm>=3.3.0 pysr>=0.16.0 shap>=0.40.0 matplotlib>=3.4.0 seaborn>=0.11.0 plotly>=5.0.0
```
## Code Pipeline & Outputs

The machine learning and symbolic regression pipeline is divided into logical steps. Please run the Python scripts in the numerical order provided below. 

All scripts read from and append data to the core dataset file: `Supplementary_Data_Raw_and_Processed_Datasets.xlsx`.

#### Phase 1: Data Preprocessing & Feature Engineering
| Script | Description | Key Outputs |
| :--- | :--- | :--- |
| `code01_data_preprocessing.py` | Data cleaning and extraction of dimensionless physical descriptors (e.g., Steric, Donnan, Peclet numbers). | Appends `Sheet4`, `Sheet5`, `Sheet6` to Excel. |
| `code02_pearson_analysis.py` | Pearson correlation analysis to remove highly collinear redundant features. | `Pearson_Matrix.png`, Appends `Sheet7`, `Sheet8`. |
| `code03_infinity_handling.py` | Physics-constrained clipping to handle infinite values for model stability. | Appends `Sheet9` (Final clean data for baseline). |

#### Phase 2: Single-Salt Reference Model
| Script | Description | Key Outputs |
| :--- | :--- | :--- |
| `code04_model_selection.py` | Hyperparameter grid search and evaluation of baseline models (RF, XGB, LGBM, etc.). | Console metrics, `Hyperparameter_Search_Plot.html`. |
| `code05_shap_analysis.py` | SHAP analysis (Beeswarm and Bar plots) to interpret single-salt transport mechanisms. | `SHAP_Beeswarm.tif/svg`, `SHAP_Bar.tif/svg`, CSV data. |
| `code06_pdp_ice_analysis.py` | 1D and 2D Partial Dependence Plots (PDP) and ICE curves. | 16 High-res PNG plots (1D trends & 2D contours). |

#### Phase 3: Mixed-Salt Deviation Extraction & Competitive Modeling
| Script | Description | Key Outputs |
| :--- | :--- | :--- |
| `code07_mixed_salt_residual.py` | Calculates the mixed-salt rejection deviation ($\Delta R$) from the single-salt baseline. | Appends `Sheet14_Mix_Resi_ML` (Final residual data). |
| `code08_mixed_correlation.py` | Lower-triangle correlation heatmap for competitive physical descriptors. | `Pearson_Correlation_Triangle.png`. |
| `code09_mixed_nested_cv.py` | Nested cross-validation (20 splits) for mixed-salt deviation prediction. | `ML_Performance_Summary.csv`, `ML_TrainTest_BoxPlot.png`. |
| `code10_lightgbm_residual_shap.py` | SHAP analysis identifying the dominant competitive mechanisms (e.g., Concentration Fraction). | `Parity_Data.csv`, SHAP Beeswarm & Bar PNGs. |
| `code11_mixed_residual_pdp.py` | Generates aligned mixed-salt features and outputs 1D/2D PDP & ICE curves for deviation. | Appends `Sheet10_Mix1/2/3`, 6 High-res PDP PNG plots. |

#### Phase 4: Physics-Constrained Symbolic Regression
| Script | Description | Key Outputs |
| :--- | :--- | :--- |
| `code12_single_salt_symbolic_regression.py` | Discovers explicit analytical equations for the single-salt reference baseline. | Pareto CSV, Pareto Front PNG, Parity PNG, **Single-Salt Equation**. |
| `code13_mixed_salt_symbolic_regression.py` | Derives the final analytical surrogate models for the competitive rejection deviation (r). | Final Parity CSV, Pareto PNG, Final Parity PNG, **Mixed-Salt Residual Equation**. |
