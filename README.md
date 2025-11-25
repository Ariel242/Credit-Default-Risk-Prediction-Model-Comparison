# Credit Default Prediction – Model Comparison

## 1. Overview
This project builds and compares several machine-learning models to predict **credit default risk** at the individual-loan level.  
Goal: identify a robust, interpretable model that differentiates well between **good** and **risky** borrowers, and to extract practical insights for credit-risk policy.

## 2. Business Question
Given applicant and loan characteristics at origination, estimate the probability that a borrower will **default**.  
The focus is on:
- Improving default-prediction accuracy vs. simple benchmarks.
- Understanding which variables drive risk.
- Quantifying the trade-off between approval rate and default rate under different decision thresholds.

## 3. Data
- **Source:** Kaggle – *Credit Risk Dataset*  
  https://www.kaggle.com/datasets/laotse/credit-risk-dataset
- **Target variable:** `loan_status`  
  - `0` – non-default  
  - `1` – default
- **Key feature groups:**
  - Applicant profile: age, income, employment length, home ownership, credit history.
  - Loan characteristics: amount, interest rate, grade, purpose.
  - Credit bureau information: default on file, credit history length.

Basic preprocessing includes:
- Handling missing values and obvious data issues.
- Type conversion (numeric / factor / ordered factor).
- Train/validation/test split with stratification on `loan_status`.

## 4. Methodology

1. **Exploratory Data Analysis (EDA)**
   - Target distribution and class imbalance.
   - Univariate and bivariate analysis.
   - Initial data quality checks.

2. **Preprocessing & Feature Engineering**
   - Encoding of categorical variables.
   - Scaling for models that require it.
   - Creation of train / validation / test sets.

3. **Model Training & Tuning**
   Using **R** and the `caret` framework with cross-validation:
   - Hyperparameter tuning for each model.
   - Class-imbalance handling via class weights / resampling.

4. **Model Evaluation & Comparison**
   - Main metrics: **ROC-AUC**, **PR-AUC**, **Accuracy**, **Sensitivity**, **Specificity**.
   - Threshold selection using **Youden’s J statistic** on the validation set.
   - Confusion matrices and lift/precision–recall analysis on the **test** set.
   - Short written insights for each model and a final comparison table.

## 5. Models Implemented

Planned/implemented models:

- Logistic Regression (baseline)
- LASSO / Ridge Logistic Regression
- Decision Tree (conditional inference tree)
- Random Forest
- Gradient Boosting / XGBoost
- Neural Network (shallow MLP via `nnet`)

Each model is trained on the same train/validation splits and evaluated on the same test set to ensure a fair comparison.

## 6. Repository Structure

Planned R-script structure (numbered pipeline):

- `01_load_and_clean_data.R` – data loading, cleaning, basic preprocessing.
- `02_exploratory_data_analysis.R` – EDA and plots.
- `03_feature_engineering_and_splits.R` – feature preparation, train/valid/test creation.
- `04_model_logistic_regression.R`
- `05_model_regularized_logit.R`
- `06_model_decision_tree.R`
- `07_model_random_forest.R`
- `08_model_xgboost.R`
- `09_model_neural_network.R`
- `10_model_calibration_and_thresholds.R` (optional, depending on final design)
- `11_models_performance_comparison_insights.R` – **final comparison across models + summary table + key insights.**

Additional folders (to be created as needed):

- `data/` – raw dataset (`credit_risk_dataset.csv`) and any processed versions.
- `output/` – saved models, performance tables, and CSVs.
- `figures/` – ROC/PR curves, calibration plots, feature-importance charts.
- `docs/` – slides, reports, and notes.

## 7. Technical Stack

- **Language:** R
- **Key Packages:** `dplyr`, `ggplot2`, `caret`, `pROC`, `precrec`,  
  plus model-specific packages: `randomForest`, `xgboost`, `party`, `nnet`, etc.

## 8. How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/credit-default-model-comparison.git
   cd credit-default-model-comparison
