# Loan Default Prediction

An end-to-end machine learning project that predicts whether a borrower is likely to default on a loan using statistical analysis, feature engineering, and machine learning models.

---

## Project Overview

Loan default prediction is an important problem in credit risk management. Financial institutions rely on predictive models to identify high-risk borrowers before approving loans.

This project follows a complete machine learning workflow, beginning with raw data and ending with model comparison and evaluation.

The project emphasizes both **statistical reasoning** and **machine learning**, making it suitable for real-world credit risk analysis.

---

## Dataset

**Dataset:** Loan Default Prediction Dataset

**Source:** https://www.kaggle.com/datasets/nikhil1e9/loan-default

**License:** CC0 1.0 (Public Domain)

Dataset contains over **255,000 loan applications** with demographic, financial and loan-related attributes.

---

## Repository Structure

```
loan-default-prediction/
│
├── data/
│   ├── Loan_default.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_Data_Loading_and_Cleaning.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Preprocessing_and_Feature_Engineering.ipynb
│   └── 04_Model_Building_and_Evaluation.ipynb
│
├── requirements.txt
└── README.md
```

---

## Project Workflow

### 1. Data Cleaning

- Imported raw dataset
- Checked data quality
- Missing value analysis
- Duplicate detection
- Data type validation

---

### 2. Exploratory Data Analysis

Performed extensive exploratory analysis including:

- Target class distribution
- Numerical feature distributions
- Categorical feature analysis
- Correlation analysis
- Relationship between features and loan default
- Data visualization using Matplotlib and Seaborn

---

### 3. Statistical Analysis

Several statistical techniques were used to understand the data before modelling.

- Mann–Whitney U Test
- Chi-Square Test
- Cohen's d Effect Size
- Cramér's V
- Spearman Correlation
- Variance Inflation Factor (VIF)

These methods helped identify informative features while checking for multicollinearity and effect size.

---

### 4. Preprocessing & Feature Engineering

The preprocessing pipeline included:

- Missing value treatment
- Outlier handling
- Log transformation
- Feature encoding
- Feature scaling
- Feature selection
- Train-Test split

Processed datasets were saved for reproducible modelling.

---

### 5. Machine Learning Models

Three supervised learning models were implemented and compared.

- Logistic Regression
- Random Forest
- XGBoost

Each model was evaluated before and after hyperparameter tuning where applicable.

---

## Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

The final model was selected based on overall predictive performance and generalization capability.

---

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- XGBoost
- Statsmodels
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Installation

Clone the repository

```bash
git clone https://github.com/harshit87612/loan-default-prediction.git
```

Move into the project

```bash
cd loan-default-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## Future Improvements

- SHAP-based model explainability
- Probability calibration
- Model deployment using Streamlit
- Automated training pipeline
- Hyperparameter optimization using Bayesian search

---

## Author

**Harshit Bhatt**

M.Sc. Statistics  
University of Delhi
