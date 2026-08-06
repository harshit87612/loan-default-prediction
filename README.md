# Loan Default Prediction

A machine learning project that predicts whether a borrower will default on a loan using a dataset of over 255,000 consumer lending records. The project follows a complete machine learning pipeline, from exploratory data analysis and formal statistical testing to preprocessing and model comparison across Logistic Regression, Random Forest, and XGBoost, with an emphasis on statistically justifying every preprocessing decision rather than applying it by convention.

## Project Structure

```text
loan-default-prediction/
├── data/
│   ├── Loan_default.csv          # raw dataset
│   └── processed/
│       └── train_test_data.pkl   # cleaned, encoded, scaled train/test splits
├── notebooks/
│   ├── 01_eda.ipynb              # exploratory data analysis & hypothesis testing
│   ├── 02_preprocessing.ipynb    # cleaning, encoding, scaling, splitting
│   └── 03_modeling.ipynb         # model training, tuning, evaluation, comparison
├── requirements.txt
└── README.md
```

## Dataset

- **Source:** [Loan Default Prediction Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default/data) (Kaggle, nikhil1e9)
- **License:** [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
- **Size:** 255,347 rows × 18 columns
- **Target:** `Default` (binary) - 88.4% non-default, 11.6% default (imbalanced)
- **Features:** Demographic information (Age, Income, Education, Marital Status), loan attributes (Loan Amount, Interest Rate, Loan Term, Loan Purpose), and credit/employment history (Credit Score, Months Employed, Employment Type, DTI Ratio, Number of Credit Lines, Has Mortgage, Has Dependents, Has Co-Signer)

---

## Methodology

### 1. Exploratory Data Analysis (`01_eda.ipynb`)

The analysis began with understanding the structure and quality of the dataset before building any models.

- Checked feature distributions, missing values (none), and duplicate records (none).
- Assessed outliers using the 1.5×IQR rule, and no numeric feature contained observations outside the accepted range.
- Evaluated multicollinearity using **Pearson correlation** and **Variance Inflation Factor (VIF)**, which indicated negligible correlation among numeric predictors.
- Performed **Levene's test** to assess variance homogeneity. Since equal variances could not be assumed, **Welch's independent samples t-test** was used instead of Student's t-test.
- Tested every feature for association with the target variable:
  - **Categorical features:** Chi-square test of independence with Cramér's V for effect size.
  - **Numerical features:** Welch's independent samples t-test with Cohen's d for effect size.
- A key finding was that **LoanTerm** was the only feature not significantly associated with loan default (p = 0.78), so it was removed. Although most remaining variables showed only small or negligible individual effect sizes, they were retained because weak predictors can still improve model performance when considered together.

---

### 2. Preprocessing (`02_preprocessing.ipynb`)

The preprocessing pipeline was designed to prepare the data while preventing information leakage.

- Removed `LoanID` (identifier with no predictive value) and `LoanTerm` (not statistically significant).
- Binary encoded `HasMortgage`, `HasDependents`, and `HasCoSigner` (Yes/No → 1/0).
- One-hot encoded `Education`, `EmploymentType`, `MaritalStatus`, and `LoanPurpose` using `drop_first=True` to avoid the dummy variable trap.
- Performed an 80/20 train-test split with stratification to preserve the original class distribution.
- Applied `StandardScaler` to numerical features, fitting the scaler only on the training data before transforming the test set.

---

### 3. Modeling (`03_modeling.ipynb`)

Three supervised learning algorithms were trained in increasing order of complexity. Each model was first implemented with baseline settings and then optimized through hyperparameter tuning.

| Model | Tuning method | Notes |
|---|---|---|
| Logistic Regression | GridSearchCV (5-fold stratified CV) over `C` and solver | `class_weight='balanced'` used to address class imbalance |
| Random Forest | RandomizedSearchCV (30 iterations, 5-fold CV) | Baseline model showed severe overfitting, with train ROC-AUC = 1.00 vs test ROC-AUC ≈ 0.74; tuning substantially reduced this |
| XGBoost | RandomizedSearchCV (30 iterations, 5-fold CV) | Used `scale_pos_weight` to account for class imbalance |

Because the dataset is highly imbalanced, model evaluation prioritized **ROC-AUC** and **Average Precision** over overall accuracy, which can be misleading in imbalanced classification problems.

---

## Results

Final tuned models evaluated on the held-out test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.677 | 0.220 | **0.700** | 0.335 | 0.753 |
| Random Forest | **0.785** | **0.279** | 0.534 | **0.366** | 0.754 |
| XGBoost | 0.691 | 0.227 | 0.690 | 0.342 | **0.759** |

No single model outperformed the others across every metric.

- **XGBoost** achieved the highest ROC-AUC and Average Precision, demonstrating the strongest overall ability to distinguish between defaulters and non-defaulters.
- **Random Forest** produced the highest Accuracy, Precision, and F1-score, providing the best overall balance at the chosen decision threshold.
- **Logistic Regression** achieved the highest Recall, making it the most effective model for identifying actual defaulters while remaining the easiest to interpret.

Across all three models, the most influential predictors were consistently **Age**, **Interest Rate**, **Income**, **Months Employed**, and **Loan Amount**, with **Employment Status** and **Co-Signer Status** also contributing meaningfully to prediction.

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn (Logistic Regression, Random Forest, GridSearchCV, RandomizedSearchCV, StandardScaler)
- XGBoost
- SciPy
- statsmodels
- matplotlib
- seaborn

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/harshit87612/loan-default-prediction.git
cd loan-default-prediction
pip install -r requirements.txt
```

Run the notebooks in the following order:

```text
01_eda.ipynb
→ 02_preprocessing.ipynb
→ 03_modeling.ipynb
```

Each notebook reads from and writes to the `../data/` directory, so they should be executed from within the `notebooks/` folder.

---

## Limitations

- The dataset is synthetic and may not fully capture the complexity of real-world loan portfolios.
- Models were evaluated using a single train-test split. Although hyperparameter tuning used cross-validation, no external validation dataset was available.
- Predictions are based on a fixed decision threshold of 0.5. Different thresholds may be preferable depending on business objectives, such as maximizing recall in risk-averse lending.
- Feature importance reflects predictive contribution and should not be interpreted as evidence of causality.

---

## License

The dataset is licensed under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).
