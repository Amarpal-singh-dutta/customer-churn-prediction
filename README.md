# 📉 Customer Churn Prediction
> Predicting telecom customer churn using classical ML with production-grade pipeline design.

---

## Overview

Built a complete churn prediction system on the IBM Telco dataset (~7,000 customers). The focus was on real-world ML concerns: handling class imbalance correctly, preventing data leakage, and optimising for the metric that actually matters to the business — **Recall** (catching churners, not just accuracy).

---

## Dataset

**IBM Telco Customer Churn** — available on Kaggle  
7,043 customers × 21 features including contract type, monthly charges, internet service, tenure, and payment method.

- Target: `Churn` (Yes / No) — ~26% positive (imbalanced)
- Mix of numerical and categorical features
- 11 rows with blank `TotalCharges` (new customers, handled explicitly)

---

## Approach

### EDA
- Churn rate by contract type, internet service, payment method, and tech support
- Numerical distributions (tenure, monthly charges, total charges)
- Boxplots: churned vs non-churned customers across key features
- **Finding:** Month-to-month contract + fiber optic + no tech support = highest risk segment

### Preprocessing Pipeline
- `ColumnTransformer` with separate pipelines for numerical (StandardScaler + median imputation) and categorical (OneHotEncoder + mode imputation) features
- **SMOTE applied inside each CV fold** — not before splitting — to prevent data leakage

### Model Comparison
Five models compared via 5-fold Stratified Cross-Validation:

| Model               | CV Recall | CV F1  | CV ROC-AUC |
| ------------------- | --------- | ------ | ---------- |
| Logistic Regression | 0.7906    | 0.6317 | 0.8446     |
| Decision Tree       | 0.5592    | 0.5251 | 0.6770     |
| Random Forest       | 0.5712    | 0.5807 | 0.8217     |
| XGBoost             | 0.5592    | 0.5744 | 0.8222     |
| ANN (MLP)           | 0.6575    | 0.5803 | 0.7934     |


> **Recall** is the primary metric — missing a churner is more costly than a false alarm.

### Hyperparameter Tuning
- `GridSearchCV` over C, penalty, solver, class_weight for Logistic Regression
- Optimised on Recall across the same stratified folds

### Final Evaluation
- Confusion matrix, classification report, ROC curve on held-out test set
- Model serialised with `joblib`

---

## Results

The tuned Logistic Regression achieved **~80% recall** on the test set, successfully identifying 4 out of every 5 churners — the key business goal.

**Business recommendation:** Target retention campaigns at month-to-month contract customers in their first 12 months with monthly charges above the dataset median.

---

## Tech Stack

```
Python · pandas · scikit-learn · imbalanced-learn · XGBoost · matplotlib · seaborn · joblib
```

---

## Run It

```bash
pip install scikit-learn imbalanced-learn xgboost pandas matplotlib seaborn joblib
```

Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` to `/content/` in Google Colab and run all cells top to bottom.

---

## File Structure

```
Customer_Churn_Prediction.ipynb   ← main notebook
churn_model.pkl                   ← saved best model
```
