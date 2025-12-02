# 🏦 Bank Customer Churn Prediction

**Predicting whether a bank customer will leave (churn) using Machine Learning**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Customer-Churn-Prediction/blob/main/notebooks/Churn_Prediction.ipynb)

## Overview
A complete end-to-end churn prediction project using the famous **Bank Churn Modelling** dataset (~10,000 customers).

The goal is to identify customers at risk of leaving the bank so that retention strategies can be applied proactively.

### Key Results (Random Forest)
| Metric           | Score    | Notes                              |
|------------------|----------|------------------------------------|
| **Accuracy**     | **86.45%**   | Overall correct predictions        |
| **Precision**    | 78.33%   | When predicted churn → 78% correct |
| **Recall**       | 46.19%   | Captures ~46% of actual churners   |
| **F1-Score**     | 58.11%   | Balance between precision & recall |
| **ROC AUC**      | **0.8526**   | Strong discrimination power        |

**Confusion Matrix**  


## Dataset
- 10,000 bank customers
- Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- Target: `Exited` (1 = churned, 0 = stayed)
- ~20% churn rate (imbalanced classes)

## What Was Done
1. **EDA** – Explored churn patterns by Age, Geography, Active status, etc.
2. **Preprocessing**
   - One-Hot Encoding for `Geography` & `Gender`
   - Dropped irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
3. **Train/Test Split** – 80/20 stratified split
4. **Model** – RandomForestClassifier (baseline + best performer)
5. **Evaluation** – Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix

## Next Steps / Improvements
- Try **XGBoost / LightGBM** – often better on tabular data
- Apply **SMOTE / class weights** to improve recall for churn class
- **Hyperparameter tuning** (GridSearch / Optuna)
- **Feature importance analysis** with SHAP values
- Build a **Streamlit / FastAPI** deployment app for real-time predictions

## How to Run
1. Clone the repo
2. Open `notebooks/Churn_Prediction.ipynb` -  Remember to upload the raw Churn_modelling.csv 
3. Run all cells 
