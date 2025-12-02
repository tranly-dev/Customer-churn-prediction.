# Customer-churn-prediction.
Predict customer churn using Python, SQL, and Power BI. Logistic Regression model for customer retention analytics.
# Customer Churn Prediction  
**Predicting whether a bank customer will churn using Machine Learning**  


## Objective  
Build and compare classification models to predict customer churn (`Exited = 1`) using the classic **Churn_Modelling.csv** dataset (10,000 customers).

### Goals  
- Clean & preprocess data  
- Train Logistic Regression (baseline) + Random Forest  
- Evaluate with AUC-ROC, Confusion Matrix, Classification Report  
- Save the best model for inference/deployment  

## Dataset  
`data/raw/Churn_Modelling.csv`  
Key features: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited` (target)

## Full Model Training Pipeline (Ready to Run)

```python
# 1. Imports
import numpy as np, pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

# 2. Load data
df = pd.read_csv("./data/raw/Churn_Modelling.csv")

# 3. Preprocessing
drop_cols = ["RowNumber", "CustomerId", "Surname"]
df = df.drop(columns=drop_cols)
X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Pipeline (Scaling + OneHot + Model)
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features)
], verbose_feature_names_out=False)

# Logistic Regression (balanced classes)
pipe_lr = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
])

# Random Forest (usually best performer)
rf = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", 
        random_state=42, n_jobs=-1))
])

# 5. Train & evaluate
rf.fit(X_train, y_train)
proba_rf = rf.predict_proba(X_test)[:, 1]
print("Random Forest AUC:", roc_auc_score(y_test, proba_rf))
print(classification_report(y_test, rf.predict(X_test)))

pipe_lr.fit(X_train, y_train)
print("Logistic Regression AUC:", roc_auc_score(y_test, pipe_lr.predict_proba(X_test)[:, 1]))

# 6. Hyperparameter tuning (optional)
grid = GridSearchCV(pipe_lr, {"model__C": [0.01, 0.1, 1, 3, 10]}, 
                    cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
best_lr = grid.best_estimator_

# 7. Save best model
joblib.dump(rf, "./data/processed/churn_randomforest.pkl")   # usually the winner
joblib.dump(best_lr, "./data/processed/churn_logistic.pkl")

# 8. Single prediction example
sample = X_test.iloc[[0]]
churn_prob = rf.predict_proba(sample)[:, 1][0]
print(f"Customer churn probability: {churn_prob:.2%}")

## Typical Results

| Model               | AUC          | Notes                          |
|---------------------|--------------|---------------------------------|
| Logistic Regression | ~0.85        | Fast & interpretable            |
| Random Forest       | ~0.86–0.87   | Best accuracy, production-ready |
