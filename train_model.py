import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"].fillna(
    df["MonthlyCharges"] * df["tenure"],
    inplace=True
)

# Convert target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Numerical and Categorical Columns
num_features = ["tenure", "MonthlyCharges", "TotalCharges"]

cat_features = [col for col in X.columns if col not in num_features]

# Numerical Pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical Pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Transform Features
X_processed = preprocessor.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

# XGBoost Model
xgb = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

# Hyperparameter Grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

# Grid Search
grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

# Train Model
grid.fit(X_train_resampled, y_train_resampled)

# Best Model
best_model = grid.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(best_model, "churn_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("Model Saved Successfully")