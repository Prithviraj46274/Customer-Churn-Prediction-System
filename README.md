# Customer-Churn-Prediction-System

## Overview

The Customer Churn Prediction System is a Machine Learning application that predicts whether a telecom customer is likely to churn (leave the service) based on customer demographics, subscription details, billing information, and service usage patterns.

The project uses an XGBoost Classifier with hyperparameter tuning through GridSearchCV and handles class imbalance using SMOTE. A Streamlit-based web interface allows users to enter customer details and receive real-time churn predictions along with churn probability and risk level.

---

## Features

* Predicts customer churn using Machine Learning.
* Interactive Streamlit web application.
* Data preprocessing using Scikit-learn Pipelines.
* Missing value handling with SimpleImputer.
* Feature scaling using StandardScaler.
* Categorical feature encoding using OneHotEncoder.
* Class imbalance handling using SMOTE.
* Hyperparameter optimization using GridSearchCV.
* Displays churn probability and risk category.

---

## Machine Learning Pipeline

### Data Preprocessing

* Removes unnecessary customer identifiers.
* Converts TotalCharges to numeric values.
* Handles missing values.
* Separates numerical and categorical features.
* Applies:

  * StandardScaler for numerical features.
  * OneHotEncoder for categorical features.
  * SimpleImputer for missing value treatment.

### Model Training

The project uses:

* XGBoost Classifier
* SMOTE for balancing classes
* GridSearchCV for hyperparameter tuning

The model is trained on the Telco Customer Churn dataset and the best-performing model is saved for deployment.

### Saved Files

After training, the following files are generated:

```text
churn_model.pkl
preprocessor.pkl
```

These files are required for running the Streamlit application.

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── train_model.py
├── app.py
├── churn_model.pkl
├── preprocessor.pkl
├── requirements.txt
├── README.md
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1: Train the Model

Run the training script:

```bash
python train_model.py
```

This will:

* Load and preprocess the dataset
* Train the XGBoost model
* Apply SMOTE balancing
* Perform hyperparameter tuning using GridSearchCV
* Save:

  * `churn_model.pkl`
  * `preprocessor.pkl`

---

### Step 2: Launch the Streamlit Application

After the model files are generated, run:

```bash
streamlit run app.py
```

The application will open in your browser, typically at:

```text
http://localhost:8501
```

---

## Input Features

The application accepts the following customer information:

* Gender
* Senior Citizen Status
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract Type
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

---

## Prediction Output

The system provides:

### Churn Prediction

* Customer is likely to CHURN
* Customer is likely to STAY

### Churn Probability

Displays the probability of customer churn as a percentage.

### Risk Category

| Probability | Risk Level  |
| ----------- | ----------- |
| >= 75%      | High Risk   |
| 40% - 74%   | Medium Risk |
| < 40%       | Low Risk    |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-Learn (SMOTE)
* Joblib
* Streamlit

---

## Dataset

Dataset Used:

**Telco Customer Churn Dataset**

The dataset contains customer demographics, account information, service subscriptions, and churn status, which are used to train the prediction model.

---

Developed as a Machine Learning project for Customer Churn Prediction using XGBoost, Scikit-learn, and Streamlit.
