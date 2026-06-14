import streamlit as st
import pandas as pd
import joblib

# Load Model and Preprocessor
model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Page Config
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("Customer Churn Prediction System")

st.write("Enter customer details to predict churn probability.")

# =========================
# USER INPUTS
# =========================

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

Partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

Dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

PhoneService = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

Contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# =========================
# PREDICTION
# =========================

if st.button("Predict Churn"):

    input_data = pd.DataFrame({

        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]

    })

    # Preprocess Input
    processed_data = preprocessor.transform(input_data)

    # Prediction
    prediction = model.predict(processed_data)[0]

    # Probability
    probability = model.predict_proba(processed_data)[0][1]

    st.subheader("Prediction Result")

    # Output
    if prediction == 1:
        st.error("Customer is likely to CHURN")
    else:
        st.success("Customer is likely to STAY")

    st.write(f"Churn Probability: {probability:.2%}")

    # Risk Category
    if probability >= 0.75:
        st.warning("Risk Category: HIGH RISK")

    elif probability >= 0.40:
        st.info("Risk Category: MEDIUM RISK")

    else:
        st.success("Risk Category: LOW RISK")