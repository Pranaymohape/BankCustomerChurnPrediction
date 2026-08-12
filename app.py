import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Setup dynamic path resolution to prevent FileNotFoundError on cloud deployments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "churn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# Load pre-trained model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Configure Streamlit page layout and metadata
st.set_page_config(
    page_title="Bank Customer Churn Predictor", page_icon="🏦", layout="centered"
)

# Header Section
st.title("🏦 Bank Customer Churn Prediction")
st.write(
    "Provide customer details below to predict the likelihood of account"
    " closure (Churn Risk)."
)
st.markdown("---")

# User Input Form Layout (Divided into two columns)
col1, col2 = st.columns(2)

with col1:
  credit_score = st.number_input(
      "Credit Score:", min_value=300, max_value=850, value=650
  )
  country = st.selectbox("Country:", ["France", "Germany", "Spain"])
  gender = st.selectbox("Gender:", ["Female", "Male"])
  age = st.number_input("Age:", min_value=18, max_value=100, value=38)
  tenure = st.number_input(
      "Tenure (Years with bank):", min_value=0, max_value=10, value=5
  )

with col2:
  balance = st.number_input(
      "Account Balance ($):", min_value=0.0, value=75000.0, step=1000.0
  )
  products_number = st.selectbox(
      "Number of Products Used:", [1, 2, 3, 4]
  )
  credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
  active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
  estimated_salary = st.number_input(
      "Estimated Salary ($):", min_value=0.0, value=100000.0, step=1000.0
  )

# Preprocessing UI inputs to match model features
gender_val = 1 if gender == "Male" else 0
credit_card_val = 1 if credit_card == "Yes" else 0
active_member_val = 1 if active_member == "Yes" else 0

# Encoded values matching 'country' dummy variables from training
country_Germany = 1 if country == "Germany" else 0
country_Spain = 1 if country == "Spain" else 0

st.markdown("---")

# Prediction Trigger Button
if st.button("Predict Churn Risk 🔮", use_container_width=True):
  # Construct input array in the exact feature order expected by the model
  input_data = np.array([[
      credit_score,
      gender_val,
      age,
      tenure,
      balance,
      products_number,
      credit_card_val,
      active_member_val,
      estimated_salary,
      country_Germany,
      country_Spain,
  ]])

  # Apply the saved StandardScaler transformation to the user input
  input_scaled = scaler.transform(input_data)

  # Perform inference using the Logistic Regression model
  prediction = model.predict(input_scaled)[0]
  probability = model.predict_proba(input_scaled)[0][1] * 100

  # Display Prediction Results
  if prediction == 1:
    st.error(
        f"⚠️ **High Risk!** This customer is likely to churn from the bank."
        f" (**{probability:.1f}% Churn Probability**)"
    )
    st.info(
        "💡 **Recommendation:** Offer special retention incentives, customized"
        " offers, or targeted customer support."
    )
  else:
    st.success(
        f"✅ **Low Risk!** This customer is likely to stay with the bank."
        f" (**{100-probability:.1f}% Retention Probability**)"
    )