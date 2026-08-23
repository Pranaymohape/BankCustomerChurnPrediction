import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main {
        background: #f7f9fc;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 2rem 2.2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.25rem;
    }

    .hero p {
        margin: 0.6rem 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: yellow;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }

    .model-card {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    .model-card .label {
        color: #64748b;
        font-size: 0.85rem;
    }

    .model-card .value {
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    .result-box {
        padding: 1.3rem;
        border-radius: 16px;
        margin-top: 1rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3rem;
        font-size: 1rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL BUNDLE
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
bundle_path = os.path.join(BASE_DIR, "best_classification_bundle.pkl")

try:
    bundle = joblib.load(bundle_path)

    model = bundle["model"]
    scaler = bundle["scaler"]
    encoder = bundle["encoder"]
    numerical_features = bundle["numerical_features"]
    nominal_features = bundle["nominal_features"]

except FileNotFoundError:
    st.error(
        "❌ best_classification_bundle.pkl was not found. "
        "Keep the .pkl file in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Could not load the model bundle: {e}")
    st.stop()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🏦 Bank Customer Churn Prediction</h1>
    <p>
        Enter customer details to estimate the likelihood of account closure
        using the selected Random Forest classification model.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# CUSTOMER INPUTS
# =========================================================
st.markdown('<div class="section-title">👤 Customer Information</div>',
            unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1
    )

    country = st.selectbox(
        "Country",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=38,
        step=1
    )

    tenure = st.number_input(
        "Tenure (Years with Bank)",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )

with right:
    balance = st.number_input(
        "Account Balance ($)",
        min_value=0.0,
        value=75000.0,
        step=1000.0
    )

    products_number = st.selectbox(
        "Number of Products Used",
        [1, 2, 3, 4]
    )

    credit_card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

    active_member = st.selectbox(
        "Is Active Member?",
        ["Yes", "No"]
    )

    estimated_salary = st.number_input(
        "Estimated Salary ($)",
        min_value=0.0,
        value=100000.0,
        step=1000.0
    )

st.markdown("---")

# =========================================================
# PREDICTION
# =========================================================
if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):

    try:
        # ---------------------------------------------
        # 1. Create numerical input DataFrame
        # ---------------------------------------------
        numerical_input = pd.DataFrame([{
            "credit_score": credit_score,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "products_number": products_number,
            "credit_card": 1 if credit_card == "Yes" else 0,
            "active_member": 1 if active_member == "Yes" else 0,
            "estimated_salary": estimated_salary
        }])

        # Make sure the order matches training
        numerical_input = numerical_input[numerical_features]

        # ---------------------------------------------
        # 2. Apply the SAME StandardScaler
        # ---------------------------------------------
        scaled_values = scaler.transform(numerical_input)

        scaled_df = pd.DataFrame(
            scaled_values,
            columns=numerical_features
        )

        # ---------------------------------------------
        # 3. Create categorical input DataFrame
        # ---------------------------------------------
        categorical_input = pd.DataFrame([{
            "country": country,
            "gender": gender
        }])

        categorical_input = categorical_input[nominal_features]

        # ---------------------------------------------
        # 4. Apply the SAME OneHotEncoder
        # ---------------------------------------------
        encoded_values = encoder.transform(categorical_input)

        encoded_columns = encoder.get_feature_names_out(
            nominal_features
        )

        encoded_df = pd.DataFrame(
            encoded_values,
            columns=encoded_columns
        )

        # ---------------------------------------------
        # 5. Combine exactly like training
        # ---------------------------------------------
        final_input = pd.concat(
            [scaled_df, encoded_df],
            axis=1
        )

        # ---------------------------------------------
        # 6. Prediction using Random Forest
        # ---------------------------------------------
        prediction = model.predict(final_input)[0]

        if hasattr(model, "predict_proba"):
            churn_probability = model.predict_proba(final_input)[0][1] * 100
        else:
            churn_probability = None

        # ---------------------------------------------
        # 7. Display result
        # ---------------------------------------------
        st.markdown("### 📊 Prediction Result")

        if prediction == 1:
            st.error(
                f"⚠️ **High Churn Risk**\n\n"
                f"This customer is likely to leave the bank."
            )

            if churn_probability is not None:
                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.1f}%"
                )

            

        else:
            retention_probability = (
                100 - churn_probability
                if churn_probability is not None
                else None
            )

            st.success(
                "✅ **Low Churn Risk**\n\n"
                "This customer is likely to stay with the bank."
            )

    except Exception as e:
        st.error(
            f"❌ Prediction could not be completed: {e}"
        )
    
