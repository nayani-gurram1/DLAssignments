import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer will churn or not.")

# -------------------------------
# LOAD MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model("churn_model.keras")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))
    return model, scaler, columns

model, scaler, columns = load_all()

# -------------------------------
# USER INPUTS
# -------------------------------
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0, 150, 70)
total_charges = tenure * monthly_charges

contract = st.selectbox("Contract Type", [
    "Month-to-month", "One year", "Two year"
])

internet_service = st.selectbox("Internet Service", [
    "DSL", "Fiber optic", "No"
])

payment_method = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

# -------------------------------
# PREPROCESS INPUT
# -------------------------------
def preprocess():
    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method
    }

    df = pd.DataFrame([data])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    return df_scaled

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Churn"):
    try:
        input_data = preprocess()
        prediction = model.predict(input_data)[0][0]

        st.subheader("Result")

        if prediction > 0.5:
            st.error(f"⚠️ High Risk of Churn ({prediction:.2f})")
        else:
            st.success(f"✅ Customer Likely to Stay ({prediction:.2f})")

    except Exception as e:
        st.error(f"Error: {e}")