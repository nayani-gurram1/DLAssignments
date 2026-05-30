import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

st.set_page_config(page_title="Stock Price Predictor")

st.title("📈 Stock Price Prediction")

st.write("Enter last 30 days closing prices:")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model("stock_model.keras")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_all()

# -------------------------------
# USER INPUT (30 DAYS)
# -------------------------------
prices = []

for i in range(30):
    val = st.number_input(f"Day {i+1} Price", value=100.0)
    prices.append(val)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Next Price"):
    try:
        data = np.array(prices).reshape(-1,1)

        # Scale input
        data_scaled = scaler.transform(data)

        # Reshape for model
        data_scaled = np.reshape(data_scaled, (1,30,1))

        # Predict
        pred = model.predict(data_scaled)

        # Inverse scale
        pred_price = scaler.inverse_transform(pred)

        st.success(f"📊 Predicted Next Price: {pred_price[0][0]:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")