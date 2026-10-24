"""Beginner-friendly Streamlit website for transaction classification."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_model.joblib"

st.set_page_config(page_title="Online Fraud Detection", page_icon="🛡️", layout="centered")
st.title("Online Fraud Detection System")
st.caption("AI Application Lab • Random Forest • Educational prototype")
st.warning("This demo uses a synthetic teaching dataset. Its prediction is not proof of fraud and must not be used for real payment decisions.")

if not MODEL_PATH.exists():
    st.error("The model file is missing. First run: python train_model.py")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

st.subheader("Enter transaction details")
with st.form("transaction_form"):
    amount = st.number_input("Transaction amount", min_value=0.0, value=50.0, step=1.0)
    hour = st.slider("Transaction hour (0–23)", min_value=0, max_value=23, value=14)
    new_device = st.selectbox("Is this a new device?", options=["No", "Yes"])
    international = st.selectbox("Is this an international transaction?", options=["No", "Yes"])
    failed_attempts = st.number_input("Number of failed attempts", min_value=0, max_value=100, value=0, step=1)
    account_age = st.number_input("Account age (days)", min_value=0, value=365, step=1)
    submitted = st.form_submit_button("Check transaction", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "transaction_amount": amount,
        "transaction_hour": hour,
        "new_device": 1 if new_device == "Yes" else 0,
        "international_transaction": 1 if international == "Yes" else 0,
        "failed_attempts": failed_attempts,
        "account_age_days": account_age,
    }], columns=features)
    prediction = int(model.predict(row)[0])
    fraud_score = float(model.predict_proba(row)[0][1])
    st.subheader("Model result")
    if prediction == 1:
        st.error("Flagged as potentially fraudulent")
    else:
        st.success("Classified as legitimate by the model")
    st.metric("Model-estimated fraud score", f"{fraud_score:.1%}")
    st.caption("This score is the Random Forest's estimated class probability on synthetic data; it is not a calibrated real-world fraud probability.")
    with st.expander("See the values sent to the model"):
        st.dataframe(row, use_container_width=True, hide_index=True)
