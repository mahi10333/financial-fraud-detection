import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Fraud Detection App")

st.title("💳 Financial Fraud Detection System")
st.write("Enter transaction details below:")

# Create input fields (example: 5 features)
f1 = st.number_input("Amount")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")
f4 = st.number_input("Feature 4")
f5 = st.number_input("Feature 5")

# Prediction button
if st.button("Predict"):
    try:
        features = np.array([[f1, f2, f3, f4, f5]])
        prediction = model.predict(features)

        if prediction[0] == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Safe Transaction")

    except:
        st.warning("Please enter valid data")
