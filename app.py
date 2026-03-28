DIGITAL LITERACY AWARENESS QUIZ
https://docs.google.com/forms/d/e/1FAIpQLSfZoYsgfcBRrDP3879CnYjfnWdRxAoJcAIvjIgmKieBh3BluA/viewform?usp=publish-editor


# ---------------- INSTALL REQUIRED LIBRARIES ----------------
!pip install streamlit pyngrok seaborn joblib

# ---------------- IMPORTS ----------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pyngrok import ngrok

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

# ---------------- LOAD DATASET ----------------
data = pd.read_csv("creditcard.csv")

# ---------------- STREAMLIT APP ----------------
def main():
    st.title("💳 Financial Fraud Detection Dashboard")

    # -------- GRAPH 1 --------
    st.subheader("📊 Fraud vs Normal Transactions")
    fig1, ax1 = plt.subplots()
    data['Class'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title("Fraud vs Normal")
    st.pyplot(fig1)

    # -------- GRAPH 2 --------
    st.subheader("📈 Transaction Amount Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(data['Amount'], bins=50, ax=ax2)
    st.pyplot(fig2)

    # -------- GRAPH 3 --------
    st.subheader("🔥 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(data.corr(), cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    # -------- PREDICTION --------
    st.subheader("🔍 Predict Transaction")

    # IMPORTANT: match number of features used in training
    f1 = st.number_input("Amount")
    f2 = st.number_input("Feature 2")
    f3 = st.number_input("Feature 3")
    f4 = st.number_input("Feature 4")
    f5 = st.number_input("Feature 5")

    if st.button("Predict"):
        try:
            features = np.array([[f1, f2, f3, f4, f5]])
            prediction = model.predict(features)
            prob = model.predict_proba(features)[0][1]

            if prediction[0] == 1:
                st.error(f"🚨 Fraud Detected! Probability: {prob:.2f}")
            else:
                st.success(f"✅ Safe Transaction. Probability: {prob:.2f}")

        except Exception as e:
            st.warning(f"Error: {e}")

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    main()
