import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load the model once
with open("model/heart_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Heart Prediction", layout="centered")
st.title("❤️ Heart Attack Prediction System")
st.markdown("### Enter patient details below 👇")

# Input columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1)
    chol = st.number_input("Cholesterol")
    bp = st.number_input("Resting BP")

with col2:
    hr = st.number_input("Max Heart Rate")
    oldpeak = st.number_input("Oldpeak")
    bmi = st.number_input("BMI")

risk = st.slider("Risk Factor", 0.0, 1.0)

# Predict button
if st.button("Predict"):
    # Adjust feature order to match your trained model
    input_data = [[age, chol, bp, hr, oldpeak, bmi, risk]]
    result = model.predict(input_data)[0]

    if result == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk")

# Data visualization
st.markdown("## 📊 Data Visualization")
df = pd.read_csv("data/heart_updated.csv", sep='\t')

fig, ax = plt.subplots()
df['Age'].hist(ax=ax)
ax.set_title("Age Distribution")
st.pyplot(fig)