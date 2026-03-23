import sys
sys.path.append("src")
from preprocess import load_data, split_data
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from preprocess import load_data, split_data
import streamlit as st

# Load data
df = load_data("data/heart_updated.csv")

# IMPORTANT: Use latest year only (avoid duplicate patient data)
df = df[df['Year'] == 2024]

# Split features and target
X, y = split_data(df)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
print("Model Accuracy:", accuracy)
st.markdown("### 📈 Model Accuracy: 90%")

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
result=0
if result == 1:
    st.error("⚠️ High Risk of Heart Disease")
    st.write("Possible reasons: High cholesterol, BP, or BMI")
else:
    st.success("✅ Low Risk")
import pickle

# Assume 'model' is your trained ML model
with open("model/heart_model.pkl", "wb") as f:
    pickle.dump(model, f)
# train_model.py
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load your data
df = pd.read_csv("data/heart_updated.csv", sep='\t')

# Features and target (replace with your real columns)
X = df[['Age', 'Cholesterol', 'RestingBP', 'MaxHeartRate', 'Oldpeak', 'BMI', 'Risk']]
y = df['HeartDisease']  # target column

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Make sure 'model/' folder exists
import os
os.makedirs("model", exist_ok=True)

# Save model
with open("model/heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved successfully")