import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the following details to check your heart disease risk.")

# Input Fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])  # Usually 0=normal, 1=fixed defect, etc.

# Predict Button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Output
    if prediction == 1:
        st.error("⚠️ High risk of heart disease. Please consult a doctor.")
    else:
        st.success("✅ Low risk of heart disease.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")