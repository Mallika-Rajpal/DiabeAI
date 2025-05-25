import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model/ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')

# Theme toggle
theme = st.sidebar.radio("Choose Theme", ["🌙 Dark Mode", "☀️ Light Mode"])
if theme == "🌙 Dark Mode":
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body { background-color: white; color: black; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# App title
st.title("DiabeAI - Smart Diabetes Prediction")

# Sidebar info
st.sidebar.title("About DiabeAI")
st.sidebar.info(
    "DiabeAI uses ensemble machine learning to predict diabetes risk. "
    "Not a medical diagnosis — consult a professional!"
)

# User inputs
st.subheader("Enter Your Health Info:")
pregnancies = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose Level", 0, 200, 100)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 22.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 1, 120, 25)

if st.button("Predict Diabetes Risk"):
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    if prediction[0] == 1:
        st.error(f"High risk of diabetes detected! ({probability:.2%} probability)")
    else:
        st.success(f"Low risk of diabetes. ({probability:.2%} probability)")
