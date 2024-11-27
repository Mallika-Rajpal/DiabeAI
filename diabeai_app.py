import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Page configuration
st.set_page_config(
    page_title="DiabeAI - Diabetes Prediction",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Apply a pastel-themed background and style
page_bg = """
<style>
body {
    background-color: #f7f3e9; /* Pastel cream */
    color: #3c3b3d; /* Dark gray */
    font-family: 'Arial', sans-serif;
}
.sidebar .sidebar-content {
    background-color: #e8f5e9; /* Pastel green */
}
div.stButton > button:first-child {
    background-color: #ffc5d9; /* Pastel pink */
    color: #ffffff; /* White text */
    border-radius: 10px;
    height: 50px;
    width: 100%;
    border: none;
    font-size: 18px;
}
div.stButton > button:hover {
    background-color: #ffa5c2; /* Slightly darker pink */
    color: #ffffff;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# App title
st.title("DiabeAI - Diabetes Prediction")

# Sidebar
st.sidebar.header("Enter Patient Details")
st.sidebar.write("Provide the following health metrics for prediction:")

def user_input_features():
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.sidebar.slider("Glucose Level", 0, 200, 100)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 140, 80)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin Level", 0, 900, 30)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 20.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 0, 120, 25)
    features = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    return np.array(list(features.values())).reshape(1, -1)

input_data = user_input_features()

# Prediction button
if st.button("Predict Diabetes Risk"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error(
            """
            High Risk Detected!  
            The model predicts a **HIGH RISK** of diabetes.  
            Please consult a healthcare professional.
            """
        )
    else:
        st.success(
            """
            Low Risk Detected!  
            The model predicts a **LOW RISK** of diabetes.  
            Keep maintaining a healthy lifestyle!
            """
        )

# Footer
st.write(" ")
st.markdown(
    """
    <div style="text-align: center; font-size: 14px; color: #8c8c8c;">
        Designed with care by Mallika Rajpal | Powered by Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)


