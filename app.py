import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Glucose Level Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Glucose Level Predictor")
st.write("Uses a **Decision Tree Regressor** to predict a patient's continuous Glucose Level.")

MODEL_PATH = "models/dt_regressor.pkl"
FEATURES_PATH = "models/feature_names.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Model not found. Please run the Jupyter notebook first.")
    st.stop()

model, feature_names = load_model()

# Sidebar: dataset preview
st.sidebar.header("📂 Dataset Preview")
if os.path.exists("data/diabetes_regression.csv"):
    df = pd.read_csv("data/diabetes_regression.csv")
    st.sidebar.write(f"Shape: {df.shape}")
    st.sidebar.dataframe(df.head(5))

# Input form
st.subheader("🔢 Enter Patient Details")
col1, col2 = st.columns(2)

with col1:
    pregnancies   = st.number_input("Pregnancies", 0, 20, 2)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 150, 70)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    insulin       = st.number_input("Insulin (mu U/ml)", 0, 900, 80)

with col2:
    bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, 28.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.001)
    age = st.number_input("Age (years)", 1, 120, 35)

# Predict
if st.button("🩺 Predict Glucose Level"):
    input_data = np.array([[pregnancies, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]

    st.divider()
    st.metric(label="Predicted Glucose Level", value=f"{prediction:.2f} mg/dL")

    if prediction >= 126:
        st.error("⚠️ High glucose — may indicate diabetes risk.")
    elif prediction >= 100:
        st.warning("🟡 Borderline glucose — pre-diabetic range.")
    else:
        st.success("✅ Normal glucose range.")

    st.caption("⚠️ Educational tool only. Not a substitute for medical advice.")
