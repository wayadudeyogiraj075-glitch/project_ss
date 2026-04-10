import streamlit as st
import pickle
import numpy as np

# Load the trained model
def load_model():
    with open('model (1).pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()

# Streamlit App UI
st.title("Diabetes Prediction App")
st.write("Enter the following health metrics to predict the likelihood of diabetes.")

# Create input fields based on model features
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=0, step=1)

# Prediction button
if st.button("Predict"):
    # Arrange inputs as a numpy array for the model
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, dpf, age]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Display result
    if prediction[0] == 1:
        st.error("The model predicts a high likelihood of Diabetes.")
    else:
        st.success("The model predicts a low likelihood of Diabetes.")

st.info("Note: This is a machine learning demonstration and should not be used for medical diagnosis.")
