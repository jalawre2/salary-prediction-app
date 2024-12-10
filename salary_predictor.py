# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pickle
import numpy as np

# Load the trained model, scaler, and polynomial features from the .pkl file
model_path = r"C:\Users\timeb\OneDrive\mldeploy\salary_model.pkl"
with open(model_path, "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
scaler = model_data["scaler"]
poly = model_data["poly"]

# App title and instructions
st.markdown("<h1 style='text-align: center; color: red; font-size: 36px; text-decoration: underline;'>Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Please answer all of the questions below, and click predict salary</h3>", unsafe_allow_html=True)

# Display the image
image_path = r"C:\Users\timeb\Downloads\download (1).jpg"
st.image(image_path, caption="", use_column_width="auto", output_format="auto")

# Input fields
st.markdown("### 1. Do you currently reside in the United States?")
country = st.selectbox("Select your answer:", ["Yes", "No"])
country_encoded = 1 if country == "Yes" else 0

st.markdown("### 2. How many years of Machine Learning have you done?")
yrs_ml = st.slider("Drag to select the number of years:", min_value=0, max_value=20, value=0)

st.markdown("### 3. Approximately how much money have you spent on Machine Learning?")
money_spent_ml_input = st.text_input("Enter the amount in dollars (e.g., 1000):", value="")
try:
    money_spent_ml = float(money_spent_ml_input.replace("$", "")) if money_spent_ml_input else None
except ValueError:
    money_spent_ml = None

st.markdown("### 4. How many years of coding experience do you have?")
years_coding_input = st.text_input("Enter the number of years:", value="")
try:
    years_coding = float(years_coding_input) if years_coding_input else None
except ValueError:
    years_coding = None

st.markdown("### 5. What is your age?")
age = st.slider("Drag to select your age:", min_value=18, max_value=70, value=18)

# Prediction button
if st.button("Predict Salary"):
    if money_spent_ml is None or years_coding is None:
        st.error("All questions must be answered.")
    else:
        # Prepare input features
        features = np.array([[
            country_encoded, 
            min(yrs_ml, 20),  # Cap the input to 20 if above
            money_spent_ml, 
            years_coding, 
            min(age, 70)      # Cap the input to 70 if above
        ]])
        
        # Apply polynomial transformation and scaling
        features_poly = poly.transform(features)
        features_scaled = scaler.transform(features_poly)
        
        # Predict salary
        prediction = model.predict(features_scaled)[0]
        
        # Display the prediction
        st.success(f"Your predicted salary is: ${prediction:,.2f}")
