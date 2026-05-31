import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("house_price_model.pkl")

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Predict House Prices using Linear Regression")

st.divider()

# Inputs
area = st.number_input(
    "Square Footage (GrLivArea)",
    min_value=100,
    max_value=10000,
    value=1500
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

if st.button("Predict Price"):

    features = np.array([[area, bedrooms, bathrooms]])

    prediction = model.predict(features)

    st.success(
        f"Estimated House Price: ${prediction[0]:,.2f}"
    )