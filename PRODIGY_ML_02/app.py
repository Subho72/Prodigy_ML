import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("Mall_Customers.csv")

# Load Model
kmeans = joblib.load("kmeans_model.pkl")

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️")

st.title("🛍️ Customer Segmentation using K-Means Clustering")

st.write("Group customers based on Annual Income and Spending Score.")

# Scatter Plot
fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=kmeans.labels_,
    cmap='viridis'
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segments")

st.pyplot(fig)

st.subheader("Predict Customer Segment")

income = st.slider(
    "Annual Income (k$)",
    min_value=10,
    max_value=150,
    value=50
)

score = st.slider(
    "Spending Score",
    min_value=1,
    max_value=100,
    value=50
)

if st.button("Predict Segment"):

    cluster = kmeans.predict([[income, score]])[0]

    st.success(f"Customer belongs to Segment {cluster}")