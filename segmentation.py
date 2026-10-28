import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Customer Segmentation App")
st.write("Enter customer details to predict their segment.")


age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (in $)", min_value=0, max_value=200000, value=50000)
total_spent = st.number_input("Total Spending (in $)", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Number of Web Visits", min_value=0, max_value=50, value=20)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30) 


input_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'Total_Spending': [total_spent],
    'NumWebPurchases': [num_web_purchases],
    'NumStorePurchases': [num_store_purchases],
    'NumWebVisitsMonth': [num_web_visits],
    'Recency': [recency]
})

input_data_scaled = scaler.transform(input_data)

if st.button("Predict Customer Segment"):
    cluster = kmeans.predict(input_data_scaled)[0]
    cluster_names = {
        0: "High-Value Active Multichannel Customers",
        1: "Low-Value Inactive / At-Risk Customers",
        2: "High-Income / VIP Customers",
        3: "Low-Value Recently Active Customers"
    }
    segment_name = cluster_names[cluster]
    st.success(f"The predicted customer segment is: **{segment_name}**")