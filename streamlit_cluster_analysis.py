import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

# Load model and cluster info
model = joblib.load('best_rf.pkl')  # Replace with your model path
cluster_k_info = joblib.load('cluster_k_info.pkl')  # Contains {cluster_label: 'description'}

# Load training features
X_train = joblib.load('X_train.pkl')

# Load clustering dataset
@st.cache_data
def load_data():
    df = pd.read_csv('clustering_results.csv')
    return df

df = load_data()

# Predict cluster labels if missing
if 'Cluster_Label' not in df.columns:
    try:
        df['Cluster_Label'] = model.predict(df[X_train.columns])
    except Exception as e:
        st.error("Failed to add Cluster_Label: " + str(e))

# Function to display cluster insights
def cluster_analysis(df):
    st.header("🔍 Cluster Analysis")
    cluster_counts = df['Cluster_Label'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    st.subheader("Cluster Description Summary")
    for cluster, description in cluster_k_info.items():
        st.markdown(f"**Cluster {cluster}:** {description}")

    st.subheader("📊 Cluster Feature Means")
    cluster_means = df.groupby('Cluster_Label')[X_train.columns].mean()
    st.dataframe(cluster_means.style.format("{:.2f}"))

    st.subheader("📈 Feature Distribution by Cluster")
    feature = st.selectbox("Choose feature", X_train.columns)
    fig = px.box(df, x='Cluster_Label', y=feature, points='all', title=f"{feature} by Cluster")
    st.plotly_chart(fig)

# Function to analyze new customer
def analyze_new_customer(new_customer, model, X_train, cluster_k_info):
    st.subheader("🧪 New Customer Cluster Prediction")

    # Ensure only model features are used
    try:
        new_customer_filtered = new_customer[X_train.columns]
    except KeyError as e:
        st.error(f"Missing columns in new data: {e}")
        return

    cluster_label = model.predict(new_customer_filtered)[0]
    st.markdown(f"### This new customer belongs to **Cluster {cluster_label}**.")
    st.markdown(f"**Description:** {cluster_k_info.get(cluster_label, 'No description available')}")

    st.markdown("#### 📋 Customer Feature Values:")
    st.dataframe(new_customer)

# Streamlit layout
st.title("📈 E-Commerce Customer Cluster Dashboard")
menu = st.sidebar.radio("Navigation", ["Cluster Analysis", "Analyze New Customer"])

if menu == "Cluster Analysis":
    cluster_analysis(df)

elif menu == "Analyze New Customer":
    st.header("🔍 Analyze New Customer Data")
    
    st.markdown("Upload a CSV file containing **one row** of new customer data with the same features as model training data.")
    uploaded_file = st.file_uploader("Upload Customer CSV", type=['csv'])

    if uploaded_file:
        try:
            new_data = pd.read_csv(uploaded_file)
            if new_data.shape[0] != 1:
                st.error("Please upload a file with exactly one customer (one row).")
            else:
                analyze_new_customer(new_data, model, X_train, cluster_k_info)
        except Exception as e:
            st.error("Error processing file: " + str(e))
