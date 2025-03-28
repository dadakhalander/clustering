import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (Modify to your actual dataset loading method)
@st.cache_data
def load_data():
    return pd.read_csv("clustering_results.csv")  # Replace with actual file path

df = load_data()

# Streamlit App Title
st.title("📊 Cluster Analysis: Spending & Gender Distribution")

# Select cluster type (GMM or K-Means)
cluster_type = st.sidebar.selectbox("Select Cluster Type", ["Cluster_gmm", "Cluster_k"])

# Get unique clusters
clusters = sorted(df[cluster_type].unique())

# Iterate through clusters and plot
for cluster_label in clusters:
    cluster_data = df[df[cluster_type] == cluster_label]

    # Create Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Spending Score Histogram
    axes[0].hist(cluster_data['Spending_Score_original'], bins=10, edgecolor='black')
    axes[0].set_xlabel("Spending Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Spending Score Distribution - {cluster_type} {cluster_label}")

    # Gender Pie Chart
    gender_counts = cluster_data["Gender_Male"].value_counts()
    labels = ["Male", "Female"]
    colors = ["skyblue", "lightcoral"]
    axes[1].pie(gender_counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    axes[1].set_title(f"Gender Distribution - {cluster_type} {cluster_label}")
    axes[1].axis("equal")  # Ensures pie chart is circular

    # Display plots in Streamlit
    st.pyplot(fig)
