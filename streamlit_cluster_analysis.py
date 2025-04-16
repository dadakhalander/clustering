import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import plotly.express as px
import plotly.graph_objects as go
import io

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("E-Commerce_Data_Set_4034.csv")
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
selected_gender = st.sidebar.multiselect("Select Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
filtered_data = df[df["Gender"].isin(selected_gender)]

# Features for clustering
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
data = filtered_data[features]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Auto-ML for optimal K
st.subheader(" Optimal Cluster Count Detection")

elbow_wcss = []
silhouette_scores = []
k_values = list(range(2, 11))

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    elbow_wcss.append(kmeans.inertia_)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(data_scaled, labels))

fig_elbow = go.Figure()
fig_elbow.add_trace(go.Scatter(x=k_values, y=elbow_wcss, mode='lines+markers', name='WCSS'))
fig_elbow.update_layout(title='Elbow Method', xaxis_title='Number of clusters (k)', yaxis_title='WCSS')
st.plotly_chart(fig_elbow)

fig_silhouette = go.Figure()
fig_silhouette.add_trace(go.Scatter(x=k_values, y=silhouette_scores, mode='lines+markers', name='Silhouette Score'))
fig_silhouette.update_layout(title='Silhouette Scores vs. k', xaxis_title='Number of clusters (k)', yaxis_title='Silhouette Score')
st.plotly_chart(fig_silhouette)

# Gap Statistic (optional for simplicity)
# Due to complexity, not implemented here directly – can be added with gap-stat module if required

# Choose number of clusters
st.sidebar.subheader("Model Parameters")
k = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=4)
model_type = st.sidebar.selectbox("Select Clustering Model", ("KMeans", "GMM"))

if model_type == "KMeans":
    model = KMeans(n_clusters=k, random_state=42)
    cluster_labels = model.fit_predict(data_scaled)
else:
    model = GaussianMixture(n_components=k, random_state=42)
    cluster_labels = model.fit_predict(data_scaled)

# Add cluster labels to data
data_with_labels = filtered_data.copy()
data_with_labels["Cluster"] = cluster_labels

# Visualizations
st.subheader(" Cluster Visualization")
fig_cluster = px.scatter_3d(data_with_labels, x="Age", y="Annual Income (k$)", z="Spending Score (1-100)",
                            color="Cluster", symbol="Gender",
                            title="Customer Clusters (3D View)",
                            width=900, height=600)
st.plotly_chart(fig_cluster)

st.subheader(" Cluster Distribution")
st.write(data_with_labels.groupby("Cluster")[features].mean())

# Downloadable output
buffer = io.BytesIO()
data_with_labels.to_csv(buffer, index=False)
st.download_button(" Download Clustered Data", data=buffer.getvalue(), file_name="clustered_customers.csv", mime="text/csv")
