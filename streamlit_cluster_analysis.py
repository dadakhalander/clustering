import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.datasets import make_blobs
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# ---- Load Pretrained Artifacts ----
model = joblib.load("best_rf.pkl")
X_train = joblib.load("X_train.pkl")
cluster_k_info = joblib.load("cluster_k_info.pkl")

# ---- Elbow Method Function ----
def elbow_method(X, max_k=10):
    wcss = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss

# ---- Silhouette Score Function ----
def silhouette_method(X, max_k=10):
    silhouette_avg = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg.append(silhouette_score(X, cluster_labels))
    return silhouette_avg

# ---- Gap Statistic Function ----
def gap_statistic(X, max_k=10, n_ref=10):
    gaps = []
    for k in range(1, max_k+1):
        # Perform K-means on original data
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        W_k = kmeans.inertia_
        
        # Create random reference datasets
        gaps_k = []
        for _ in range(n_ref):
            random_data = np.random.uniform(low=X.min(axis=0), high=X.max(axis=0), size=X.shape)
            kmeans_ref = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
            kmeans_ref.fit(random_data)
            W_k_ref = kmeans_ref.inertia_
            gaps_k.append(np.log(W_k_ref) - np.log(W_k))
        
        gaps.append(np.mean(gaps_k))
    return gaps

# ---- Function to Apply K-Means and Display Clusters ----
def apply_kmeans(X, k):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_

# ---- Streamlit App UI ----
st.set_page_config(page_title="Customer Cluster Dashboard", layout="wide")
st.title("Customer Segmentation Analysis Dashboard")

# ---- Sidebar Navigation ----
section = st.sidebar.radio("Choose Section", ["Cluster Analysis", "Analyze New Customer Data", "Auto-ML for Optimal Clusters"])

if section == "Auto-ML for Optimal Clusters":
    st.header("Auto-ML for Optimal Cluster Count")

    # Load the dataset
    @st.cache
    def load_data():
        return pd.read_csv("clustering_results.csv")

    df = load_data()
    
    # Extract relevant features
    X = df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]
    X_scaled = StandardScaler().fit_transform(X)
    
    st.markdown("### Explore Optimal K using different methods")
    max_k = st.slider("Max K to Explore", 2, 15, 10)

    # Elbow Method Visualization
    wcss = elbow_method(X_scaled, max_k)
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(range(1, max_k+1)), y=wcss, mode='lines+markers', name='WCSS'))
    fig_elbow.update_layout(
        title="Elbow Method - K vs WCSS",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="WCSS (Within-Cluster Sum of Squares)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_elbow)

    # Silhouette Score Visualization
    silhouette_avg = silhouette_method(X_scaled, max_k)
    fig_silhouette = go.Figure()
    fig_silhouette.add_trace(go.Scatter(x=list(range(2, max_k+1)), y=silhouette_avg, mode='lines+markers', name='Silhouette Score'))
    fig_silhouette.update_layout(
        title="Silhouette Score - K vs Average Silhouette",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score",
        template="plotly_dark"
    )
    st.plotly_chart(fig_silhouette)

    # Gap Statistic Visualization
    gaps = gap_statistic(X_scaled, max_k)
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(x=list(range(1, max_k+1)), y=gaps, mode='lines+markers', name='Gap Statistic'))
    fig_gap.update_layout(
        title="Gap Statistic - K vs Gap Value",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Gap Statistic",
        template="plotly_dark"
    )
    st.plotly_chart(fig_gap)

    # Let the user select the optimal K
    k_selected = st.selectbox("Select Optimal K based on the methods above", list(range(2, max_k+1)))

    st.subheader(f"Clustering Results for K={k_selected}")
    cluster_labels = apply_kmeans(X_scaled, k_selected)
    df['Cluster'] = cluster_labels
    st.dataframe(df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original', 'Cluster']].head())

    # Visualize clusters
    fig_cluster = go.Figure()
    fig_cluster.add_trace(go.Scatter(x=df['Age_original'], y=df['Annual_Income (£K)_original'], mode='markers', 
                                     marker=dict(color=df['Cluster'], colorscale='Viridis', size=10)))
    fig_cluster.update_layout(
        title=f"Clusters Visualized for K={k_selected}",
        xaxis_title="Age",
        yaxis_title="Annual Income (£K)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_cluster)

elif section == "Cluster Analysis":
    st.header("Cluster Analysis with Existing Data")
    # Add code for cluster analysis from previous steps

elif section == "Analyze New Customer Data":
    st.header("Analyze New Customer Data")
    # Add code for analyzing new customer data
