import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Generate some sample data for the clustering app
def generate_data():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    return pd.DataFrame(X, columns=["Feature_1", "Feature_2"])

# Load or generate dataset
df = generate_data()

# Preprocessing: Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Streamlit UI
st.title("Clustering Analysis App")

# Sidebar for clustering
section = st.sidebar.selectbox("Choose Section", ["Clustering Quality Metric Comparison", "Custom Clustering"])

# Clustering Quality Metric Comparison
if section == "Clustering Quality Metric Comparison":
    st.header("Clustering Quality Metric Comparison")

    # KMeans Clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(df_scaled)

    # Agglomerative Clustering
    agg_clust = AgglomerativeClustering(n_clusters=4)
    agg_labels = agg_clust.fit_predict(df_scaled)

    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm_labels = gmm.fit_predict(df_scaled)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(df_scaled)

    # Create a DataFrame to compare clustering results
    df_comparison = pd.DataFrame({
        'KMeans': kmeans_labels,
        'Agglomerative': agg_labels,
        'Gaussian Mixture': gmm_labels,
        'DBSCAN': dbscan_labels
    })

    st.write("Comparison of Cluster Labels from Different Algorithms:")
    st.dataframe(df_comparison.head())

    # Visualizing Clusters using PCA for 2D projection
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    ax[0, 0].scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_labels, cmap='viridis')
    ax[0, 0].set_title("KMeans Clustering")

    ax[0, 1].scatter(df_pca[:, 0], df_pca[:, 1], c=agg_labels, cmap='viridis')
    ax[0, 1].set_title("Agglomerative Clustering")

    ax[1, 0].scatter(df_pca[:, 0], df_pca[:, 1], c=gmm_labels, cmap='viridis')
    ax[1, 0].set_title("Gaussian Mixture Clustering")

    ax[1, 1].scatter(df_pca[:, 0], df_pca[:, 1], c=dbscan_labels, cmap='viridis')
    ax[1, 1].set_title("DBSCAN Clustering")

    plt.tight_layout()
    st.pyplot(fig)

# Custom Clustering Functionality
elif section == "Custom Clustering":
    st.header("Custom Clustering Functionality")

    # Select clustering method
    clustering_method = st.selectbox("Choose Clustering Algorithm", 
                                    ["K-Means", "Agglomerative Clustering", "Gaussian Mixture", "DBSCAN"])

    # Custom number of clusters for KMeans and Agglomerative
    n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=10, value=4, step=1)

    # For DBSCAN, allow input for epsilon and min_samples
    if clustering_method == "DBSCAN":
        eps = st.number_input("Epsilon (eps)", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
        min_samples = st.number_input("Minimum Samples (min_samples)", min_value=1, max_value=10, value=5, step=1)

    # Choose features for clustering
    features = st.multiselect("Select Features for Clustering", options=df.columns.tolist(), 
                              default=["Feature_1", "Feature_2"])

    # Perform clustering on button click
    perform_clustering = st.button("Perform Clustering")

    if perform_clustering:
        if clustering_method == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(df[features])
            st.write(f"Cluster centers: {model.cluster_centers_}")

        elif clustering_method == "Agglomerative Clustering":
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(df[features])
            st.write(f"Linkage matrix: {model.children_}")

        elif clustering_method == "Gaussian Mixture":
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = model.fit_predict(df[features])
            st.write(f"Means of Gaussian Components: {model.means_}")

        elif clustering_method == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(df[features])

        # Add the cluster labels to the DataFrame
        df['Custom_Cluster'] = labels

        # Visualize clusters
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df[features[0]], df[features[1]], c=labels, cmap="viridis")
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title(f"{clustering_method} Clustering")

        # Add color bar
        fig.colorbar(scatter)
        st.pyplot(fig)

        # Show cluster sizes
        cluster_sizes = pd.Series(labels).value_counts()
        st.write(f"Cluster Sizes: {cluster_sizes}")

        if clustering_method == "K-Means":
            st.write(f"Cluster Centers (K-Means): {model.cluster_centers_}")
        elif clustering_method == "Gaussian Mixture":
            st.write(f"Gaussian Component Means: {model.means_}")
