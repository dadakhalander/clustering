import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np

st.set_page_config(page_title="Customer Clustering Dashboard", layout="wide")

st.title("🧠 Customer Clustering Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df.head())

    with st.expander("🧠 Show Clustering Concepts Explanation"):
        st.markdown("""
        **Silhouette Score** measures how well data points fit into their clusters (ranges from -1 to 1).
        
        **DBSCAN** identifies clusters based on density. It groups closely packed points and labels outliers as noise.
        """)

    # Feature Selection
    st.sidebar.subheader("Feature Selection")
    features = st.sidebar.multiselect("Select features for clustering", df.select_dtypes(include=np.number).columns.tolist(), default=df.select_dtypes(include=np.number).columns.tolist()[:3])

    if len(features) >= 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])

        # KMeans Clustering
        k = st.sidebar.slider("Select number of clusters (KMeans)", 2, 10, 4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans.fit_predict(scaled_data)
        df['KMeans_Cluster'] = kmeans_labels
        silhouette_kmeans = silhouette_score(scaled_data, kmeans_labels)

        # DBSCAN Clustering
        eps = st.sidebar.slider("DBSCAN - Epsilon", 0.1, 5.0, 0.5)
        min_samples = st.sidebar.slider("DBSCAN - Min Samples", 2, 10, 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        db_labels = dbscan.fit_predict(scaled_data)
        df['DBSCAN_Cluster'] = db_labels

        # Silhouette Score for DBSCAN
        try:
            silhouette_db = silhouette_score(scaled_data, db_labels)
        except:
            silhouette_db = "Not applicable (only one cluster or all noise)"

        st.subheader("📊 Clustering Evaluation")
        st.write(f"**KMeans Silhouette Score:** {silhouette_kmeans:.3f}")
        st.write(f"**DBSCAN Silhouette Score:** {silhouette_db}")

        # Cluster Naming Based on Traits
        cluster_profiles = df.groupby('KMeans_Cluster')[features].mean().round(1)

        def assign_cluster_names(row):
            if row['KMeans_Cluster'] == cluster_profiles[features[2]].idxmax():
                return "High Spenders"
            elif row['KMeans_Cluster'] == cluster_profiles[features[0]].idxmin():
                return "Young Customers"
            elif row['KMeans_Cluster'] == cluster_profiles[features[1]].idxmin():
                return "Low Income Group"
            else:
                return "Moderate Group"

        df['Custom_Cluster'] = df.apply(assign_cluster_names, axis=1)

        st.subheader("📌 Cluster Summary")
        st.dataframe(cluster_profiles)

        # 3D Scatter Plot
        st.subheader("🎯 3D Cluster Visualization")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("X Axis", features, index=0)
        with col2:
            y_axis = st.selectbox("Y Axis", features, index=1)
        with col3:
            z_axis = st.selectbox("Z Axis", features, index=2)

        color_scheme = st.selectbox("🎨 Select Color Scheme", ["Viridis", "Cividis", "Plasma", "Turbo"], index=0)

        fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis,
                            color='Custom_Cluster', color_continuous_scale=color_scheme,
                            symbol='Custom_Cluster', opacity=0.8)
        st.plotly_chart(fig, use_container_width=True)

        # Optional spending score deviation
        if features[2]:
            st.subheader("📉 Cluster Spending Score Deviation")
            deviation = df.groupby('Custom_Cluster')[features[2]].std().round(2)
            st.bar_chart(deviation)
    else:
        st.warning("Please select at least 2 features for clustering.")
