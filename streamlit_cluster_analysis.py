import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🧠 Customer Segmentation Dashboard")
st.markdown("Use this interactive dashboard to explore clustering models for customer segmentation.")

# Upload data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Show data
    with st.expander("📊 View Raw Data"):
        st.dataframe(df)

    # Feature selection
    st.sidebar.header("🔧 Feature Selection")
    features = st.sidebar.multiselect("Select features for clustering", df.select_dtypes(include=[np.number]).columns.tolist(), default=["Age", "Annual Income (k$)", "Spending Score (1-100)"])

    if len(features) >= 2:
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Model selection
        model_type = st.sidebar.selectbox("Select clustering model", ["KMeans", "DBSCAN"])

        if model_type == "KMeans":
            k = st.sidebar.slider("Select number of clusters (K)", 2, 10, 3)
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)

            if st.toggle("Show Explanation for Silhouette Score"):
                st.info("""
                The Silhouette Score measures how well each data point fits within its cluster compared to other clusters.
                - Ranges from -1 to 1
                - Higher values indicate better defined clusters
                """)

            st.success(f"Silhouette Score: {score:.2f}")

        elif model_type == "DBSCAN":
            eps = st.sidebar.slider("Select epsilon (eps)", 0.1, 5.0, 0.5)
            min_samples = st.sidebar.slider("Minimum samples", 1, 20, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                st.success(f"Silhouette Score: {score:.2f}")
            else:
                st.warning("DBSCAN found only 1 cluster. Adjust parameters.")

            if st.toggle("Show Explanation for DBSCAN"):
                st.info("""
                DBSCAN clusters based on density:
                - eps: Radius of neighborhood
                - min_samples: Minimum points to form a dense region
                - Label -1 means noise/outlier
                """)

        # Attach cluster labels
        df["Cluster"] = labels

        # Automatically assign descriptive names based on profile averages
        cluster_profiles = df.groupby("Cluster")[features].mean()

        # Define labeling logic
        def generate_cluster_names(profile_df):
            cluster_names = {}
            for cluster_id, row in profile_df.iterrows():
                traits = []
                if "Age" in row:
                    if row["Age"] < 30:
                        traits.append("Young")
                    elif row["Age"] < 50:
                        traits.append("Middle Aged")
                    else:
                        traits.append("Senior")
                if "Annual Income (k$)" in row:
                    if row["Annual Income (k$)"] < 40:
                        traits.append("Low Income")
                    elif row["Annual Income (k$)"] < 80:
                        traits.append("Mid Income")
                    else:
                        traits.append("High Income")
                if "Spending Score (1-100)" in row:
                    if row["Spending Score (1-100)"] < 40:
                        traits.append("Low Spender")
                    elif row["Spending Score (1-100)"] < 70:
                        traits.append("Moderate Spender")
                    else:
                        traits.append("High Spender")
                cluster_names[cluster_id] = " ".join(traits)
            return cluster_names

        cluster_name_map = generate_cluster_names(cluster_profiles)
        df["Custom_Cluster"] = df["Cluster"].map(cluster_name_map)

        st.markdown("### 🧬 Cluster Profile Summary")
        st.dataframe(cluster_profiles)

        # Show deviations of spending score per cluster if available
        if "Spending Score (1-100)" in df.columns:
            st.markdown("#### 📈 Spending Score Deviation per Cluster")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="Custom_Cluster", y="Spending Score (1-100)", palette="viridis", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Dynamic 3D plot
        st.markdown("### 🔍 3D Cluster Visualization")
        available_features = df.select_dtypes(include=[np.number]).columns.tolist()
        x_axis = st.selectbox("X-axis", available_features, index=0)
        y_axis = st.selectbox("Y-axis", available_features, index=1)
        z_axis = st.selectbox("Z-axis", available_features, index=2)

        color_scheme = st.selectbox("Choose color scheme", ["Plotly", "Viridis", "Cividis", "Turbo"], index=0)

        fig3d = px.scatter_3d(
            df,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color="Custom_Cluster",
            color_continuous_scale=color_scheme.lower(),
            opacity=0.8
        )
        st.plotly_chart(fig3d, use_container_width=True)

        # Optional: Download results
        with st.expander("⬇️ Download clustered data"):
            st.download_button("Download as CSV", df.to_csv(index=False), "clustered_data.csv")

    else:
        st.warning("Please select at least 2 numerical features for clustering.")
else:
    st.info("Upload a CSV file to begin analysis.")
