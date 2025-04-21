# --- Imports ---
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram

# --- Page Setup ---
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("Customer Segmentation Analysis Dashboard")

# --- Load Models and Data ---
@st.cache_data
def load_artifacts():
    return (
        joblib.load("best_rf.pkl"),
        joblib.load("X_train.pkl"),
        joblib.load("cluster_k_info.pkl"),
        pd.read_csv("clustering_results.csv")
    )

model, X_train, cluster_k_info, df = load_artifacts()

# --- Sidebar Navigation ---
section = st.sidebar.radio("Navigation", ["Cluster Analysis", "Analyze New Customer", "Custom Clustering"])

# --- Utility: Cluster Naming Based on Traits ---
def get_cluster_name(cluster_df):
    mean_age = cluster_df['Age_original'].mean()
    mean_income = cluster_df['Annual_Income (£K)_original'].mean()
    mean_spending = cluster_df['Spending_Score_original'].mean()
    age_group = "Young" if mean_age < 30 else "Middle Age" if mean_age < 55 else "Senior"
    income_group = "High Income" if mean_income > 70 else "Low Income"
    spender_type = "High Spenders" if mean_spending > 60 else "Low Spenders"
    return f"{age_group} {income_group} {spender_type}"

# --- Section 1: Cluster Analysis ---
if section == "Cluster Analysis":
    st.header("Cluster Analysis")
    clustering_method = st.sidebar.selectbox("Choose Method", ["K-Means", "GMM", "Agglomerative", "DBSCAN"])

    # Filters
    age_range = st.sidebar.slider("Filter by Age", 15, 100, (25, 60))
    income_range = st.sidebar.slider("Filter by Income", 0, 200, (20, 100))
    df_filtered = df[(df['Age_original'].between(*age_range)) &
                     (df['Annual_Income (£K)_original'].between(*income_range))]

    # Apply clustering
    def apply_cluster(method):
        features = df_filtered[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]
        if method == "K-Means":
            return df_filtered['Cluster_k']
        elif method == "GMM":
            return df_filtered['Cluster_gmm']
        elif method == "Agglomerative":
            return AgglomerativeClustering(n_clusters=5).fit_predict(features)
        elif method == "DBSCAN":
            return DBSCAN(eps=10, min_samples=5).fit_predict(features)

    df_filtered['Active_Cluster'] = apply_cluster(clustering_method)

    # Show Cluster Names
    cluster_names = {}
    for label in sorted(df_filtered['Active_Cluster'].unique()):
        if label == -1:
            cluster_names[label] = "Noise"
        else:
            cluster_data = df_filtered[df_filtered['Active_Cluster'] == label]
            cluster_names[label] = get_cluster_name(cluster_data)

    df_filtered['Cluster_Name'] = df_filtered['Active_Cluster'].map(cluster_names)

    # Clustering Metrics
    st.markdown("### Clustering Quality Metrics")
    valid = df_filtered['Active_Cluster'] != -1
    if df_filtered[valid]['Active_Cluster'].nunique() > 1:
        X_valid = df_filtered[valid][['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]
        labels = df_filtered[valid]['Active_Cluster']
        st.write(f"**Silhouette Score:** {silhouette_score(X_valid, labels):.2f}")
        st.write(f"**Davies-Bouldin Index:** {davies_bouldin_score(X_valid, labels):.2f}")
        st.write(f"**Calinski-Harabasz Score:** {calinski_harabasz_score(X_valid, labels):.2f}")
    else:
        st.warning("Not enough clusters for metric evaluation.")

    # Explanation Toggle
    if st.sidebar.checkbox("Show Explanations"):
        with st.expander("Metric Explanations"):
            st.markdown("- **Silhouette Score**: Measures how similar an object is to its own cluster vs others.")
            st.markdown("- **Davies-Bouldin Index**: Measures cluster separation and compactness. Lower is better.")
            st.markdown("- **Calinski-Harabasz Score**: Ratio of dispersion between and within clusters. Higher is better.")

    # 3D Plot
    st.markdown("### 3D Cluster Visualization")
    x_feat = st.selectbox("X-axis", df_filtered.columns, index=0)
    y_feat = st.selectbox("Y-axis", df_filtered.columns, index=1)
    z_feat = st.selectbox("Z-axis", df_filtered.columns, index=2)
    color_theme = st.selectbox("Color Scheme", ["Viridis", "Plasma", "Cividis", "Turbo"])

    fig = px.scatter_3d(df_filtered, x=x_feat, y=y_feat, z=z_feat,
                        color='Cluster_Name', color_continuous_scale=color_theme,
                        title="Interactive Cluster Plot")
    st.plotly_chart(fig, use_container_width=True)

# --- Section 2: Analyze New Customer ---
elif section == "Analyze New Customer":
    st.header("Analyze New Customer")
    with st.form("new_customer"):
        age = st.number_input("Age", 18, 100, 32)
        income = st.number_input("Annual Income (£K)", 0, 200, 70)
        spending = st.number_input("Spending Score", 0, 100, 85)
        gender = st.radio("Gender", ["Female", "Male"])
        submitted = st.form_submit_button("Predict Cluster")

    if submitted:
        new_data = pd.DataFrame([{
            'Age_original': age,
            'Annual_Income (£K)_original': income,
            'Spending_Score_original': spending,
            'Gender_Female': 1 if gender == 'Female' else 0,
            'Gender_Male': 1 if gender == 'Male' else 0
        }])

        cluster = model.predict(new_data[X_train.columns])[0]
        st.success(f"Predicted Cluster: {cluster}")

        similar = cluster_k_info[cluster]
        similarity = cosine_similarity(similar[X_train.columns], new_data[X_train.columns])
        idx = similarity.argmax()

        st.subheader("Most Similar Existing Customer")
        st.dataframe(similar.iloc[idx][X_train.columns])

        # Radar Chart
        trace1 = go.Scatterpolar(r=similar[X_train.columns].mean(), theta=X_train.columns, name="Cluster Mean")
        trace2 = go.Scatterpolar(r=new_data.iloc[0], theta=X_train.columns, name="New Customer")
        radar = go.Figure(data=[trace1, trace2])
        radar.update_layout(title="Customer vs Cluster Comparison", polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(radar)

# --- Section 3: Custom Clustering ---
elif section == "Custom Clustering":
    st.header("Custom Clustering")
    algo = st.selectbox("Select Algorithm", ["K-Means", "GMM", "Agglomerative", "DBSCAN"])
    if algo in ["K-Means", "GMM", "Agglomerative"]:
        k = st.slider("Number of Clusters", 2, 10, 4)
    if algo == "DBSCAN":
        eps = st.slider("Epsilon (eps)", 1.0, 20.0, 10.0)
        min_samples = st.slider("Minimum Samples", 1, 10, 5)

    data = df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]

    if st.button("Run Clustering"):
        if algo == "K-Means":
            labels = KMeans(n_clusters=k).fit_predict(data)
        elif algo == "GMM":
            labels = GaussianMixture(n_components=k).fit_predict(data)
        elif algo == "Agglomerative":
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(data)
        elif algo == "DBSCAN":
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)

        df['Custom_Cluster'] = labels
        fig = px.scatter_3d(df, x='Age_original', y='Annual_Income (£K)_original', z='Spending_Score_original',
                            color=df['Custom_Cluster'].astype(str), title="Custom Clustering 3D Visualization")
        st.plotly_chart(fig)
