import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram

# ---- Load Pretrained Artifacts ----
model = joblib.load("best_rf.pkl")
X_train = joblib.load("X_train.pkl")
cluster_k_info = joblib.load("cluster_k_info.pkl")

# ---- Streamlit Page Config ----
st.set_page_config(page_title="Customer Clustering Dashboard", layout="wide")
st.title("📊 Customer Segmentation & Clustering Dashboard")

# ---- Sidebar Navigation ----
section = st.sidebar.radio("Navigation", ["Cluster Analysis", "Analyze New Customer", "Custom Clustering"])

@st.cache_data

def load_data():
    return pd.read_csv("clustering_results.csv")

df = load_data()

# ---- Helper: Explain Metric Tooltips ----
def metric_tooltip(metric):
    explanations = {
        "Silhouette Score": "Measures how similar an object is to its own cluster vs other clusters. Range: -1 to 1.",
        "Davies-Bouldin Index": "Lower DB index indicates better clustering. Measures similarity within clusters.",
        "Calinski-Harabasz Score": "Higher values indicate better defined clusters. Based on between/within dispersion.",
        "DBSCAN": "Density-based clustering. Clusters dense regions and treats sparse points as noise."
    }
    return explanations.get(metric, "")

# ---- Helper: Generate Cluster Names ----
def generate_cluster_names(df, label_col):
    names = {}
    for label in sorted(df[label_col].unique()):
        group = df[df[label_col] == label]
        age_mean = group['Age_original'].mean()
        income_mean = group['Annual_Income (£K)_original'].mean()
        spending_mean = group['Spending_Score_original'].mean()

        profile = []
        if age_mean < 30:
            profile.append("Young")
        elif age_mean < 50:
            profile.append("Middle Age")
        else:
            profile.append("Senior")

        if income_mean > 70:
            profile.append("High Income")
        elif income_mean < 40:
            profile.append("Low Income")

        if spending_mean > 70:
            profile.append("High Spenders")
        elif spending_mean < 40:
            profile.append("Low Spenders")

        names[label] = " ".join(profile) or f"Cluster {label}"
    return names

# ---- Cluster Analysis Section ----
if section == "Cluster Analysis":
    st.header("🔎 Cluster Analysis")

    st.sidebar.markdown("## Filters")
    cluster_method = st.sidebar.selectbox("Select Clustering Method", ["K-Means", "GMM", "Agglomerative", "DBSCAN"])
    color_scheme = st.sidebar.selectbox("Color Scheme", ["Viridis", "Cividis", "Plasma", "Rainbow"])
    x_axis = st.sidebar.selectbox("X Axis", df.columns, index=0)
    y_axis = st.sidebar.selectbox("Y Axis", df.columns, index=1)
    z_axis = st.sidebar.selectbox("Z Axis", df.columns, index=2)

    age_range = st.sidebar.slider("Age Range", int(df['Age_original'].min()), int(df['Age_original'].max()), (20, 60))
    income_range = st.sidebar.slider("Income (£K)", int(df['Annual_Income (£K)_original'].min()), int(df['Annual_Income (£K)_original'].max()), (20, 100))

    df_filtered = df[(df['Age_original'].between(*age_range)) & (df['Annual_Income (£K)_original'].between(*income_range))]

    def apply_clustering(method, df):
        X = df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]
        if method == "K-Means":
            return df['Cluster_k'], "Cluster_k"
        elif method == "GMM":
            return df['Cluster_gmm'], "Cluster_gmm"
        elif method == "Agglomerative":
            model = AgglomerativeClustering(n_clusters=5)
            return model.fit_predict(X), "Agglomerative"
        elif method == "DBSCAN":
            model = DBSCAN(eps=10, min_samples=5)
            return model.fit_predict(X), "DBSCAN"

    labels, label_col = apply_clustering(cluster_method, df_filtered)
    df_filtered['Active_Cluster'] = labels
    cluster_names = generate_cluster_names(df_filtered, 'Active_Cluster')

    with st.expander("📌 Metric Explanations"):
        for metric in ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Score", "DBSCAN"]:
            st.markdown(f"**{metric}**: {metric_tooltip(metric)}")

    if len(set(labels)) > 1 and -1 not in set(labels):
        X_valid = df_filtered[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]
        sil = silhouette_score(X_valid, labels)
        db = davies_bouldin_score(X_valid, labels)
        ch = calinski_harabasz_score(X_valid, labels)
        st.success(f"Silhouette Score: {sil:.2f}")
        st.info(f"Davies-Bouldin Index: {db:.2f}")
        st.info(f"Calinski-Harabasz Score: {ch:.2f}")
    else:
        st.warning("Not enough distinct clusters for metric evaluation.")

    st.subheader("🧠 3D Cluster Plot")
    fig = px.scatter_3d(df_filtered, x=x_axis, y=y_axis, z=z_axis,
                        color=df_filtered['Active_Cluster'].astype(str),
                        color_continuous_scale=color_scheme,
                        title="3D Visualization of Clusters")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🪪 Cluster Names and Sizes")
    cluster_summary = df_filtered.groupby('Active_Cluster').agg({
        'Age_original': 'mean',
        'Annual_Income (£K)_original': 'mean',
        'Spending_Score_original': 'mean',
        'Gender_Male': 'count'
    }).rename(columns={'Gender_Male': 'Size'}).reset_index()
    cluster_summary['Cluster Name'] = cluster_summary['Active_Cluster'].map(cluster_names)
    st.dataframe(cluster_summary)

# ---- New Customer Prediction ----
elif section == "Analyze New Customer":
    st.header("📥 Analyze New Customer")
    with st.form("NewCustomerForm"):
        age = st.number_input("Age", 0, 100, 30)
        income = st.number_input("Annual Income (£K)", 0, 150, 50)
        score = st.number_input("Spending Score", 0, 100, 60)
        gender = st.radio("Gender", ["Female", "Male"])
        submit = st.form_submit_button("Predict")

    if submit:
        new_data = pd.DataFrame([{
            'Age_original': age,
            'Annual_Income (£K)_original': income,
            'Spending_Score_original': score,
            'Gender_Female': 1 if gender == "Female" else 0,
            'Gender_Male': 1 if gender == "Male" else 0
        }])

        prediction = model.predict(new_data[X_train.columns])[0]
        st.success(f"Predicted Cluster: {prediction} ({cluster_names.get(prediction, 'Unknown')})")

# ---- Custom Clustering ----
elif section == "Custom Clustering":
    st.header("🔧 Custom Clustering")
    method = st.selectbox("Clustering Algorithm", ["K-Means", "GMM", "Agglomerative", "DBSCAN"])
    data = df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]

    if method in ["K-Means", "GMM", "Agglomerative"]:
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)

    if method == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(data)
    elif method == "GMM":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(data)
    elif method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(data)
    elif method == "DBSCAN":
        eps = st.slider("Epsilon (eps)", 1.0, 20.0, 10.0)
        min_samples = st.slider("Minimum Samples", 1, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(data)

    df['Custom_Cluster'] = labels
    fig = px.scatter_3d(df, x='Age_original', y='Annual_Income (£K)_original', z='Spending_Score_original',
                        color=df['Custom_Cluster'].astype(str),
                        title="Custom Clustering Result")
    st.plotly_chart(fig)
