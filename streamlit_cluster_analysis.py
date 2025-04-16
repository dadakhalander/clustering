import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ---- Load Pretrained Artifacts ----
model = joblib.load("best_rf.pkl")
X_train = joblib.load("X_train.pkl")
cluster_k_info = joblib.load("cluster_k_info.pkl")

# ---- Analyze New Customer Function ----
def analyze_new_customer(new_data, model, X_train, cluster_info):
    new_customer = pd.DataFrame([new_data])
    new_customer = new_customer[X_train.columns]

    new_customer['Gender_Female'] = new_customer['Gender_Female'].astype(int)
    new_customer['Gender_Male'] = new_customer['Gender_Male'].astype(int)

    predicted_cluster = model.predict(new_customer)[0]
    st.subheader(f" Predicted Cluster: {predicted_cluster}")

    similar_customers = cluster_info[predicted_cluster].copy()
    similar_customers['Gender_Female'] = similar_customers['Gender_Female'].astype(int)
    similar_customers['Gender_Male'] = similar_customers['Gender_Male'].astype(int)

    sims = cosine_similarity(similar_customers[X_train.columns], new_customer)
    most_similar_index = sims.argmax()
    most_similar_customer = similar_customers.iloc[most_similar_index]

    st.subheader(" Most Similar Customer in Cluster:")
    st.dataframe(most_similar_customer[X_train.columns])

    cluster_mean = similar_customers[X_train.columns].mean()
    cluster_mean_max = cluster_mean.max()
    new_customer_max = new_customer.max().values[0]

    trace1 = go.Scatterpolar(
        r=cluster_mean,
        theta=X_train.columns,
        name='Cluster Mean',
        fill='toself',
        opacity=0.7
    )

    trace2 = go.Scatterpolar(
        r=new_customer.values[0],
        theta=X_train.columns,
        name='New Customer',
        fill='toself',
        opacity=0.7
    )

    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(
        title=f' Comparison: New Customer vs Cluster {predicted_cluster}',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(cluster_mean_max, new_customer_max) + 1])
        ),
        showlegend=True
    )

    st.plotly_chart(fig)

# ---- Streamlit App ----
st.set_page_config(page_title="Customer Cluster Dashboard", layout="wide")
st.title("🔍 Customer Segmentation Dashboard")

section = st.sidebar.radio("Select Section", ["Cluster Analysis", "Analyze New Customer Data", "Custom Clustering"])

@st.cache_data
def load_data():
    return pd.read_csv("clustering_results.csv")

df = load_data()

if section == "Cluster Analysis":
    st.header("Cluster Distribution")
    st.plotly_chart(px.scatter_3d(
        df,
        x='Age_original',
        y='Annual_Income (£K)_original',
        z='Spending_Score_original',
        color='Cluster_Label',
        title="Precomputed Clusters in 3D"
    ))

elif section == "Analyze New Customer Data":
    st.header("🧍‍♂️ Analyze a New Customer")
    new_data = {
        'Age': st.slider("Age", 18, 70, 30),
        'Annual_Income (£K)': st.slider("Annual Income (£K)", 10, 150, 50),
        'Spending_Score': st.slider("Spending Score", 1, 100, 50),
        'Gender_Female': 0,
        'Gender_Male': 0
    }

    gender = st.radio("Gender", ['Male', 'Female'])
    if gender == 'Male':
        new_data['Gender_Male'] = 1
    else:
        new_data['Gender_Female'] = 1

    if st.button("Analyze"):
        analyze_new_customer(new_data, model, X_train, cluster_k_info)

elif section == "Custom Clustering":
    st.header("⚙️ Custom Clustering Explorer")

    method = st.selectbox("Choose Clustering Algorithm", ["Agglomerative", "DBSCAN", "K-Means", "GMM"])
    data = df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]

    if method == "Agglomerative":
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(data)

    elif method == "DBSCAN":
        eps = st.slider("Epsilon (eps)", 1.0, 20.0, 10.0)
        min_samples = st.slider("Minimum Samples", 1, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(data)

    elif method == "K-Means":
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(data)

    elif method == "GMM":
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(data)

    df['Custom_Cluster'] = labels

    st.subheader("Clustered Data Preview")
    st.write(df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original', 'Custom_Cluster']].head())

    fig = px.scatter_3d(
        df,
        x='Age_original',
        y='Annual_Income (£K)_original',
        z='Spending_Score_original',
        color=df['Custom_Cluster'].astype(str),
        title=f"{method} Clustering Result"
    )
    st.plotly_chart(fig)

    # Clustering metrics
    if method != "DBSCAN" or len(set(labels)) > 1:
        valid_data = df[df['Custom_Cluster'] != -1]
        X_valid = valid_data[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]
        y_valid = valid_data['Custom_Cluster']

        if y_valid.nunique() > 1:
            sil = silhouette_score(X_valid, y_valid)
            db = davies_bouldin_score(X_valid, y_valid)
            ch = calinski_harabasz_score(X_valid, y_valid)

            st.markdown("### 📊 Clustering Quality Metrics")
            st.markdown(f"- **Silhouette Score:** {sil:.2f}")
            st.markdown(f"- **Davies-Bouldin Index:** {db:.2f}")
            st.markdown(f"- **Calinski-Harabasz Score:** {ch:.2f}")
        else:
            st.warning("🚨 Not enough clusters to compute clustering quality scores.")
