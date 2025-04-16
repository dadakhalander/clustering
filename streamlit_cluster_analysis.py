import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram

# ---- Load Pretrained Artifacts ----
model = joblib.load("best_rf.pkl")
X_train = joblib.load("X_train.pkl")
cluster_k_info = joblib.load("cluster_k_info.pkl")

# ---- Function to Analyze New Customer ----
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
        line=dict(color='royalblue', width=3),
        fill='toself',
        fillcolor='rgba(65, 105, 225, 0.3)',
        opacity=0.8
    )

    trace2 = go.Scatterpolar(
        r=new_customer.values[0],
        theta=X_train.columns,
        name='New Customer',
        line=dict(color='darkorange', width=3),
        fill='toself',
        fillcolor='rgba(255, 140, 0, 0.3)',
        opacity=0.8
    )

    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(
        title=f' Comparison: New Customer vs Cluster {predicted_cluster}',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(cluster_mean_max, new_customer_max) + 1]),
            angularaxis=dict(tickmode='array', tickvals=list(range(len(X_train.columns))), ticktext=X_train.columns)
        ),
        template="plotly_dark",
        font=dict(family="Arial, sans-serif", size=12, color="white"),
        showlegend=True
    )

    st.plotly_chart(fig)

# ---- Streamlit App UI ----
st.set_page_config(page_title="Customer Cluster Dashboard", layout="wide")
st.title(" Customer Segmentation Analysis Dashboard")

# ---- Sidebar Navigation ----
section = st.sidebar.radio(" Choose Section", ["Cluster Analysis", "Custom Clustering", "Analyze New Customer Data"])

# ---- Load Data Function ----
@st.cache_data
def load_data():
    return pd.read_csv("clustering_results.csv")

# ---- Custom Clustering Section ----
if section == "Custom Clustering":
    st.header("Custom Clustering with Your Parameters")
    df = load_data()

    st.sidebar.header("Custom Clustering Options")
    cluster_alg = st.sidebar.selectbox("Choose Clustering Algorithm", ["KMeans", "GMM", "Agglomerative", "DBSCAN"])
    n_clusters = st.sidebar.slider("Number of Clusters (if applicable)", 2, 10, 5)
    eps = st.sidebar.slider("DBSCAN eps", 1.0, 20.0, 10.0)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 1, 10, 5)

    features = ['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if cluster_alg == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif cluster_alg == "GMM":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif cluster_alg == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
    elif cluster_alg == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)

    df['Custom_Cluster'] = labels

    st.markdown("### Custom Clustering Metrics")
    valid_idx = df['Custom_Cluster'] != -1
    if len(set(df[valid_idx]['Custom_Cluster'])) > 1:
        sil = silhouette_score(X_scaled[valid_idx], df[valid_idx]['Custom_Cluster'])
        db = davies_bouldin_score(X_scaled[valid_idx], df[valid_idx]['Custom_Cluster'])
        ch = calinski_harabasz_score(X_scaled[valid_idx], df[valid_idx]['Custom_Cluster'])

        st.markdown(f"- **Silhouette Score:** {sil:.2f}")
        st.markdown(f"- **Davies-Bouldin Index:** {db:.2f}")
        st.markdown(f"- **Calinski-Harabasz Score:** {ch:.2f}")
    else:
        st.warning("⚠ Not enough clusters to compute metrics.")

    st.subheader("📊 Custom Cluster Sizes")
    cluster_counts = df['Custom_Cluster'].value_counts().sort_index()
    fig_bar, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="coolwarm", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Custom Cluster Sizes")
    st.pyplot(fig_bar)

    for cluster_label in sorted(df['Custom_Cluster'].unique()):
        cluster_data = df[df['Custom_Cluster'] == cluster_label]

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "histogram"}, {"type": "pie"}],
                   [{"type": "box"}, {"type": "violin"}]],
            subplot_titles=(
                f'Spending Score - Cluster {cluster_label}',
                f'Gender Split - Cluster {cluster_label}',
                f'Age Box Plot - Cluster {cluster_label}',
                f'Income Violin - Cluster {cluster_label}'
            )
        )

        fig.add_trace(go.Histogram(x=cluster_data['Spending_Score_original'], nbinsx=10,
                                   marker_color='skyblue'), row=1, col=1)

        gender_counts = cluster_data['Gender_Male'].value_counts()
        fig.add_trace(go.Pie(labels=['Male', 'Female'],
                             values=[gender_counts.get(1, 0), gender_counts.get(0, 0)],
                             marker_colors=['skyblue', 'lightcoral'],
                             textinfo='percent+label'), row=1, col=2)

        fig.add_trace(go.Box(y=cluster_data['Age_original'],
                             marker_color='lightgreen', boxmean='sd'), row=2, col=1)

        fig.add_trace(go.Violin(y=cluster_data['Annual_Income (£K)_original'],
                                box_visible=True, meanline_visible=True,
                                line_color='orange'), row=2, col=2)

        fig.update_layout(
            title_text=f' Custom Cluster {cluster_label} Detailed Analysis',
            showlegend=False, height=900, width=1000
        )

        st.plotly_chart(fig)

# --- (The rest of the app remains unchanged below for Analyze New Customer Data section) ---
