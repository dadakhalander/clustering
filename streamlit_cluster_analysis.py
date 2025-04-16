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
section = st.sidebar.radio(" Choose Section", ["Cluster Analysis", "Analyze New Customer Data", "Custom Clustering"])

@st.cache_data
def load_data():
    return pd.read_csv("clustering_results.csv")

df = load_data()
required_columns = ['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original', 'Gender_Male']
if not all(col in df.columns for col in required_columns):
    st.error("Missing required columns in the dataset.")
    st.stop()

if section == "Custom Clustering":
    st.header("🛠️ Custom Clustering")

    algorithm = st.selectbox("Choose Clustering Algorithm", ["KMeans", "Gaussian Mixture", "Agglomerative", "DBSCAN"])
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
    eps = st.slider("DBSCAN: Epsilon (eps)", min_value=1.0, max_value=20.0, step=0.5, value=5.0)
    min_samples = st.slider("DBSCAN: Min Samples", min_value=1, max_value=20, value=5)

    run = st.button("Run Clustering")
    if run:
        X_custom = df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]

        if algorithm == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=0)
        elif algorithm == "Gaussian Mixture":
            model = GaussianMixture(n_components=n_clusters, random_state=0)
        elif algorithm == "Agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif algorithm == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)

        labels = model.fit_predict(X_custom)
        df['CustomCluster'] = labels

        valid = df['CustomCluster'] != -1
        if len(set(df[valid]['CustomCluster'])) > 1:
            sil = silhouette_score(X_custom[valid], df[valid]['CustomCluster'])
            db = davies_bouldin_score(X_custom[valid], df[valid]['CustomCluster'])
            ch = calinski_harabasz_score(X_custom[valid], df[valid]['CustomCluster'])

            st.markdown(f"**Silhouette Score:** {sil:.2f}")
            st.markdown(f"**Davies-Bouldin Index:** {db:.2f}")
            st.markdown(f"**Calinski-Harabasz Score:** {ch:.2f}")
        else:
            st.warning("⚠ Not enough clusters to compute metrics.")

        with st.expander("📊 Show Cluster Visuals"):
            for cluster_id in sorted(df['CustomCluster'].unique()):
                cluster_data = df[df['CustomCluster'] == cluster_id]
                fig = make_subplots(
                    rows=2, cols=2,
                    specs=[[{"type": "histogram"}, {"type": "pie"}],
                           [{"type": "box"}, {"type": "violin"}]],
                    subplot_titles=(
                        f'Spending Score - Cluster {cluster_id}',
                        f'Gender Split - Cluster {cluster_id}',
                        f'Age Box Plot - Cluster {cluster_id}',
                        f'Income Violin - Cluster {cluster_id}'
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
                    title_text=f' Cluster {cluster_id} Detailed Analysis',
                    showlegend=False, height=900, width=1000
                )

                st.plotly_chart(fig)

elif section == "Analyze New Customer Data":
    st.header("Analyze New Customer Data")
    with st.form(key='customer_form'):
        age = st.number_input('Age', min_value=0, max_value=100, value=32)
        income = st.number_input('Annual Income (£K)', min_value=0, max_value=500, value=70)
        spending_score = st.number_input('Spending Score', min_value=0, max_value=100, value=85)
        gender = st.radio('⚧ Gender', ['Female', 'Male'], index=0)
        submitted = st.form_submit_button("Analyze")

    if submitted:
        gender_female = 1 if gender == 'Female' else 0
        gender_male = 1 if gender == 'Male' else 0

        new_data = {
            'Age_original': age,
            'Annual_Income (£K)_original': income,
            'Spending_Score_original': spending_score,
            'Gender_Female': gender_female,
            'Gender_Male': gender_male
        }

        analyze_new_customer(new_data, model, X_train, cluster_k_info)
