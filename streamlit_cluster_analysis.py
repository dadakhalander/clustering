import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

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

# ---- Advanced Filter Combinations ----
st.sidebar.header("Advanced Filter Options")

# Filter for Gender
gender_filter = st.sidebar.multiselect("Select Gender", options=["Male", "Female"], default=["Male", "Female"])

# Filter for Age Range
age_filter_min = st.sidebar.slider("Min Age", min_value=int(df['Age_original'].min()), max_value=int(df['Age_original'].max()), value=20)
age_filter_max = st.sidebar.slider("Max Age", min_value=int(df['Age_original'].min()), max_value=int(df['Age_original'].max()), value=60)

# Apply filter to the dataset
df_filtered = df[(df['Age_original'] >= age_filter_min) & (df['Age_original'] <= age_filter_max)]

if "Male" not in gender_filter:
    df_filtered = df_filtered[df_filtered['Gender_Male'] == 0]
if "Female" not in gender_filter:
    df_filtered = df_filtered[df_filtered['Gender_Female'] == 0]

# Display the filtered data
st.subheader(f"Filtered Data (Age: {age_filter_min}-{age_filter_max}, Gender: {', '.join(gender_filter)})")
st.write(df_filtered)

if section == "Cluster Analysis":
    st.header("Cluster Analysis with Existing Data")

    feature_importance_section = st.sidebar.checkbox("Show Feature Importance", False)

    if feature_importance_section:
        with st.expander(" Feature Importance Chart"):
            st.subheader("Feature Importance Analysis")
            feature_importances = model.feature_importances_
            feature_names = X_train.columns
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(8, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_df, palette='Blues_d')
            st.pyplot(plt.gcf())

    required_columns = ['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original',
                        'Gender_Male', 'Cluster_gmm', 'Cluster_k']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {missing_cols}")
        st.stop()

    st.sidebar.header(" Filter Options")
    cluster_method = st.sidebar.selectbox("Clustering Method", ["K-Means", "GMM", "Agglomerative", "DBSCAN"])

    st.sidebar.markdown("### Demographics")
    min_age, max_age = int(df['Age_original'].min()), int(df['Age_original'].max())
    min_income, max_income = int(df['Annual_Income (£K)_original'].min()), int(df['Annual_Income (£K)_original'].max())
    age_range = st.sidebar.slider("Age Range", min_age, max_age, (25, 60))
    income_range = st.sidebar.slider("Income Range (£K)", min_income, max_income, (20, 100))

    df_filtered = df[(df['Age_original'] >= age_range[0]) & (df['Age_original'] <= age_range[1]) &
                     (df['Annual_Income (£K)_original'] >= income_range[0]) &
                     (df['Annual_Income (£K)_original'] <= income_range[1])]

    def apply_clustering(method, df):
        X = df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]

        if method == "K-Means":
            model = KMeans(n_clusters=5, random_state=42)
            labels = model.fit_predict(X)
            return labels, "KMeans"
        elif method == "GMM":
            model = GaussianMixture(n_components=5, random_state=42)
            labels = model.fit_predict(X)
            return labels, "GMM"
        elif method == "Agglomerative":
            model = AgglomerativeClustering(n_clusters=5)
            labels = model.fit_predict(X)
            return labels, "Agglomerative"
        elif method == "DBSCAN":
            model = DBSCAN(eps=10, min_samples=5)
            labels = model.fit_predict(X)
            return labels, "DBSCAN"

    labels, label_col = apply_clustering(cluster_method, df_filtered)
    df_filtered['Active_Cluster'] = labels

    st.markdown("### Clustering Quality Metrics")
    valid_idx = df_filtered['Active_Cluster'] != -1
    X_valid = df_filtered[valid_idx][['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]
    labels_valid = df_filtered[valid_idx]['Active_Cluster']

    if len(set(labels_valid)) > 1:
        sil = silhouette_score(X_valid, labels_valid)
        db = davies_bouldin_score(X_valid, labels_valid)
        ch = calinski_harabasz_score(X_valid, labels_valid)

        st.markdown(f"- **Silhouette Score:** {sil:.2f}")
        st.markdown(f"- **Davies-Bouldin Index:** {db:.2f}")
        st.markdown(f"- **Calinski-Harabasz Score:** {ch:.2f}")
    else:
        st.warning("⚠ Not enough clusters to compute metrics.")

    # ---- Comparison of Metrics Across Algorithms ----
    st.header("Clustering Quality Metric Comparison")

    metrics = ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Score"]
    algorithms = ["K-Means", "GMM", "Agglomerative", "DBSCAN"]

    # Placeholder for storing metric scores
    metric_scores = {algo: [] for algo in algorithms}

    for algo in algorithms:
        if algo == "K-Means":
            model = KMeans(n_clusters=5, random_state=42)
        elif algo == "GMM":
            model = GaussianMixture(n_components=5, random_state=42)
        elif algo == "Agglomerative":
            model = AgglomerativeClustering(n_clusters=5)
        else:  # DBSCAN
            model = DBSCAN(eps=10, min_samples=5)

        # Fit model and calculate metrics
        if algo != "DBSCAN":
            model.fit(df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']])
            labels = model.labels_
        else:
            labels = model.fit_predict(df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']])

        # Calculate metrics
        if len(set(labels)) > 1:
            sil = silhouette_score(df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']], labels)
            db = davies_bouldin_score(df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']], labels)
            ch = calinski_harabasz_score(df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']], labels)

            metric_scores[algo] = [sil, db, ch]
        else:
            metric_scores[algo] = [None, None, None]  # In case metrics can't be computed

    # Create a dataframe for plotting
    metric_df = pd.DataFrame(metric_scores, index=metrics)

    # Plotting the comparison of metric scores
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_df.plot(kind='bar', ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title('Comparison of Clustering Algorithm Performance Metrics')
    plt.ylabel('Metric Score')
    plt.xticks(rotation=45)
    st.pyplot(fig)
