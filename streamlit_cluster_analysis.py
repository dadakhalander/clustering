import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering, DBSCAN
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
            return df['Cluster_k'], "Cluster_k"
        elif method == "GMM":
            return df['Cluster_gmm'], "Cluster_gmm"
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

    st.markdown("###  Clustering Quality Metrics")
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

    if cluster_method == "Agglomerative":
        st.markdown("###  Hierarchical Dendrogram")
        X = df_filtered[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]
        Z = linkage(X, method='ward')
        fig_dendro, ax = plt.subplots(figsize=(10, 4))
        dendrogram(Z, truncate_mode='level', p=5, ax=ax)
        st.pyplot(fig_dendro)

    st.header(" Cluster Ranking by Avg. Spending Score")
    cluster_spending = df_filtered.groupby('Active_Cluster')['Spending_Score_original'].mean().sort_values(ascending=False)
    st.dataframe(cluster_spending.rename("Mean Spending Score").reset_index(), use_container_width=True)

    st.subheader("👥 Cluster Sizes")
    cluster_counts = df_filtered['Active_Cluster'].value_counts().sort_index()
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Set2", ax=ax_bar)
    ax_bar.set_xlabel("Cluster")
    ax_bar.set_ylabel("Number of Customers")
    ax_bar.set_title("Cluster Sizes")
    st.pyplot(fig_bar)

    if cluster_method != "DBSCAN":
        for cluster_label in sorted(df_filtered['Active_Cluster'].unique()):
            cluster_data = df_filtered[df_filtered['Active_Cluster'] == cluster_label]

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
                title_text=f' Cluster {cluster_label} Detailed Analysis',
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

elif section == "Custom Clustering":
    st.header(" Custom Clustering")
    st.markdown("Let users experiment with their own clustering parameters.")

    method = st.selectbox("Choose Clustering Algorithm", ["Agglomerative", "DBSCAN"])
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

    df['Custom_Cluster'] = labels

    st.subheader("Cluster Results")
    st.write(df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original', 'Custom_Cluster']].head())

    fig = px.scatter_3d(df, x='Age_original', y='Annual_Income (£K)_original', z='Spending_Score_original',
                        color=df['Custom_Cluster'].astype(str), title="Custom Clustering Visualization")
    st.plotly_chart(fig)
