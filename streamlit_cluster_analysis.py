# IMPORT STATEMENTS - Must come before any Streamlit commands
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
from scipy.cluster.hierarchy import linkage, dendrogram
from io import BytesIO
import base64

# PAGE CONFIG - Must be the first Streamlit command
st.set_page_config(
    page_title="Customer Cluster Dashboard", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ---- Load Pretrained Artifacts ----
@st.cache_resource
def load_model():
    return joblib.load("best_rf.pkl")

@st.cache_resource
def load_train_data():
    return joblib.load("X_train.pkl")

@st.cache_resource
def load_cluster_info():
    return joblib.load("cluster_k_info.pkl")

# ---- Function to Analyze New Customer ----
def analyze_new_customer(new_data, model, X_train, cluster_info):
    new_customer = pd.DataFrame([new_data])
    new_customer = new_customer[X_train.columns]

    new_customer['Gender_Female'] = new_customer['Gender_Female'].astype(int)
    new_customer['Gender_Male'] = new_customer['Gender_Male'].astype(int)

    predicted_cluster = model.predict(new_customer)[0]
    
    # Enhanced cluster description
    cluster_desc = {
        0: "Budget-Conscious Shoppers",
        1: "Premium Luxury Buyers",
        2: "Middle-Class Mainstream",
        3: "Young Trendsetters",
        4: "Value-Seeking Families"
    }
    
    st.subheader(f"📊 Predicted Cluster: {predicted_cluster} - {cluster_desc.get(predicted_cluster, 'General Customers')}")
    
    # Cluster statistics
    similar_customers = cluster_info[predicted_cluster].copy()
    similar_customers['Gender_Female'] = similar_customers['Gender_Female'].astype(int)
    similar_customers['Gender_Male'] = similar_customers['Gender_Male'].astype(int)

    # Similarity analysis
    sims = cosine_similarity(similar_customers[X_train.columns], new_customer)
    most_similar_index = sims.argmax()
    most_similar_customer = similar_customers.iloc[most_similar_index]
    
    # Cluster statistics
    with st.expander("📈 Cluster Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg. Age", f"{similar_customers['Age_original'].mean():.1f} years")
        with col2:
            st.metric("Avg. Income", f"£{similar_customers['Annual_Income (£K)_original'].mean():.1f}K")
        with col3:
            st.metric("Avg. Spending", f"{similar_customers['Spending_Score_original'].mean():.1f}/100")
    
    st.subheader("👥 Most Similar Customer in Cluster:")
    st.dataframe(most_similar_customer[X_train.columns], use_container_width=True)
    
    # Radar chart comparison
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
        title=f'📊 Comparison: New Customer vs Cluster {predicted_cluster}',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(cluster_mean_max, new_customer_max) + 1]),
            angularaxis=dict(tickmode='array', tickvals=list(range(len(X_train.columns))), ticktext=X_train.columns)
        ),
        template="plotly_dark",
        font=dict(family="Arial, sans-serif", size=12, color="white"),
        showlegend=True,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Marketing recommendations based on cluster
    st.subheader("🎯 Marketing Recommendations")
    recommendations = {
        0: ["Discount offers", "Value bundles", "Budget-friendly products"],
        1: ["Premium products", "Exclusive memberships", "Personal shopping services"],
        2: ["Family packages", "Mid-range products", "Seasonal promotions"],
        3: ["Trendy new arrivals", "Social media campaigns", "Influencer collaborations"],
        4: ["Bulk purchase discounts", "Loyalty programs", "Practical product bundles"]
    }
    
    for rec in recommendations.get(predicted_cluster, ["General promotions", "Newsletter campaigns"]):
        st.markdown(f"- {rec}")

# ---- Data Loading and Caching ----
@st.cache_data
def load_data():
    df = pd.read_csv("clustering_results.csv")
    
    # Add synthetic data for demonstration if needed
    if 'Loyalty_Score' not in df.columns:
        import numpy as np
        np.random.seed(42)
        df['Loyalty_Score'] = np.random.randint(1, 6, size=len(df))
        df['Purchase_Frequency'] = np.random.choice(['Weekly', 'Bi-Weekly', 'Monthly', 'Quarterly'], size=len(df))
        df['Preferred_Category'] = np.random.choice(['Electronics', 'Fashion', 'Home', 'Beauty', 'Sports'], size=len(df))
    
    return df

# ---- Helper Functions ----
def get_cluster_descriptions():
    return {
        0: {"name": "Budget-Conscious", "desc": "Lower income, value-oriented shoppers"},
        1: {"name": "Premium Buyers", "desc": "High income, luxury purchasers"},
        2: {"name": "Mainstream", "desc": "Middle-class, balanced spenders"},
        3: {"name": "Young Trendsetters", "desc": "Younger demographic, high spending"},
        4: {"name": "Family Shoppers", "desc": "Middle-aged, practical purchases"}
    }

def get_table_download_link(df, filename="clustering_results.csv"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# ---- Main App ----
def main():
    st.title("📊 Customer Segmentation Analysis Dashboard")
    
    # Load data and models
    model = load_model()
    X_train = load_train_data()
    cluster_k_info = load_cluster_info()
    df = load_data()

    # ---- Sidebar Navigation ----
    with st.sidebar:
        st.title("Navigation")
        section = st.radio(
            "Choose Section", 
            ["📊 Cluster Analysis", "👤 Analyze New Customer", "🔧 Custom Clustering", "📚 Data Explorer"],
            label_visibility="collapsed"
        )

    # Common required columns check
    required_columns = ['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original',
                        'Gender_Male', 'Cluster_gmm', 'Cluster_k']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {missing_cols}")
        st.stop()

    if section == "📊 Cluster Analysis":
        st.header("📊 Cluster Analysis with Existing Data")
        
        # Sidebar controls
        st.sidebar.header("Filter Options")
        cluster_method = st.sidebar.selectbox("Clustering Method", ["K-Means", "GMM", "Agglomerative", "DBSCAN"])
        
        # Feature importance section
        feature_importance_section = st.sidebar.checkbox("Show Feature Importance", True)
        if feature_importance_section:
            with st.expander("📊 Feature Importance Analysis", expanded=True):
                feature_importances = model.feature_importances_
                feature_names = X_train.columns
                feature_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importances
                }).sort_values(by='Importance', ascending=False)
                
                fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                             title='Feature Importance in Cluster Prediction',
                             color='Importance', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        
        # Demographics filters
        st.sidebar.markdown("### Demographics")
        min_age, max_age = int(df['Age_original'].min()), int(df['Age_original'].max())
        min_income, max_income = int(df['Annual_Income (£K)_original'].min()), int(df['Annual_Income (£K)_original'].max())
        age_range = st.sidebar.slider("Age Range", min_age, max_age, (25, 60))
        income_range = st.sidebar.slider("Income Range (£K)", min_income, max_income, (20, 100))
        
        # Additional filters
        st.sidebar.markdown("### Behavioral Filters")
        if 'Spending_Score_original' in df.columns:
            spending_range = st.sidebar.slider("Spending Score Range", 
                                              int(df['Spending_Score_original'].min()), 
                                              int(df['Spending_Score_original'].max()), 
                                              (20, 80))
        
        if 'Loyalty_Score' in df.columns:
            loyalty_range = st.sidebar.slider("Loyalty Score", 1, 5, (1, 5))
        
        # Apply filters
        df_filtered = df[
            (df['Age_original'] >= age_range[0]) & 
            (df['Age_original'] <= age_range[1]) &
            (df['Annual_Income (£K)_original'] >= income_range[0]) &
            (df['Annual_Income (£K)_original'] <= income_range[1])
        ]
        
        if 'Spending_Score_original' in df.columns:
            df_filtered = df_filtered[
                (df_filtered['Spending_Score_original'] >= spending_range[0]) &
                (df_filtered['Spending_Score_original'] <= spending_range[1])
            ]
        
        if 'Loyalty_Score' in df.columns:
            df_filtered = df_filtered[
                (df_filtered['Loyalty_Score'] >= loyalty_range[0]) &
                (df_filtered['Loyalty_Score'] <= loyalty_range[1])
            ]
        
        # Clustering method application
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
        
        # Cluster quality metrics
        st.markdown("### 📈 Clustering Quality Metrics")
        valid_idx = df_filtered['Active_Cluster'] != -1
        X_valid = df_filtered[valid_idx][['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]
        labels_valid = df_filtered[valid_idx]['Active_Cluster']
        
        if len(set(labels_valid)) > 1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score", f"{silhouette_score(X_valid, labels_valid):.2f}",
                         help="Higher values indicate better defined clusters (range: -1 to 1)")
            with col2:
                st.metric("Davies-Bouldin Index", f"{davies_bouldin_score(X_valid, labels_valid):.2f}",
                         help="Lower values indicate better separation between clusters")
            with col3:
                st.metric("Calinski-Harabasz", f"{calinski_harabasz_score(X_valid, labels_valid):.2f}",
                         help="Higher values indicate better cluster separation")
        else:
            st.warning("⚠ Not enough clusters to compute metrics.")
        
        # Hierarchical dendrogram
        if cluster_method == "Agglomerative":
            with st.expander("🌳 Hierarchical Dendrogram", expanded=True):
                X = df_filtered[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]
                Z = linkage(X, method='ward')
                fig_dendro, ax = plt.subplots(figsize=(12, 5))
                dendrogram(Z, truncate_mode='level', p=5, ax=ax)
                plt.title("Hierarchical Clustering Dendrogram")
                plt.xlabel("Customer Index")
                plt.ylabel("Distance")
                st.pyplot(fig_dendro)
        
        # Cluster statistics
        st.header("📊 Cluster Statistics")
        
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Distributions", "📊 Comparison"])
        
        with tab1:
            st.subheader("👥 Cluster Sizes")
            cluster_counts = df_filtered['Active_Cluster'].value_counts().sort_index()
            fig_bar = px.bar(cluster_counts, 
                             x=cluster_counts.index, 
                             y=cluster_counts.values,
                             color=cluster_counts.index.astype(str),
                             labels={'x':'Cluster', 'y':'Number of Customers'},
                             title="Customer Distribution Across Clusters")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader("💰 Cluster Ranking by Avg. Spending Score")
            cluster_spending = df_filtered.groupby('Active_Cluster')['Spending_Score_original'].mean().sort_values(ascending=False)
            st.dataframe(cluster_spending.rename("Mean Spending Score").reset_index(), use_container_width=True)
        
        with tab2:
            st.subheader("📊 Feature Distributions by Cluster")
            feature = st.selectbox("Select feature to visualize", 
                                  ['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original'])
            
            fig_dist = px.box(df_filtered, x='Active_Cluster', y=feature, 
                             color='Active_Cluster', 
                             title=f"{feature} Distribution by Cluster")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab3:
            st.subheader("📈 Cluster Comparison")
            x_axis = st.selectbox("X-axis feature", 
                                 ['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original'])
            y_axis = st.selectbox("Y-axis feature", 
                                 ['Annual_Income (£K)_original', 'Spending_Score_original', 'Age_original'])
            
            fig_scatter = px.scatter(df_filtered, 
                                    x=x_axis, 
                                    y=y_axis,
                                    color='Active_Cluster',
                                    hover_data=['Gender_Male'],
                                    title=f"{y_axis} vs {x_axis} by Cluster")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Detailed cluster analysis
        st.header("🔍 Detailed Cluster Analysis")
        cluster_descriptions = get_cluster_descriptions()
        
        for cluster_label in sorted(df_filtered['Active_Cluster'].unique()):
            if cluster_label == -1:
                continue  # Skip noise points in DBSCAN
            
            cluster_data = df_filtered[df_filtered['Active_Cluster'] == cluster_label]
            desc = cluster_descriptions.get(cluster_label % 5, {"name": f"Cluster {cluster_label}", "desc": ""})
            
            with st.expander(f"🔍 {desc['name']} - Cluster {cluster_label}: {desc['desc']}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Customers", len(cluster_data))
                with col2:
                    st.metric("Avg. Age", f"{cluster_data['Age_original'].mean():.1f} years")
                with col3:
                    st.metric("Avg. Income", f"£{cluster_data['Annual_Income (£K)_original'].mean():.1f}K")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "👥 Demographics", "💰 Spending", "📈 Trends"])
                
                with tab1:
                    fig = make_subplots(
                        rows=2, cols=2,
                        specs=[[{"type": "histogram"}, {"type": "pie"}],
                               [{"type": "box"}, {"type": "violin"}]],
                        subplot_titles=(
                            f'Spending Distribution - Cluster {cluster_label}',
                            f'Gender Split - Cluster {cluster_label}',
                            f'Age Distribution - Cluster {cluster_label}',
                            f'Income Distribution - Cluster {cluster_label}'
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
                        title_text=f'Cluster {cluster_label} Detailed Analysis',
                        showlegend=False, 
                        height=700
                    )

                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    if 'Preferred_Category' in cluster_data.columns:
                        fig_cat = px.bar(cluster_data['Preferred_Category'].value_counts(),
                                        title="Preferred Product Categories")
                        st.plotly_chart(fig_cat, use_container_width=True)
                    
                    if 'Loyalty_Score' in cluster_data.columns:
                        fig_loyalty = px.histogram(cluster_data, x='Loyalty_Score', nbins=5,
                                                 title="Loyalty Score Distribution")
                        st.plotly_chart(fig_loyalty, use_container_width=True)
                
                with tab3:
                    fig_spend_age = px.scatter(cluster_data, 
                                             x='Age_original', 
                                             y='Spending_Score_original',
                                             color='Gender_Male',
                                             title="Spending vs Age")
                    st.plotly_chart(fig_spend_age, use_container_width=True)
                    
                    fig_spend_income = px.scatter(cluster_data,
                                                x='Annual_Income (£K)_original',
                                                y='Spending_Score_original',
                                                color='Gender_Male',
                                                title="Spending vs Income")
                    st.plotly_chart(fig_spend_income, use_container_width=True)
                
                with tab4:
                    if 'Purchase_Frequency' in cluster_data.columns:
                        fig_freq = px.pie(cluster_data, names='Purchase_Frequency',
                                        title="Purchase Frequency")
                        st.plotly_chart(fig_freq, use_container_width=True)
                    
                    # Time-based analysis would go here if we had date data
        
        # Data export
        st.markdown("### 💾 Export Data")
        st.markdown(get_table_download_link(df_filtered, "filtered_clusters.csv"), unsafe_allow_html=True)

    elif section == "👤 Analyze New Customer":
        st.header("👤 Analyze New Customer Data")
        st.markdown("Enter customer details to predict their cluster and find similar customers.")
        
        with st.form(key='customer_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input('Age', min_value=0, max_value=100, value=32)
                income = st.number_input('Annual Income (£K)', min_value=0, max_value=500, value=70)
            
            with col2:
                spending_score = st.number_input('Spending Score (1-100)', min_value=0, max_value=100, value=85)
                gender = st.radio('⚧ Gender', ['Female', 'Male'], index=0)
            
            # Additional behavioral data
            with st.expander("➕ Additional Behavioral Data"):
                loyalty_score = st.slider('Loyalty Score (1-5)', 1, 5, 3)
                purchase_freq = st.selectbox('Purchase Frequency', 
                                           ['Weekly', 'Bi-Weekly', 'Monthly', 'Quarterly', 'Yearly'])
                preferred_category = st.selectbox('Preferred Category', 
                                                ['Electronics', 'Fashion', 'Home', 'Beauty', 'Sports', 'Other'])
            
            submitted = st.form_submit_button("Analyze Customer")

        if submitted:
            gender_female = 1 if gender == 'Female' else 0
            gender_male = 1 if gender == 'Male' else 0

            new_data = {
                'Age_original': age,
                'Annual_Income (£K)_original': income,
                'Spending_Score_original': spending_score,
                'Gender_Female': gender_female,
                'Gender_Male': gender_male,
                'Loyalty_Score': loyalty_score,
                'Purchase_Frequency': purchase_freq,
                'Preferred_Category': preferred_category
            }

            analyze_new_customer(new_data, model, X_train, cluster_k_info)

    elif section == "🔧 Custom Clustering":
        st.header("🔧 Custom Clustering")
        st.markdown("Experiment with different clustering algorithms and parameters.")
        
        # Algorithm selection
        method = st.selectbox("Choose Clustering Algorithm", ["K-Means", "GMM", "Agglomerative", "DBSCAN"])
        data = df[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]
        
        # Parameter controls
        with st.expander("⚙️ Algorithm Parameters", expanded=True):
            if method in ["K-Means", "GMM", "Agglomerative"]:
                n_clusters = st.slider("Number of Clusters", 2, 10, 4)
            
            if method == "K-Means":
                init_method = st.selectbox("Initialization Method", ["k-means++", "random"])
                max_iter = st.slider("Maximum Iterations", 100, 500, 300)
            
            elif method == "GMM":
                covariance_type = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])
            
            elif method == "DBSCAN":
                col1, col2 = st.columns(2)
                with col1:
                    eps = st.slider("Epsilon (eps)", 1.0, 20.0, 10.0, step=0.5)
                with col2:
                    min_samples = st.slider("Minimum Samples", 1, 10, 5)
        
        # Apply clustering
        if st.button("Run Clustering"):
            with st.spinner("Clustering in progress..."):
                if method == "K-Means":
                    model = KMeans(n_clusters=n_clusters, init=init_method, max_iter=max_iter, random_state=42)
                    labels = model.fit_predict(data)
                
                elif method == "GMM":
                    model = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, random_state=42)
                    labels = model.fit_predict(data)
                
                elif method == "Agglomerative":
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = model.fit_predict(data)
                
                elif method == "DBSCAN":
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = model.fit_predict(data)
                
                df['Custom_Cluster'] = labels
                
                # Show results
                st.success("Clustering completed!")
                
                # Cluster metrics
                valid_idx = df['Custom_Cluster'] != -1
                X_valid = df[valid_idx][['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']]
                labels_valid = df[valid_idx]['Custom_Cluster']
                
                if len(set(labels_valid)) > 1:
                    st.subheader("📊 Clustering Quality Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Silhouette Score", f"{silhouette_score(X_valid, labels_valid):.2f}")
                    with col2:
                        st.metric("Davies-Bouldin Index", f"{davies_bouldin_score(X_valid, labels_valid):.2f}")
                    with col3:
                        st.metric("Calinski-Harabasz", f"{calinski_harabasz_score(X_valid, labels_valid):.2f}")
                
                # Cluster visualization
                st.subheader("📊 Cluster Visualization")
                
                tab1, tab2 = st.tabs(["3D Scatter Plot", "2D Projections"])
                
                with tab1:
                    fig_3d = px.scatter_3d(df, 
                                          x='Age_original', 
                                          y='Annual_Income (£K)_original', 
                                          z='Spending_Score_original',
                                          color='Custom_Cluster',
                                          title="3D Cluster Visualization",
                                          hover_name='Custom_Cluster',
                                          opacity=0.7)
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                with tab2:
                    fig_age_income = px.scatter(df, 
                                              x='Age_original', 
                                              y='Annual_Income (£K)_original',
                                              color='Custom_Cluster',
                                              title="Age vs Income")
                    st.plotly_chart(fig_age_income, use_container_width=True)
                    
                    fig_age_spend = px.scatter(df,
                                             x='Age_original',
                                             y='Spending_Score_original',
                                             color='Custom_Cluster',
                                             title="Age vs Spending")
                    st.plotly_chart(fig_age_spend, use_container_width=True)
                    
                    fig_income_spend = px.scatter(df,
                                                x='Annual_Income (£K)_original',
                                                y='Spending_Score_original',
                                                color='Custom_Cluster',
                                                title="Income vs Spending")
                    st.plotly_chart(fig_income_spend, use_container_width=True)
                
                # Cluster statistics
                st.subheader("📈 Cluster Statistics")
                cluster_stats = df.groupby('Custom_Cluster').agg({
                    'Age_original': ['mean', 'std', 'count'],
                    'Annual_Income (£K)_original': ['mean', 'std'],
                    'Spending_Score_original': ['mean', 'std']
                })
                st.dataframe(cluster_stats, use_container_width=True)
                
                # Export results
                st.markdown(get_table_download_link(df, "custom_clustering_results.csv"), unsafe_allow_html=True)

    elif section == "📚 Data Explorer":
        st.header("📚 Data Explorer")
        st.markdown("Explore and visualize the raw customer data.")
        
        # Show raw data
        with st.expander("🔍 View Raw Data", expanded=True):
            st.dataframe(df, use_container_width=True)
        
        # Data summary
        st.subheader("📊 Data Summary")
        st.write(df.describe())
        
        # Interactive visualizations
        st.subheader("📈 Interactive Visualizations")
        
        viz_type = st.selectbox("Visualization Type", 
                               ["Scatter Plot", "Histogram", "Box Plot", "Violin Plot", "Correlation Matrix"])
        
        if viz_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", df.select_dtypes(include='number').columns)
            with col2:
                y_axis = st.selectbox("Y-axis", df.select_dtypes(include='number').columns)
            
            color_by = st.selectbox("Color by", [None] + list(df.columns))
            
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, hover_data=df.columns)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Histogram":
            column = st.selectbox("Select column", df.select_dtypes(include='number').columns)
            bins = st.slider("Number of bins", 5, 100, 20)
            
            fig = px.histogram(df, x=column, nbins=bins)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            column = st.selectbox("Select column", df.select_dtypes(include='number').columns)
            group_by = st.selectbox("Group by", [None] + list(df.select_dtypes(exclude='number').columns))
            
            fig = px.box(df, y=column, x=group_by)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Violin Plot":
            column = st.selectbox("Select column", df.select_dtypes(include='number').columns)
            group_by = st.selectbox("Group by", [None] + list(df.select_dtypes(exclude='number').columns))
            
            fig = px.violin(df, y=column, x=group_by, box=True)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Correlation Matrix":
            numeric_df = df.select_dtypes(include='number')
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(corr_matrix,
                           text_auto=True,
                           aspect="auto",
                           color_continuous_scale='RdBu',
                           range_color=[-1, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        # Data export
        st.markdown("### 💾 Export Data")
        st.markdown(get_table_download_link(df, "customer_data.csv"), unsafe_allow_html=True)

    # ---- Footer ----
    st.markdown("---")
    st.markdown("""
        **Customer Segmentation Dashboard**  
        *Powered by Streamlit and Scikit-learn*  
        [GitHub Repository](#) | [Documentation](#)
    """)

if __name__ == "__main__":
    main()
