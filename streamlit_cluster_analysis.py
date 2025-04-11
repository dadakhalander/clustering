import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="🧠 Customer Cluster Dashboard", layout="wide")
st.title("🧠 Customer Segmentation Analysis Dashboard")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("clustering_results.csv")

df = load_data()

# Validate required columns
required_columns = [
    'Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original',
    'Gender_Male', 'Cluster_gmm', 'Cluster_k'
]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns in dataset: {missing_cols}")
    st.stop()

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
cluster_type = st.sidebar.selectbox("📂 Select Clustering Method", ["Cluster_gmm", "Cluster_k"])
age_range = st.sidebar.slider("🎂 Age Range Filter", int(df['Age_original'].min()), int(df['Age_original'].max()), (18, 70))
income_range = st.sidebar.slider("💰 Income Range Filter (in £K)", int(df['Annual_Income (£K)_original'].min()), int(df['Annual_Income (£K)_original'].max()), (0, 150))

# Filter dataset
df_filtered = df[
    (df['Age_original'] >= age_range[0]) & (df['Age_original'] <= age_range[1]) &
    (df['Annual_Income (£K)_original'] >= income_range[0]) & (df['Annual_Income (£K)_original'] <= income_range[1])
]

# Cluster Summary Table
st.header("📌 Cluster Ranking Based on Average Spending Score")
cluster_spending = df_filtered.groupby(cluster_type)['Spending_Score_original'].mean().sort_values(ascending=False)
st.dataframe(cluster_spending.rename("Mean Spending Score").reset_index(), use_container_width=True)

# Cluster Size Bar Chart
st.subheader("👥 Cluster Sizes")
cluster_counts = df_filtered[cluster_type].value_counts().sort_index()
fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Set2", ax=ax_bar)
ax_bar.set_xlabel("Cluster")
ax_bar.set_ylabel("Number of Customers")
ax_bar.set_title("Cluster Sizes")
st.pyplot(fig_bar)

# Define Colors
cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Cluster Deep Dive
for cluster_label in sorted(df_filtered[cluster_type].unique()):
    st.markdown("---")
    st.subheader(f"🔬 Cluster {cluster_label} Analysis")

    cluster_data = df_filtered[df_filtered[cluster_type] == cluster_label]
    color = cluster_colors[cluster_label % len(cluster_colors)]

    # Gender breakdown
    gender_counts = cluster_data['Gender_Male'].value_counts()
    gender_labels = ["Female", "Male"]
    gender_colors = ["#f78da7", "#7ec8e3"]

    col1, col2 = st.columns([2, 1])

    # Distribution Plots
    with col1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        features = ['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']
        titles = ['Age', 'Annual Income (£K)', 'Spending Score']

        for i in range(3):
            sns.histplot(cluster_data[cluster_data['Gender_Male'] == 0][features[i]],
                         kde=True, color=color, label="Female", ax=axes[i])
            sns.histplot(cluster_data[cluster_data['Gender_Male'] == 1][features[i]],
                         kde=True, color="gray", label="Male", ax=axes[i])
            axes[i].set_title(f"{titles[i]} Distribution")
            axes[i].legend()

        plt.tight_layout()
        st.pyplot(fig)

    # Pie Chart
    with col2:
        st.markdown("#### 👥 Gender Distribution")
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        ax_pie.pie(gender_counts, labels=gender_labels, colors=gender_colors, autopct='%1.1f%%', startangle=90)
        ax_pie.set_title("Gender Composition")
        ax_pie.axis('equal')
        st.pyplot(fig_pie)

    # Stats
    st.markdown("#### 📈 Cluster Statistics")
    st.dataframe(cluster_data[features].describe().T.style.format("{:.2f}"))

    # Download Button
    csv = cluster_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Cluster {cluster_label} Data as CSV",
        data=csv,
        file_name=f'cluster_{cluster_label}_data.csv',
        mime='text/csv'
    )

# Feature Boxplots
st.markdown("---")
st.header("📦 Feature Distribution Across Clusters")

for feature in ['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']:
    st.subheader(f"📊 {feature}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**GMM Clustering**")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.boxplot(x=df_filtered['Cluster_gmm'], y=df_filtered[feature], palette="Set3", ax=ax1)
        ax1.set_xlabel("GMM Cluster")
        ax1.set_ylabel(feature)
        st.pyplot(fig1)

    with col2:
        st.markdown("**K-Means Clustering**")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.boxplot(x=df_filtered['Cluster_k'], y=df_filtered[feature], palette="Set2", ax=ax2)
        ax2.set_xlabel("K-Means Cluster")
        ax2.set_ylabel(feature)
        st.pyplot(fig2)

# Optional: Cluster Center Visualization (only if available)
st.markdown("---")
st.header("📍 Optional: Cluster Centers (If Available)")
if 'Centroid_X' in df.columns and 'Centroid_Y' in df.columns:
    fig_centers, ax_centers = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df_filtered, x='Centroid_X', y='Centroid_Y', hue=cluster_type, palette='tab10', s=100)
    ax_centers.set_title("Visualized Cluster Centers")
    st.pyplot(fig_centers)
else:
    st.info("ℹ️ Cluster center coordinates not available in this dataset.")

# Final Notes
st.markdown("---")
st.info("""
📌 **Use Cases:**
- Identify high spenders with low income to target with discount promotions.
- Offer premium services to high income, low spending clusters.
- Monitor age-wise cluster behavior for seasonal targeting.
""")

st.caption("Built with ❤️ using Streamlit | Data source: clustering_results.csv")
