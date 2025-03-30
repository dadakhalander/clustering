import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (Modify to your actual dataset loading method)
@st.cache_data
def load_data():
    return pd.read_csv("clustering_results.csv")  # Replace with actual file path

df = load_data()

# Streamlit App Title
st.title("📊 Cluster Analysis: Spending, Income & Gender Distribution")

# Select cluster type (GMM or K-Means)
cluster_type = st.sidebar.selectbox("Select Cluster Type", ["Cluster_gmm", "Cluster_k"])

# Group data by cluster and sort by mean Spending_Score in descending order
cluster_spending = df.groupby(cluster_type)['Spending_Score_original'].mean().sort_values(ascending=False)

# Print the cluster rankings
st.subheader("Cluster Ranking by Mean Spending Score:")
for cluster, mean_spending in cluster_spending.items():
    st.write(f"Cluster {cluster}: {mean_spending:.2f}")

# Get unique clusters
clusters = sorted(df[cluster_type].unique())

# Define colors for each cluster (you can adjust or add more if necessary)
cluster_colors = ['blue', 'orange', 'green', 'red']

# Iterate through clusters and create analysis plots
for cluster_label in clusters:
    cluster_data = df[df[cluster_type] == cluster_label]

    # Display cluster title
    st.subheader(f"Cluster {cluster_label} Analysis")

    # Create a figure for the three subplots (Age, Income, Spending)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get the color for the current cluster using modulo operator
    color = cluster_colors[cluster_label % len(cluster_colors)]

    # Plot Age distribution, divided by Gender_Male
    sns.histplot(cluster_data[cluster_data['Gender_Male'] == 0]['Age_original'],
                 kde=True, color=color, label='Female', ax=axes[0])
    sns.histplot(cluster_data[cluster_data['Gender_Male'] == 1]['Age_original'],
                 kde=True, color='gray', label='Male', ax=axes[0])
    axes[0].set_title(f'Age Distribution - Cluster {cluster_label}')
    axes[0].legend()  # Add legend to distinguish genders

    # Plot Annual Income distribution, divided by Gender_Male
    sns.histplot(cluster_data[cluster_data['Gender_Male'] == 0]['Annual_Income (£K)_original'],
                 kde=True, color=color, label='Female', ax=axes[1])
    sns.histplot(cluster_data[cluster_data['Gender_Male'] == 1]['Annual_Income (£K)_original'],
                 kde=True, color='gray', label='Male', ax=axes[1])
    axes[1].set_title(f'Annual Income Distribution - Cluster {cluster_label}')
    axes[1].legend()

    # Plot Spending Score distribution, divided by Gender_Male
    sns.histplot(cluster_data[cluster_data['Gender_Male'] == 0]['Spending_Score_original'],
                 kde=True, color=color, label='Female', ax=axes[2])
    sns.histplot(cluster_data[cluster_data['Gender_Male'] == 1]['Spending_Score_original'],
                 kde=True, color='gray', label='Male', ax=axes[2])
    axes[2].set_title(f'Spending Score Distribution - Cluster {cluster_label}')
    axes[2].legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)

    # Gender Pie Chart
    st.subheader(f"Gender Distribution - Cluster {cluster_label}")
    gender_counts = cluster_data["Gender_Male"].value_counts()
    labels = ["Male", "Female"]
    colors = ["skyblue", "lightcoral"]
    fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
    ax_pie.pie(gender_counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    ax_pie.set_title(f"Gender Distribution - {cluster_type} {cluster_label}")
    ax_pie.axis("equal")  # Ensures pie chart is circular
    st.pyplot(fig_pie)

    # Display additional analysis (e.g., descriptive statistics)
    st.write(f"Cluster {cluster_label} Statistics:")
    st.write(cluster_data[['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']].describe())

    # Add a separator for better readability
    st.write("-" * 50)

# Ensure that the required columns exist in df
features = ['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']
clusters = ['Cluster_gmm', 'Cluster_k']

# Validate column existence
missing_columns = [col for col in features + clusters if col not in df.columns]
if missing_columns:
    st.error(f"Error: The following required columns are missing from the DataFrame: {missing_columns}")
    st.stop()

# Title
st.title("📊 Feature Distribution Across Clusters")

# Loop through each feature and display its GMM and K-Means boxplots side by side
for feature in features:
    st.subheader(f"📌 {feature} Distribution Across Clusters")

    # Create two columns for side-by-side visualization
    col1, col2 = st.columns(2)

    # Plot GMM Feature Distribution
    with col1:
        st.subheader("GMM Cluster")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.boxplot(x=df['Cluster_gmm'], y=df[feature], palette="husl", ax=ax1)
        ax1.set_xlabel("GMM Cluster")
        ax1.set_ylabel(feature)
        st.pyplot(fig1)
        plt.close(fig1)

    # Plot K-Means Feature Distribution
    with col2:
        st.subheader("K-Means Cluster")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.boxplot(x=df['Cluster_k'], y=df[feature], palette="husl", ax=ax2)
        ax2.set_xlabel("K-Means Cluster")
        ax2.set_ylabel(feature)
        st.pyplot(fig2)
        plt.close(fig2)

