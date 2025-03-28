import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load dataset (Modify to your actual dataset loading method)
@st.cache_data
def load_data():
    return pd.read_csv("clustering_results.csv")  # Replace with actual file path

df = load_data()

# Streamlit App Title
st.title("📊 Cluster Analysis: Spending, Income & Gender Distribution")

# Select cluster type (GMM or K-Means)
cluster_type = st.sidebar.selectbox("Select Cluster Type", ["Cluster_gmm", "Cluster_k"])

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

# Boxplots for feature distribution across GMM clusters
st.title("📊 Feature Distribution Across GMM Clusters")

features = ['Age_original', 'Annual_Income (£K)_original', 'Spending_Score_original']
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['Cluster_gmm'], y=df[feature], palette="husl")
    plt.title(f"{feature} Distribution Across GMM Clusters")
    plt.xlabel("GMM Cluster")
    plt.ylabel(feature)
    st.pyplot(plt)  # Display the plot in Streamlit
    plt.clf()  # Clear the figure after displaying to prevent overlap

# GMM Clusters Analysis (mean Spending Score)
cluster_spending_gmm = df.groupby('Cluster_gmm')['Spending_Score_original'].mean().sort_values(ascending=False)

# Display cluster ranking for GMM
st.title("📊 Cluster Ranking by Mean Spending Score (GMM Clusters)")
st.write("Cluster Ranking by Mean Spending Score:")
for cluster, mean_spending in cluster_spending_gmm.items():
    st.write(f"Cluster {cluster}: {mean_spending:.2f}")
