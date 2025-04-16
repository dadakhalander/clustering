import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

def analyze_new_customer(new_data, model, X_train, cluster_info):
    """
    Analyzes new customer: predicts cluster, finds similar customer, plots comparison.

    Args:
        new_data (dict): Input values for new customer.
        model: Trained classifier (e.g., RandomForestClassifier).
        X_train (DataFrame): Feature training data used for model.
        cluster_info (dict): Dictionary containing original data split by cluster.
    """

    # Convert to DataFrame and align columns
    new_customer = pd.DataFrame([new_data])
    new_customer = new_customer[X_train.columns]

    # Handle Gender Features as Binary (0 or 1)
    new_customer['Gender_Female'] = new_customer['Gender_Female'].astype(int)
    new_customer['Gender_Male'] = new_customer['Gender_Male'].astype(int)

    # Predict the cluster
    predicted_cluster = model.predict(new_customer)[0]
    st.write(f"🔍 **Predicted Cluster**: {predicted_cluster}")

    # Get customers from predicted cluster
    similar_customers = cluster_info[predicted_cluster]

    # Handle Gender Features in similar customers as Binary (0 or 1)
    similar_customers['Gender_Female'] = similar_customers['Gender_Female'].astype(int)
    similar_customers['Gender_Male'] = similar_customers['Gender_Male'].astype(int)

    # Compute cosine similarity
    sims = cosine_similarity(similar_customers[X_train.columns], new_customer)
    most_similar_index = sims.argmax()
    most_similar_customer = similar_customers.iloc[most_similar_index]

    st.write("👤 **Most Similar Customer in Predicted Cluster:**")
    st.write(most_similar_customer[X_train.columns])

    # Get average of cluster
    cluster_mean = similar_customers[X_train.columns].mean()

    # Radar chart comparison
    trace1 = go.Scatterpolar(r=cluster_mean,
                             theta=X_train.columns,
                             name='Cluster Mean',
                             line=dict(color='blue'))
    
    trace2 = go.Scatterpolar(r=new_customer.values[0],
                             theta=X_train.columns,
                             name='New Customer',
                             line=dict(color='red'))

    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(title=f'🧭 **Comparison: New Customer vs Cluster {predicted_cluster}**',
                      polar=dict(radialaxis=dict(visible=True)))
    
    st.plotly_chart(fig)

# Streamlit UI
st.title("Customer Cluster Analysis")

# Input for new customer data
age = st.number_input('Age', min_value=0, max_value=100, value=32)
income = st.number_input('Annual Income (£K)', min_value=0, max_value=500, value=70)
spending_score = st.number_input('Spending Score', min_value=0, max_value=100, value=85)
gender_female = st.checkbox('Female', value=True)
gender_male = not gender_female  # Ensures only one gender can be selected

# Button to analyze
if st.button('Analyze New Customer'):
    new_data = {
        'Age_original': age,
        'Annual_Income (£K)_original': income,
        'Spending_Score_original': spending_score,
        'Gender_Female': gender_female,
        'Gender_Male': gender_male
    }
    
    # Assuming best_rf is the trained model and cluster_k_info contains the cluster data
    analyze_new_customer(new_data, best_rf, X_train, cluster_k_info)
