import joblib
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Load your trained Random Forest model
best_rf = joblib.load('path_to_your_model_file.pkl')

# Initialize cluster_k_info (replace with actual data)
cluster_k_info = {
    0: pd.DataFrame(...),
    1: pd.DataFrame(...),
    2: pd.DataFrame(...),
    # Add more clusters if needed
}

def analyze_new_customer(new_data, model, X_train, cluster_info):
    """
    Analyzes new customer: predicts cluster, finds similar customer, plots comparison.
    """
    # Same implementation as before...
    # Ensure gender features are handled, predictions are made, and visualizations are plotted

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
    
    analyze_new_customer(new_data, best_rf, X_train, cluster_k_info)
