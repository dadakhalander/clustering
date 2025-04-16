import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load Pretrained Artifacts ----
model = joblib.load("best_rf.pkl")               # Trained Random Forest model
X_train = joblib.load("X_train.pkl")             # Features used to train the model
cluster_k_info = joblib.load("cluster_k_info.pkl")  # Dictionary of clustered data

# ---- Function to Analyze New Customer ----
def analyze_new_customer(new_data, model, X_train, cluster_info):
    new_customer = pd.DataFrame([new_data])
    new_customer = new_customer[X_train.columns]  # Ensure same column order

    # Convert binary columns
    new_customer['Gender_Female'] = new_customer['Gender_Female'].astype(int)
    new_customer['Gender_Male'] = new_customer['Gender_Male'].astype(int)

    # Predict cluster
    predicted_cluster = model.predict(new_customer)[0]
    st.subheader(f"🔍 Predicted Cluster: {predicted_cluster}")

    similar_customers = cluster_info[predicted_cluster].copy()
    similar_customers['Gender_Female'] = similar_customers['Gender_Female'].astype(int)
    similar_customers['Gender_Male'] = similar_customers['Gender_Male'].astype(int)

    # Cosine similarity
    sims = cosine_similarity(similar_customers[X_train.columns], new_customer)
    most_similar_index = sims.argmax()
    most_similar_customer = similar_customers.iloc[most_similar_index]

    st.subheader("👤 Most Similar Customer in Cluster:")
    st.dataframe(most_similar_customer[X_train.columns])

    # Radar Chart
    cluster_mean = similar_customers[X_train.columns].mean()

    fig = go.Figure(data=[
        go.Scatterpolar(r=cluster_mean,
                        theta=X_train.columns,
                        name='Cluster Mean',
                        line=dict(color='blue')),
        go.Scatterpolar(r=new_customer.values[0],
                        theta=X_train.columns,
                        name='New Customer',
                        line=dict(color='red'))
    ])
    fig.update_layout(title=f'🧭 Comparison: New Customer vs Cluster {predicted_cluster}',
                      polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig)

# ---- Streamlit App UI ----
st.set_page_config(page_title="Customer Cluster Prediction", layout="centered")
st.title("🔎 Customer Cluster Analysis & Visualization")

# Input form
with st.form(key='customer_form'):
    age = st.number_input('🧓 Age', min_value=0, max_value=100, value=32)
    income = st.number_input('💰 Annual Income (£K)', min_value=0, max_value=500, value=70)
    spending_score = st.number_input('🛍️ Spending Score', min_value=0, max_value=100, value=85)
    gender = st.radio('⚧ Gender', ['Female', 'Male'], index=0)

    submitted = st.form_submit_button("Analyze")

# Submit handler
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
