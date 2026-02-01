import streamlit as st
import numpy as np
import joblib

# 1. Load the Saved Model
@st.cache_resource
def load_data():
    # Make sure 'tumors_model.pkl' is in the same folder
    return joblib.load('tumor_model.pkl')

data = load_data()

# Page Configuration
st.set_page_config(page_title="Tumor Classifier", page_icon="🧠")

st.title("🧠 Zenitith AI: Tumor Classifier")
st.write("This model was trained using **Manual Gradient Descent** and **Feature Engineering**.")

# 2. Collect Basic Inputs from User
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        size = st.number_input("Tumor Size (cm)", 0.1, 15.0, 2.0)
        age = st.slider("Patient Age", 18, 100, 45)
        is_rough = st.selectbox("Is the Surface Rough?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    
    with col2:
        toughness = st.slider("Toughness Score (1-10)", 1, 10, 5)
        is_hetero = st.selectbox("Is it Heterogeneous?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        growth = st.number_input("Growth Rate (cm/month)", 0.0, 5.0, 0.5)
    
    submit = st.form_submit_button("Analyze Tumor")

# 3. Calculation and Prediction
if submit:
    # YOUR LOGIC: Feature Engineering (Squaring and Interactions)
    size_sq = size ** 2
    size_growth = size * growth
    age_tough = age * toughness
    
    # 9-Feature Array (Following the same order as training!)
    raw_inputs = np.array([size, age, is_rough, toughness, is_hetero, growth, size_sq, size_growth, age_tough])
    
    # Normalization (Using 'means' and 'devs' from your saved data)
    inputs_scaled = (raw_inputs - data['means']) / data['devs']
    
    # Sigmoid Prediction (Logistic Regression)
    z = np.dot(inputs_scaled, data['w']) + data['b']
    probability = 1 / (1 + np.exp(-z))
    
    # 4. Display Results
    st.divider()
    if probability >= 0.5:
        st.error(f"Prediction: **MALIGNANT** - Confidence: {probability*100:.2f}%")
        st.write("⚠️ This indicates a high risk. Please consult a specialist.")
    else:
        st.success(f"Prediction: **BENIGN** - Confidence: {(1-probability)*100:.2f}%")
        st.write("✅ This indicates a low risk (Non-cancerous).")
