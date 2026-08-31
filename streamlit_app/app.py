import streamlit as st
import requests

def main():
    st.set_page_config(page_title='🧠 Alzheimer AI Dashboard', layout='wide')
    
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = []

    st.sidebar.title("🧠 Alzheimer AI Dashboard")
    st.sidebar.markdown("A clinical dashboard for federated Alzheimer's disease diagnosis using MRI and clinical data.")
    
    # Check API status
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
        if response.status_code == 200:
            st.sidebar.success("Backend API: Online")
        else:
            st.sidebar.error(f"Backend API: Error ({response.status_code})")
    except requests.exceptions.RequestException:
        st.sidebar.error("Backend API: Offline")

    st.title("Welcome to the Alzheimer AI Dashboard")
    st.markdown("""
    This dashboard provides clinicians with an AI-assisted diagnostic tool for Alzheimer's disease. 
    It combines deep learning on MRI scans with machine learning on clinical features.
    
    ### Key Features
    - **Multimodal Prediction**: Uses both imaging and clinical data.
    - **Explainable AI (XAI)**: Includes Grad-CAM for MRI visual explanations and SHAP for clinical feature importance.
    - **Federated Learning Ready**: Model trained across multiple institutions without sharing raw patient data.
    
    ### Architecture Overview
    - **Frontend**: Streamlit dashboard (this app).
    - **Backend**: FastAPI serving predictions.
    - **Models**: PyTorch-based CNN for images and XGBoost/LightGBM for clinical features.
    """)

if __name__ == "__main__":
    main()
