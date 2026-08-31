import streamlit as st
import requests
import base64
import time
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.prediction_card import render_prediction_card
from components.gradcam_viewer import render_gradcam
from components.shap_display import render_shap

st.title("🔬 New Prediction")

if 'predictions' not in st.session_state:
    st.session_state['predictions'] = []

col1, col2 = st.columns(2)

with col1:
    st.header("1. MRI Upload")
    uploaded_file = st.file_uploader("Choose an MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded MRI", use_container_width=True)

with col2:
    st.header("2. Clinical Features")
    
    mmse = st.number_input("MMSE (Mini-Mental State Exam)", min_value=0, max_value=30, value=28, step=1)
    cdr = st.select_slider("CDR (Clinical Dementia Rating)", options=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], value=0.0)
    age = st.slider("Age", min_value=60, max_value=100, value=72)
    educ = st.number_input("Years of Education (EDUC)", min_value=4, max_value=23, value=16)
    nwbv = st.number_input("Normalized Whole Brain Volume (nWBV)", min_value=0.6, max_value=0.85, value=0.75, step=0.01, format="%.3f")
    etiv = st.number_input("Estimated Total Intracranial Volume (eTIV)", min_value=1100, max_value=2000, value=1500)
    asf = st.number_input("Atlas Scaling Factor (ASF)", min_value=0.8, max_value=1.6, value=1.2, step=0.01, format="%.2f")
    ses = st.slider("Socioeconomic Status (SES)", min_value=1, max_value=5, value=2)
    gender_str = st.radio("Gender", ["Female", "Male"])
    gender = 1.0 if gender_str == "Male" else 0.0

st.markdown("---")

submit = st.button("Generate Prediction", type="primary", use_container_width=True)

if submit:
    if uploaded_file is None:
        st.error("Please upload an MRI image.")
    else:
        with st.spinner("Analyzing data..."):
            image_bytes = uploaded_file.getvalue()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            features = [float(mmse), float(cdr), float(age), float(educ), float(nwbv), float(etiv), float(asf), float(ses), float(gender)]
            feature_names = ["MMSE", "CDR", "Age", "EDUC", "nWBV", "eTIV", "ASF", "SES", "Gender"]
            
            payload = {
                "mri_image_base64": base64_image,
                "clinical_features": features,
                "feature_names": feature_names,
                "include_xai": True
            }
            
            try:
                response = requests.post("http://localhost:8000/api/v1/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    
                    st.session_state['predictions'].append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "request": payload,
                        "response": result
                    })
                    
                    st.success("Prediction generated successfully!")
                    
                    render_prediction_card(result)
                    
                    st.markdown("### Explainability (XAI)")
                    xai_col1, xai_col2 = st.columns(2)
                    
                    with xai_col1:
                        if "explainability" in result and "gradcam" in result["explainability"]:
                            render_gradcam(result["explainability"]["gradcam"])
                        else:
                            st.info("Grad-CAM results not available.")
                            
                    with xai_col2:
                        if "explainability" in result and "shap" in result["explainability"]:
                            render_shap(result["explainability"]["shap"])
                        else:
                            st.info("SHAP results not available.")
                            
                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to API: {e}")
