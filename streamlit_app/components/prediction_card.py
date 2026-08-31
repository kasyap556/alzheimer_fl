import streamlit as st
import pandas as pd
import plotly.express as px

def render_prediction_card(prediction_response: dict):
    st.markdown("""
        <style>
        .pred-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .pred-title {
            font-size: 1.5rem;
            color: #31333F;
            margin-bottom: 0px;
        }
        .pred-label {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-top: 0px;
        }
        .pred-conf {
            font-size: 1.2rem;
            color: #555;
        }
        </style>
    """, unsafe_allow_html=True)
    
    pred_label = prediction_response.get("predicted_label", "Unknown")
    confidence = prediction_response.get("confidence", 0.0)
    
    st.markdown(f"""
    <div class="pred-card">
        <p class="pred-title">Predicted Diagnosis</p>
        <p class="pred-label">{pred_label}</p>
        <p class="pred-conf">Confidence: {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if "class_probabilities" in prediction_response:
        probs = prediction_response["class_probabilities"]
        
        df = pd.DataFrame({
            "Class": list(probs.keys()),
            "Probability": list(probs.values())
        })
        
        df['Class'] = df['Class'].str.replace('_', ' ').str.title()
        
        df['Color'] = df['Class'].apply(lambda x: 'Predicted' if x.lower() == pred_label.lower() else 'Other')
        
        fig = px.bar(
            df, 
            y='Class', 
            x='Probability', 
            orientation='h',
            color='Color',
            color_discrete_map={'Predicted': '#1f77b4', 'Other': '#d3d3d3'},
            range_x=[0, 1]
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Probability",
            yaxis_title="",
            margin=dict(l=0, r=0, t=30, b=0),
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
