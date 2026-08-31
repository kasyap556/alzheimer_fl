import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Overview Dashboard")

if not st.session_state.get('predictions'):
    st.info("No predictions yet. Head over to the **New Prediction** page to analyze a case.")
else:
    preds = st.session_state['predictions']
    total_preds = len(preds)
    
    labels = [p['response']['predicted_label'] for p in preds]
    confidences = [p['response']['confidence'] for p in preds]
    
    most_common = max(set(labels), key=labels.count) if labels else "N/A"
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", total_preds)
    col2.metric("Most Common Diagnosis", most_common)
    col3.metric("Average Confidence", f"{avg_conf:.1%}")
    
    st.markdown("---")
    
    st.subheader("Diagnosis Distribution")
    label_counts = pd.Series(labels).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
    ax.axis('equal')
    st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("Recent Predictions (Last 10)")
    recent = preds[-10:][::-1]
    
    data = []
    for r in recent:
        data.append({
            "Timestamp": r.get('timestamp', 'N/A'),
            "Predicted Class": r['response']['predicted_label'],
            "Confidence": f"{r['response']['confidence']:.1%}"
        })
    st.dataframe(pd.DataFrame(data), use_container_width=True)
