import streamlit as st
import pandas as pd

st.title("🕰️ Prediction History")

if 'predictions' not in st.session_state or not st.session_state['predictions']:
    st.info("No prediction history available.")
else:
    preds = st.session_state['predictions']
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Clear History", type="secondary"):
            st.session_state['predictions'] = []
            st.rerun()
            
    with col2:
        csv_data = []
        for i, p in enumerate(preds):
            csv_data.append({
                "ID": i + 1,
                "Timestamp": p['timestamp'],
                "Predicted Class": p['response']['predicted_label'],
                "Confidence": p['response']['confidence']
            })
        df = pd.DataFrame(csv_data)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='prediction_history.csv',
            mime='text/csv',
        )
        
    st.markdown("---")
    
    for i, p in enumerate(reversed(preds)):
        idx = len(preds) - i
        with st.expander(f"#{idx} | {p['timestamp']} | {p['response']['predicted_label']} ({p['response']['confidence']:.1%})"):
            st.json(p['response'])
