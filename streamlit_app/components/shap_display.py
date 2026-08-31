import streamlit as st
import base64
import pandas as pd

def render_shap(shap_result: dict):
    st.subheader("SHAP Feature Attributions")
    
    if "waterfall_plot_base64" in shap_result:
        b64_img = shap_result["waterfall_plot_base64"]
        if b64_img.startswith("data:image"):
            b64_img = b64_img.split(",")[1]
            
        try:
            image_bytes = base64.b64decode(b64_img)
            st.image(image_bytes, caption="SHAP Waterfall Plot", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to decode SHAP plot: {e}")
            
    if "feature_attributions" in shap_result:
        st.markdown("#### Feature Contributions")
        
        attrs = shap_result["feature_attributions"]
        data = []
        for a in attrs:
            direction_emoji = "🔴" if a.get("direction") == "supports_diagnosis" else "🔵"
            data.append({
                "Feature": a.get("feature_name"),
                "Value": a.get("feature_value"),
                "SHAP Value": a.get("shap_value"),
                "Direction": direction_emoji
            })
            
        df = pd.DataFrame(data)
        
        def color_shap(val):
            if isinstance(val, float):
                if val > 0:
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                elif val < 0:
                    return 'background-color: rgba(0, 0, 255, 0.2)'
            return ''

        styled_df = df.style.map(color_shap, subset=['SHAP Value'])
        st.dataframe(styled_df, use_container_width=True)
        
    st.caption("SHAP values explain how much each clinical feature contributed to pushing the prediction from the base value to the final output. 🔴 Supports the prediction, 🔵 opposes it.")
