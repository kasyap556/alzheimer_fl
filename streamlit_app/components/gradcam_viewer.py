import streamlit as st
import base64

def render_gradcam(gradcam_result: dict):
    st.subheader("Grad-CAM Analysis")
    
    if "overlay_image_base64" in gradcam_result:
        b64_img = gradcam_result["overlay_image_base64"]
        if b64_img.startswith("data:image"):
            b64_img = b64_img.split(",")[1]
            
        try:
            image_bytes = base64.b64decode(b64_img)
            st.image(image_bytes, caption=f"Target: {gradcam_result.get('target_class', 'Unknown')} | Layer: {gradcam_result.get('target_layer', 'Unknown')}", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to decode Grad-CAM image: {e}")
            
    st.caption("Grad-CAM highlights the regions of the MRI scan that were most important for the model's prediction. Red areas indicate higher importance.")
