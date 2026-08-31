import shap
import numpy as np
import torch
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any

class SHAPService:
    def __init__(self, model, background_data: np.ndarray, feature_names: List[str], device: torch.device):
        self.model = model
        self.device = device
        self.feature_names = feature_names
        
        # Use background data directly (avoid expensive kmeans)
        self.background_data = background_data

    def explain(self, mri_tensor: torch.Tensor, clinical_array: np.ndarray, target_class: int, nsamples: int = 100) -> Dict[str, Any]:
        """Generates SHAP explanations for clinical features."""
        
        def predict_fn(clinical_features_np):
            clinical_tensor = torch.tensor(clinical_features_np, dtype=torch.float32).to(self.device)
            mri_batch = mri_tensor.repeat(clinical_tensor.shape[0], 1, 1, 1).to(self.device)
            
            with torch.no_grad():
                logits = self.model(mri_batch, clinical_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            return probs[:, target_class]
            
        explainer = shap.KernelExplainer(predict_fn, self.background_data)
        
        shap_values = explainer.shap_values(clinical_array, nsamples=nsamples)
        expected_value = explainer.expected_value
        
        # Extract 1D array of SHAP values
        if isinstance(shap_values, list):
            sv = shap_values[target_class][0]
        else:
            sv = shap_values[0]
            
        if isinstance(expected_value, (list, np.ndarray)):
            ev = expected_value[target_class] if isinstance(expected_value, list) else expected_value[0]
        else:
            ev = expected_value
            
        attributions = []
        for i, name in enumerate(self.feature_names):
            val = float(clinical_array[0, i])
            s_val = float(sv[i])
            attributions.append({
                "feature_name": name,
                "feature_value": val,
                "shap_value": s_val,
                "abs_importance": abs(s_val),
                "direction": "positive" if s_val >= 0 else "negative",
                "importance_rank": 0 # to be calculated
            })
            
        # Sort by absolute importance
        attributions.sort(key=lambda x: x["abs_importance"], reverse=True)
        for i, attr in enumerate(attributions):
            attr["importance_rank"] = i + 1
            
        return {
            "base_value": float(ev),
            "feature_attributions": attributions,
            "shap_values": sv
        }

    def generate_waterfall_base64(self, expected_value: float, shap_values: np.ndarray, features: np.ndarray) -> str:
        """Generates SHAP waterfall plot as base64."""
        plt.figure(figsize=(10, 6))
        
        explanation = shap.Explanation(
            values=shap_values, 
            base_values=expected_value, 
            data=features[0], 
            feature_names=self.feature_names
        )
        
        shap.plots.waterfall(explanation, show=False)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        encoded_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded_str}"
