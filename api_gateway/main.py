import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api_gateway.models.schemas import (
    PredictionRequest, PredictionResponse, ClassProbabilities,
    GradCAMResult, SHAPAttribution, SHAPResult, ExplainabilityResult,
    HealthResponse, ModelInfoResponse
)
from api_gateway.services.model_service import ModelService
from api_gateway.services.gradcam_service import GradCAMService
from api_gateway.services.shap_service import SHAPService

# Import synthetic data generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from federated_core.ml.synthetic_features import SyntheticClinicalGenerator
except ImportError:
    pass

model_service = ModelService()
gradcam_service = GradCAMService()
shap_service = None

FEATURE_NAMES = ['MMSE', 'CDR', 'Age', 'EDUC', 'nWBV', 'eTIV', 'ASF', 'SES', 'Gender']

@asynccontextmanager
async def lifespan(app: FastAPI):
    global shap_service
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = os.environ.get('MODEL_WEIGHTS_PATH', 'saved_models/global_model.pth')
    
    # Ensure dir exists for safety
    os.makedirs(os.path.dirname(weights_path) if os.path.dirname(weights_path) else '.', exist_ok=True)
    
    # Load model
    model_service.load_model(weights_path, device)
    print("✓ Model loaded")
    
    # Initialize SHAP background data (simplified - use random data to avoid slow synthetic generation)
    try:
        # Use simple random data instead of synthetic generation for faster startup
        background_data = np.random.randn(20, 9).astype(np.float32)
        print("✓ SHAP background data created")
    except Exception as e:
        print(f"Warning: Could not create background data. {e}")
        background_data = np.random.rand(20, 9)
        
    try:
        shap_service = SHAPService(
            model=model_service.model,
            background_data=background_data,
            feature_names=FEATURE_NAMES,
            device=device
        )
        print("✓ SHAP service initialized")
    except Exception as e:
        print(f"Warning: SHAP service initialization failed: {e}. API will work without SHAP explanations.")
        shap_service = None
    
    yield
    # Cleanup
    pass

app = FastAPI(title="Alzheimer's FL API Gateway", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        mri_tensor = model_service.decode_mri_base64(request.mri_image_base64)
        clinical_tensor = torch.tensor([request.clinical_features], dtype=torch.float32)
        clinical_np = clinical_tensor.numpy()
        
        logits, probs, class_idx, class_label, confidence = model_service.predict(mri_tensor, clinical_tensor)
        
        class_probs = ClassProbabilities(
            mild_dementia=float(probs[0]),
            moderate_dementia=float(probs[1]),
            non_demented=float(probs[2]),
            very_mild_dementia=float(probs[3])
        )
        
        explainability = None
        if request.include_xai:
            # GradCAM
            heatmap = gradcam_service.generate(model_service.model, mri_tensor.to(model_service.device), clinical_tensor.to(model_service.device), class_idx)
            overlay = gradcam_service.overlay_and_encode(mri_tensor, heatmap)
            
            gradcam_result = GradCAMResult(
                target_class=class_label,
                target_layer="layer4",
                overlay_image_base64=overlay
            )
            
            # SHAP
            shap_res = shap_service.explain(mri_tensor, clinical_np, class_idx)
            waterfall = shap_service.generate_waterfall_base64(shap_res['base_value'], shap_res['shap_values'], clinical_np)
            
            feature_attrs = [SHAPAttribution(**attr) for attr in shap_res['feature_attributions']]
            
            shap_result_obj = SHAPResult(
                target_class=class_label,
                base_value=shap_res['base_value'],
                feature_attributions=feature_attrs,
                waterfall_plot_base64=waterfall
            )
            
            explainability = ExplainabilityResult(gradcam=gradcam_result, shap=shap_result_obj)
            
        return PredictionResponse(
            predicted_class_index=class_idx,
            predicted_label=class_label,
            confidence=confidence,
            class_probabilities=class_probs,
            explainability=explainability
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", message="API Gateway is running")

@app.get("/api/v1/model-info", response_model=ModelInfoResponse)
async def model_info():
    num_params = sum(p.numel() for p in model_service.model.parameters()) if model_service.model else 0
    return ModelInfoResponse(
        model_name="MultimodalAlzheimersModel",
        parameter_count=num_params,
        class_labels=list(model_service.CLASS_LABELS),
        feature_names=FEATURE_NAMES
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
