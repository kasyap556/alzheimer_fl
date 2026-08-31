from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    mri_image_base64: str = Field(..., description="Base64 encoded PNG or JPG MRI image")
    clinical_features: List[float] = Field(..., min_length=9, max_length=9, description="9 clinical features: MMSE, CDR, Age, EDUC, nWBV, eTIV, ASF, SES, Gender")
    feature_names: Optional[List[str]] = Field(None, description="Optional custom feature names")
    include_xai: bool = Field(True, description="Whether to include GradCAM and SHAP explanations")

class ClassProbabilities(BaseModel):
    mild_dementia: float
    moderate_dementia: float
    non_demented: float
    very_mild_dementia: float

class GradCAMResult(BaseModel):
    target_class: str
    target_layer: str
    overlay_image_base64: str

class SHAPAttribution(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float
    abs_importance: float
    direction: str
    importance_rank: int

class SHAPResult(BaseModel):
    target_class: str
    base_value: float
    feature_attributions: List[SHAPAttribution]
    waterfall_plot_base64: str

class ExplainabilityResult(BaseModel):
    gradcam: GradCAMResult
    shap: SHAPResult

class PredictionResponse(BaseModel):
    predicted_class_index: int
    predicted_label: str
    confidence: float
    class_probabilities: ClassProbabilities
    explainability: Optional[ExplainabilityResult] = None

class HealthResponse(BaseModel):
    status: str
    message: str

class ModelInfoResponse(BaseModel):
    model_name: str
    parameter_count: int
    class_labels: List[str]
    feature_names: List[str]
