# =============================================================================
# alzheimer_fl/models/__init__.py
# ---------------------------------
# Package initializer for the models module.
# Exports the multimodal Alzheimer's prediction model and its sub-components.
# =============================================================================

from .multimodal_model import (
    MultimodalAlzheimersModel,
    CNNBranch,
    MLPBranch,
    FusionClassifier,
)

__all__ = [
    "MultimodalAlzheimersModel",
    "CNNBranch",
    "MLPBranch",
    "FusionClassifier",
]
