import os
import sys
import torch
import base64
import io
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from models.multimodal_model import MultimodalAlzheimersModel
except ImportError:
    # Try generic import as fallback
    from models import MultimodalAlzheimersModel

class ModelService:
    _instance = None
    CLASS_LABELS = ('Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very Mild Dementia')

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.device = None
            cls._instance.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        return cls._instance

    def load_model(self, weights_path: str, device: torch.device):
        """Loads the MultimodalAlzheimersModel model."""
        self.device = device
        self.model = MultimodalAlzheimersModel(num_clinical_features=9, num_classes=4)
        
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            # Handle both checkpoint dict and raw state_dict formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            print(f"Model weights loaded from {weights_path}")
        else:
            print(f"Warning: Model weights file {weights_path} not found. Using randomly initialized weights.")
            
        self.model.to(self.device)
        self.model.eval()

    def decode_mri_base64(self, base64_str: str) -> torch.Tensor:
        """Decodes a base64 encoded image string to a PyTorch tensor."""
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        tensor = self.transform(image).unsqueeze(0)
        return tensor

    def predict(self, mri_tensor: torch.Tensor, clinical_tensor: torch.Tensor):
        """Runs the model inference."""
        if self.model is None:
            raise ValueError("Model is not loaded.")
            
        mri_tensor = mri_tensor.to(self.device)
        clinical_tensor = clinical_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(mri_tensor, clinical_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        class_idx = int(probs.argmax())
        class_label = self.CLASS_LABELS[class_idx]
        confidence = float(probs[class_idx])
        
        return logits, probs, class_idx, class_label, confidence
