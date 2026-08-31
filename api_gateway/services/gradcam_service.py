import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
from PIL import Image
import io

class GradCAMService:
    def __init__(self):
        self.gradients = None
        self.activations = None

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _save_activation(self, module, input, output):
        self.activations = output

    def generate(self, model, mri_tensor, clinical_tensor, target_class: int) -> np.ndarray:
        """Generates the Grad-CAM heatmap."""
        self.gradients = None
        self.activations = None
        
        target_layer = model.get_cnn_target_layer()
        
        # Register hooks
        handle_forward = target_layer.register_forward_hook(self._save_activation)
        handle_backward = target_layer.register_full_backward_hook(self._save_gradient)
        
        model.zero_grad()
        logits = model(mri_tensor, clinical_tensor)
        
        target = logits[0, target_class]
        target.backward()
        
        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients or activations.")
            
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam

    def overlay_and_encode(self, mri_tensor: torch.Tensor, heatmap: np.ndarray, alpha: float = 0.4) -> str:
        """Overlays heatmap on MRI and returns base64 string."""
        # Convert MRI to 2D numpy array [0, 255]
        mri_2d = mri_tensor[0, 0].cpu().numpy()
        mri_2d = (mri_2d - mri_2d.min()) / (mri_2d.max() - mri_2d.min() + 1e-8)
        mri_2d = np.uint8(255 * mri_2d)
        mri_rgb = cv2.cvtColor(mri_2d, cv2.COLOR_GRAY2RGB)
        
        # Apply colormap to heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(heatmap_colored, alpha, mri_rgb, 1 - alpha, 0)
        
        # Encode to base64
        overlay_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="PNG")
        encoded_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{encoded_str}"
