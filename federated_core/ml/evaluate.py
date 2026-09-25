# =============================================================================
# federated_core/ml/evaluate.py
# --------------------------------
# Evaluation utilities for both local client validation and centralized
# server-side global evaluation in the Flower FL pipeline.
# =============================================================================

from __future__ import annotations

from typing import Dict, List, Tuple, Callable, Any
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> Tuple[float, float, Dict[str, float]]:
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    num_classes = 4
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for mri, clinical, labels in dataloader:
            mri = mri.to(device)
            clinical = clinical.to(device)
            labels = labels.to(device)

            logits = model(mri, clinical)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for c in range(num_classes):
                mask = labels == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (predicted[mask] == labels[mask]).sum().item()

    num_batches = max(len(dataloader), 1)
    avg_loss = total_loss / num_batches
    accuracy = correct / max(total, 1)

    metrics: Dict[str, float] = {
        "accuracy": accuracy,
        "loss": avg_loss,
    }

    class_names = [
        "mild_dementia", "moderate_dementia",
        "non_demented", "very_mild_dementia",
    ]
    for c in range(num_classes):
        metrics[f"acc_{class_names[c]}"] = (
            float(class_correct[c] / class_total[c])
            if class_total[c] > 0 else 0.0
        )

    return avg_loss, accuracy, metrics


def get_centralized_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
) -> Callable:
    """
    Return Flower's centralized evaluate_fn.

    The callback also persists the exact aggregated parameters supplied by
    Flower. This avoids accidentally saving a newly initialized model after
    the simulation.
    """
    import sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from models import MultimodalAlzheimersModel
    from federated_core.client.flower_client import set_parameters

    def centralized_evaluate(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        model = MultimodalAlzheimersModel(
            num_clinical_features=9,
            num_classes=4,
            pretrained_cnn=False,
        ).to(device)

        set_parameters(model, parameters)

        loss, accuracy, metrics = evaluate(model, testloader, device)

        output_path = os.environ.get("FL_MODEL_OUTPUT_PATH")
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_clinical_features": 9,
                    "num_classes": 4,
                    "server_round": server_round,
                    "final_accuracy": float(accuracy),
                    "final_loss": float(loss),
                },
                output_path,
            )
            print(f"[Server] Saved aggregated model: {output_path}")

        print(
            f"[Server] Round {server_round} — "
            f"Global Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
        )
        return loss, metrics

    return centralized_evaluate
