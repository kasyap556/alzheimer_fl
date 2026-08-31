# =============================================================================
# federated_core/ml/evaluate.py
# -------------------------------
# Evaluation utilities for both local client validation and centralized
# server-side global evaluation in the Flower FL pipeline.
#
# Author: Kasyap (Final-Year Academic Project)
# =============================================================================

from __future__ import annotations

from typing import Dict, List, Tuple, Callable, Any

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
    """
    Evaluate the multimodal model on a dataset.

    Computes loss, accuracy, and per-class metrics.

    Args:
        model:         The MultimodalAlzheimersModel.
        dataloader:    DataLoader yielding (mri, clinical, labels) tuples.
        device:        Computation device.
        class_weights: Optional class weights for loss computation.

    Returns:
        Tuple of (loss, accuracy, metrics_dict):
            loss:     Average cross-entropy loss.
            accuracy: Overall classification accuracy.
            metrics:  Dictionary with per-class accuracy.
    """
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Per-class tracking
    num_classes = 4
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for mri, clinical, labels in dataloader:
            mri = mri.to(device)
            clinical = clinical.to(device)
            labels = labels.to(device)

            logits = model(mri, clinical)  # (B, 4)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Per-class accuracy
            for c in range(num_classes):
                mask = labels == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (predicted[mask] == labels[mask]).sum().item()

    num_batches = max(len(dataloader), 1)
    avg_loss = total_loss / num_batches
    accuracy = correct / max(total, 1)

    # Build metrics dict
    metrics: Dict[str, float] = {
        "accuracy": accuracy,
        "loss": avg_loss,
    }

    class_names = [
        "mild_dementia", "moderate_dementia",
        "non_demented", "very_mild_dementia",
    ]
    for c in range(num_classes):
        if class_total[c] > 0:
            metrics[f"acc_{class_names[c]}"] = float(
                class_correct[c] / class_total[c]
            )
        else:
            metrics[f"acc_{class_names[c]}"] = 0.0

    return avg_loss, accuracy, metrics


def get_centralized_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
) -> Callable:
    """
    Factory function returning a Flower-compatible centralized evaluate_fn.

    The returned function is called by the Flower server after each
    aggregation round to evaluate the global model on the held-out
    test set.

    Flower evaluate_fn signature:
        evaluate(server_round, parameters, config) -> (loss, metrics)

    Args:
        testloader: DataLoader for the global test set.
        device:     Computation device.

    Returns:
        A callable with the Flower evaluate_fn signature.
    """
    import os
    import sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from models import MultimodalAlzheimersModel

    def centralized_evaluate(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the aggregated global model on the centralized test set.

        Args:
            server_round: Current FL round number.
            parameters:   List of NumPy arrays (global model weights).
            config:       Configuration dict (unused here).

        Returns:
            (loss, metrics_dict) — Flower expects this exact signature.
        """
        # Import here to avoid circular imports
        from federated_core.client.flower_client import set_parameters

        # Create a fresh model and load the aggregated parameters
        model = MultimodalAlzheimersModel(
            num_clinical_features=9,
            num_classes=4,
            pretrained_cnn=False,
        ).to(device)

        set_parameters(model, parameters)

        # Evaluate
        loss, accuracy, metrics = evaluate(model, testloader, device)

        print(
            f"[Server] Round {server_round} — "
            f"Global Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
        )

        return loss, metrics

    return centralized_evaluate
