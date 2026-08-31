# =============================================================================
# federated_core/ml/train.py
# ---------------------------
# Local training loop for federated clients.
#
# Implements a single-epoch training function that processes multimodal
# batches (MRI + clinical features) through the MultimodalAlzheimersModel.
#
# Author: Kasyap (Final-Year Academic Project)
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Train the multimodal model for one epoch on the local client dataset.

    Each batch contains:
        mri:      (B, 1, 224, 224) — MRI images
        clinical: (B, 9)           — clinical feature vectors
        labels:   (B,)             — integer class labels

    The training step:
        1. Forward pass:  logits = model(mri, clinical)
        2. Loss:          L = CrossEntropyLoss(logits, labels)
        3. Backward pass: ∂L/∂θ
        4. Optimizer step: θ ← θ - η·∇L

    Args:
        model:      The MultimodalAlzheimersModel.
        dataloader: DataLoader yielding (mri, clinical, labels) tuples.
        optimizer:  PyTorch optimizer (e.g., Adam, SGD).
        criterion:  Loss function (CrossEntropyLoss with optional class weights).
        device:     Computation device (CPU/GPU).

    Returns:
        Average training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for mri, clinical, labels in dataloader:
        # Move data to device
        mri = mri.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(mri, clinical)  # (B, 4)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def train_local(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    class_weights: torch.Tensor | None = None,
) -> float:
    """
    Complete local training routine for a federated client.

    Trains the model for the specified number of local epochs using
    Adam optimizer and CrossEntropyLoss.

    Args:
        model:         The MultimodalAlzheimersModel.
        dataloader:    Training DataLoader for this client's partition.
        epochs:        Number of local training epochs.
        device:        Computation device.
        lr:            Learning rate for Adam optimizer.
        class_weights: Optional tensor of shape (num_classes,) for
                       handling class imbalance in the loss function.

    Returns:
        Average training loss from the final epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        final_loss = epoch_loss

    return final_loss
