# =============================================================================
# federated_core/client/flower_client.py
# ----------------------------------------
# Flower FL client implementation for the Alzheimer's multimodal model.
#
# Uses the Flower 1.x API:
#   - NumPyClient for parameter exchange (PyTorch ↔ NumPy)
#   - client_fn(context: Context) factory for ClientApp
#   - ClientApp instance for simulation/deployment
#
# Author: Kasyap (Final-Year Academic Project)
# =============================================================================

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models import MultimodalAlzheimersModel
from federated_core.ml.data_loader import (
    AlzheimerMultimodalDataset,
    partition_data,
    get_train_transforms,
    compute_class_weights,
)
from federated_core.ml.train import train_local
from federated_core.ml.evaluate import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Serialization: PyTorch ↔ NumPy
# ─────────────────────────────────────────────────────────────────────────────
def get_parameters(net: nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as a list of NumPy arrays.

    Serializes the PyTorch state_dict for Flower's parameter transport.
    Each tensor in the state_dict is converted to a NumPy array.

    Args:
        net: PyTorch model.

    Returns:
        List of NumPy arrays, one per parameter tensor.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Load NumPy array parameters back into a PyTorch model.

    Reconstructs the state_dict from the list of NumPy arrays received
    from the Flower server after aggregation.

    The FedAvg aggregation formula:
        w_{t+1} = Σ_k (n_k / n) · w_k^{t+1}

    produces a weighted average of client parameters, which this function
    loads into the local model.

    Args:
        net:        PyTorch model to update.
        parameters: List of NumPy arrays from the server.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(np.copy(v)) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLINICAL_FEATURES = 9
NUM_CLASSES = 4
NUM_PARTITIONS = int(os.environ.get("NUM_FL_CLIENTS", "5"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "2"))
LEARNING_RATE = float(os.environ.get("FL_LEARNING_RATE", "1e-3"))
BATCH_SIZE = int(os.environ.get("FL_BATCH_SIZE", "32"))

# Dataset path
_TRAIN_DIR = os.path.join(
    _PROJECT_ROOT, "archive (2)", "AugmentedAlzheimerDataset"
)


# ─────────────────────────────────────────────────────────────────────────────
# Flower NumPyClient
# ─────────────────────────────────────────────────────────────────────────────
class AlzheimerFlowerClient(NumPyClient):
    """
    Flower federated learning client for the Alzheimer's multimodal model.

    Implements the three core methods required by Flower's NumPyClient:
        1. get_parameters  — Return current model weights as NumPy arrays
        2. fit             — Train locally and return updated weights
        3. evaluate        — Evaluate locally and return metrics
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        device: torch.device,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            model:         MultimodalAlzheimersModel instance.
            trainloader:   Training DataLoader for this client's partition.
            valloader:     Validation DataLoader for this client's partition.
            device:        Computation device.
            class_weights: Optional class weights for loss balancing.
        """
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.class_weights = class_weights

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Return current model parameters as NumPy arrays."""
        return get_parameters(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """
        Receive global parameters, train locally, return updated parameters.

        FL Training Protocol:
            1. Server sends global weights → client loads them
            2. Client trains on local data for `local_epochs` epochs
            3. Client sends back updated weights + sample count + metrics

        Args:
            parameters: Global model parameters from the server.
            config:     Per-round configuration from the server.

        Returns:
            Tuple of (updated_parameters, num_train_samples, metrics_dict).
        """
        # 1. Load global parameters into local model
        set_parameters(self.model, parameters)

        # 2. Read config (server may override local_epochs per round)
        local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        lr = float(config.get("learning_rate", LEARNING_RATE))

        # 3. Train locally
        train_loss = train_local(
            model=self.model,
            dataloader=self.trainloader,
            epochs=local_epochs,
            device=self.device,
            lr=lr,
            class_weights=self.class_weights,
        )

        # 4. Return updated parameters
        num_samples = len(self.trainloader.dataset)
        metrics = {"train_loss": float(train_loss)}

        return get_parameters(self.model), num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluate the global model on this client's local validation set.

        Args:
            parameters: Global model parameters from the server.
            config:     Per-round configuration.

        Returns:
            Tuple of (loss, num_val_samples, metrics_dict).
        """
        # Load global parameters
        set_parameters(self.model, parameters)

        # Evaluate on local validation set
        loss, accuracy, metrics = evaluate(
            model=self.model,
            dataloader=self.valloader,
            device=self.device,
            class_weights=self.class_weights,
        )

        num_samples = len(self.valloader.dataset)

        return float(loss), num_samples, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Client Factory Function (Flower 1.x API)
# ─────────────────────────────────────────────────────────────────────────────
# Global dataset reference (loaded once, shared across client_fn calls)
_global_dataset: AlzheimerMultimodalDataset | None = None


def _get_dataset() -> AlzheimerMultimodalDataset:
    """Lazy-load the training dataset (singleton pattern)."""
    global _global_dataset
    if _global_dataset is None:
        _global_dataset = AlzheimerMultimodalDataset(
            image_dir=_TRAIN_DIR,
            transform=get_train_transforms(),
            synthetic_seed=42,
        )
    return _global_dataset


def client_fn(context: Context):
    """
    Flower client factory function.

    Called by the Flower simulation engine to create a client for each
    virtual supernode. Each client receives a unique partition of the
    dataset.

    Args:
        context: Flower Context containing node_config and run_config.
                 context.node_config["partition-id"] identifies this client.

    Returns:
        A Flower Client instance (via NumPyClient.to_client()).
    """
    # Retrieve this client's partition ID
    partition_id = context.node_config["partition-id"]
    num_partitions = context.run_config.get("num-clients", NUM_PARTITIONS)

    print(f"\n[Client {partition_id}] Initializing...")

    # Load dataset and create partition
    dataset = _get_dataset()
    trainloader, valloader = partition_data(
        dataset=dataset,
        num_partitions=num_partitions,
        partition_id=partition_id,
        batch_size=BATCH_SIZE,
    )

    # Compute class weights for loss balancing
    class_weights = compute_class_weights(dataset)

    # Create model
    model = MultimodalAlzheimersModel(
        num_clinical_features=NUM_CLINICAL_FEATURES,
        num_classes=NUM_CLASSES,
        pretrained_cnn=True,
    ).to(DEVICE)

    # Create and return Flower client
    flower_client = AlzheimerFlowerClient(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        device=DEVICE,
        class_weights=class_weights,
    )

    return flower_client.to_client()


# ─────────────────────────────────────────────────────────────────────────────
# Flower ClientApp Instance
# ─────────────────────────────────────────────────────────────────────────────
client_app = ClientApp(client_fn=client_fn)
