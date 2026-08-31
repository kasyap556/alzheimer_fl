# =============================================================================
# federated_core/server/flower_server.py
# ----------------------------------------
# Flower FL server implementation with FedAvg strategy.
#
# Uses the Flower 1.x API:
#   - server_fn(context: Context) factory returning ServerAppComponents
#   - FedAvg strategy with centralized evaluation
#   - ServerApp instance for simulation/deployment
#
# FedAvg Aggregation:
#     w_{t+1} = Σ_k (n_k / n) · w_k^{t+1}
#
# where w_k is client k's model, n_k is client k's sample count,
# and n = Σ_k n_k is the total across all participating clients.
#
# Author: Kasyap (Final-Year Academic Project)
# =============================================================================

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from models import MultimodalAlzheimersModel
from federated_core.client.flower_client import get_parameters
from federated_core.ml.data_loader import load_global_test_set
from federated_core.ml.evaluate import get_centralized_evaluate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLINICAL_FEATURES = 9
NUM_CLASSES = 4
DEFAULT_NUM_ROUNDS = int(os.environ.get("NUM_FL_ROUNDS", "10"))
DEFAULT_NUM_CLIENTS = int(os.environ.get("NUM_FL_CLIENTS", "5"))


# ─────────────────────────────────────────────────────────────────────────────
# Metric Aggregation
# ─────────────────────────────────────────────────────────────────────────────
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate distributed evaluation metrics using weighted average.

    Each client reports (num_examples, metrics_dict). This function
    computes the weighted average of the 'accuracy' metric across
    all participating clients:

        accuracy_global = Σ_k (n_k × acc_k) / Σ_k n_k

    Args:
        metrics: List of (num_examples, metrics_dict) tuples from clients.

    Returns:
        Aggregated metrics dictionary.
    """
    if not metrics:
        return {"accuracy": 0.0}

    # Weighted average of accuracy
    accuracies = [
        num_examples * m.get("accuracy", 0.0)
        for num_examples, m in metrics
    ]
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)

    aggregated: Metrics = {}

    if total_examples > 0:
        aggregated["accuracy"] = sum(accuracies) / total_examples
    else:
        aggregated["accuracy"] = 0.0

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Per-Round Configuration
# ─────────────────────────────────────────────────────────────────────────────
def fit_config_fn(server_round: int) -> Dict[str, str]:
    """
    Dynamic per-round configuration sent to each client's fit() method.

    Strategy: Start with fewer local epochs in early rounds (model is
    still random, no point in overfitting locally), then increase as
    the global model converges.

    Args:
        server_round: Current FL round number (1-indexed).

    Returns:
        Configuration dictionary sent to client.fit(parameters, config).
    """
    local_epochs = int(os.environ.get("LOCAL_EPOCHS", "2"))

    # Ramp up local epochs: 1 epoch for first 3 rounds, then configured value
    if server_round <= 3:
        current_epochs = max(1, local_epochs // 2)
    else:
        current_epochs = local_epochs

    return {
        "local_epochs": str(current_epochs),
        "current_round": str(server_round),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Server Factory Function (Flower 1.x API)
# ─────────────────────────────────────────────────────────────────────────────
def server_fn(context: Context) -> ServerAppComponents:
    """
    Flower server factory function.

    Configures the FedAvg strategy with:
        1. Initial global model parameters
        2. Centralized evaluation on the held-out test set
        3. Distributed metric aggregation
        4. Dynamic per-round client configuration

    Args:
        context: Flower Context with run_config for num-server-rounds, etc.

    Returns:
        ServerAppComponents containing the strategy and config.
    """
    # Read configuration from context
    num_rounds = int(context.run_config.get("num-server-rounds", DEFAULT_NUM_ROUNDS))
    num_clients = int(context.run_config.get("num-clients", DEFAULT_NUM_CLIENTS))

    print(f"\n{'='*60}")
    print(f"  FEDERATED LEARNING SERVER INITIALIZATION")
    print(f"  Rounds: {num_rounds} | Clients: {num_clients}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    # ── 1. Initialize global model parameters ────────────────────────────
    init_model = MultimodalAlzheimersModel(
        num_clinical_features=NUM_CLINICAL_FEATURES,
        num_classes=NUM_CLASSES,
        pretrained_cnn=True,
    )
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))
    print(f"[Server] Initial model parameters: {init_model.count_parameters():,} trainable")

    # ── 2. Load global test set for centralized evaluation ───────────────
    testloader = load_global_test_set(batch_size=64)
    evaluate_fn = get_centralized_evaluate_fn(testloader, DEVICE)

    # ── 3. Configure FedAvg Strategy ─────────────────────────────────────
    strategy = FedAvg(
        # Client selection
        fraction_fit=1.0,                # Use all available clients for training
        fraction_evaluate=0.5,           # Evaluate on 50% of clients each round
        min_fit_clients=num_clients,     # Minimum clients for training
        min_evaluate_clients=max(1, num_clients // 2),
        min_available_clients=num_clients,

        # Model initialization
        initial_parameters=initial_parameters,

        # Evaluation
        evaluate_fn=evaluate_fn,                          # Server-side centralized eval
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate client metrics

        # Per-round client config
        on_fit_config_fn=fit_config_fn,
    )

    # ── 4. Server config ─────────────────────────────────────────────────
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# Flower ServerApp Instance
# ─────────────────────────────────────────────────────────────────────────────
server_app = ServerApp(server_fn=server_fn)
