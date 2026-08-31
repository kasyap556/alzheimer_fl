# =============================================================================
# federated_core/run_simulation_standalone.py
# --------------------------------------------
# Standalone FL simulation that bypasses Flower's Ray backend.
# Implements FedAvg manually using the existing train/evaluate/data modules.
#
# This is equivalent to the Flower simulation but runs natively on Windows
# without requiring Ray.
#
# Usage:
#   python federated_core/run_simulation_standalone.py --num-clients 3 --num-rounds 5
#
# Author: Kasyap (Final-Year Academic Project)
# =============================================================================

from __future__ import annotations

import os
import sys
import argparse
import time
import copy

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import numpy as np

from models import MultimodalAlzheimersModel
from federated_core.ml.data_loader import (
    AlzheimerMultimodalDataset,
    get_train_transforms,
    partition_data,
    load_global_test_set,
    compute_class_weights,
)
from federated_core.ml.train import train_local
from federated_core.ml.evaluate import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone FL simulation for Alzheimer's prediction (no Ray)"
    )
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(_PROJECT_ROOT, "saved_models"),
    )
    return parser.parse_args()


def get_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    """Load parameters from NumPy arrays into the model."""
    state_dict = model.state_dict()
    for key, param in zip(state_dict.keys(), parameters):
        state_dict[key] = torch.tensor(param)
    model.load_state_dict(state_dict, strict=True)


def fedavg_aggregate(
    client_params: list[list[np.ndarray]],
    client_sizes: list[int],
) -> list[np.ndarray]:
    """
    Federated Averaging: weighted average of client parameters.
    
    Each client's parameters are weighted by their dataset size.
    """
    total_size = sum(client_sizes)
    num_layers = len(client_params[0])

    aggregated = []
    for layer_idx in range(num_layers):
        weighted_sum = np.zeros_like(client_params[0][layer_idx], dtype=np.float64)
        for client_idx, params in enumerate(client_params):
            weight = client_sizes[client_idx] / total_size
            weighted_sum += params[layer_idx].astype(np.float64) * weight
        aggregated.append(weighted_sum.astype(client_params[0][layer_idx].dtype))

    return aggregated


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  ALZHEIMER'S FEDERATED LEARNING SIMULATION (Standalone)")
    print("=" * 60)
    print(f"  Clients:      {args.num_clients}")
    print(f"  Rounds:       {args.num_rounds}")
    print(f"  Local Epochs: {args.local_epochs}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  Learning Rate:{args.lr}")
    print(f"  Device:       {device}")
    print(f"  Output:       {args.output_dir}")
    print("=" * 60)
    print()

    start_time = time.time()

    # ── Load Dataset ────────────────────────────────────────────────────
    print("[1/4] Loading training dataset...")
    train_dataset = AlzheimerMultimodalDataset(
        image_dir=os.path.join(_PROJECT_ROOT, "archive (2)", "AugmentedAlzheimerDataset"),
        transform=get_train_transforms(),
        synthetic_seed=42,
    )
    class_weights = compute_class_weights(train_dataset)

    print("\n[2/4] Loading global test set...")
    test_loader = load_global_test_set(batch_size=64)

    # ── Partition Data per Client ───────────────────────────────────────
    print(f"\n[3/4] Partitioning data across {args.num_clients} clients...")
    client_loaders = []
    client_sizes = []
    for cid in range(args.num_clients):
        train_loader, val_loader = partition_data(
            dataset=train_dataset,
            num_partitions=args.num_clients,
            partition_id=cid,
            val_split=0.2,
            batch_size=args.batch_size,
        )
        client_loaders.append((train_loader, val_loader))
        client_sizes.append(len(train_loader.dataset))

    # ── Initialize Global Model ─────────────────────────────────────────
    print("\n[4/4] Initializing global model...")
    global_model = MultimodalAlzheimersModel(
        num_clinical_features=9,
        num_classes=4,
        pretrained_cnn=True,
    ).to(device)

    print(f"  Parameters: {global_model.count_parameters():,}")
    global_params = get_parameters(global_model)

    # ── Evaluate initial model ──────────────────────────────────────────
    loss, accuracy, metrics = evaluate(global_model, test_loader, device)
    print(f"\n  [Round 0] Initial - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # ── Federated Learning Rounds ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FEDERATED TRAINING")
    print("=" * 60)

    history = {"round": [], "loss": [], "accuracy": [], "per_class": []}

    for round_num in range(1, args.num_rounds + 1):
        round_start = time.time()
        print(f"\n  -- Round {round_num}/{args.num_rounds} --")

        client_updated_params = []

        for cid in range(args.num_clients):
            # Create a local model copy and load global params
            local_model = MultimodalAlzheimersModel(
                num_clinical_features=9,
                num_classes=4,
                pretrained_cnn=False,
            ).to(device)
            set_parameters(local_model, global_params)

            # Local training
            train_loader, val_loader = client_loaders[cid]
            local_loss = train_local(
                model=local_model,
                dataloader=train_loader,
                epochs=args.local_epochs,
                device=device,
                lr=args.lr,
                class_weights=class_weights.to(device),
            )

            # Local validation
            val_loss, val_acc, _ = evaluate(local_model, val_loader, device)
            print(
                f"    Client {cid}: train_loss={local_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            # Collect updated parameters
            client_updated_params.append(get_parameters(local_model))

            # Free memory
            del local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── FedAvg Aggregation ──────────────────────────────────────────
        global_params = fedavg_aggregate(client_updated_params, client_sizes)

        # Load aggregated params into global model
        set_parameters(global_model, global_params)

        # ── Centralized Evaluation ──────────────────────────────────────
        test_loss, test_acc, test_metrics = evaluate(
            global_model, test_loader, device
        )
        round_time = time.time() - round_start

        print(f"    [Global] Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        print(f"    Per-class: Mild={test_metrics.get('acc_mild_dementia', 0):.3f}, "
              f"Moderate={test_metrics.get('acc_moderate_dementia', 0):.3f}, "
              f"Non={test_metrics.get('acc_non_demented', 0):.3f}, "
              f"VeryMild={test_metrics.get('acc_very_mild_dementia', 0):.3f}")
        print(f"    Round time: {round_time:.1f}s")

        history["round"].append(round_num)
        history["loss"].append(test_loss)
        history["accuracy"].append(test_acc)
        history["per_class"].append(test_metrics)

    # ── Save Final Model ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "global_model.pth")

    torch.save({
        "model_state_dict": global_model.state_dict(),
        "num_clinical_features": 9,
        "num_classes": 4,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "final_accuracy": history["accuracy"][-1] if history["accuracy"] else 0,
        "final_loss": history["loss"][-1] if history["loss"] else 0,
        "history": history,
    }, output_path)

    # ── Final Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Total time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Final accuracy: {history['accuracy'][-1]:.4f}")
    print(f"  Final loss:     {history['loss'][-1]:.4f}")
    print(f"  Model saved to: {output_path}")
    print()

    print("  Round | Loss     | Accuracy")
    print("  " + "-" * 35)
    for i, rd in enumerate(history["round"]):
        print(f"  {rd:5d} | {history['loss'][i]:.4f}  | {history['accuracy'][i]:.4f}")
    print()


if __name__ == "__main__":
    main()
