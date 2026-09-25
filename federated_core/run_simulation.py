# =============================================================================
# federated_core/run_simulation.py
# ----------------------------------
# Entry point for the Flower federated learning simulation.
# =============================================================================

from __future__ import annotations

import os
import sys
import argparse
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from flwr.simulation import run_simulation

from federated_core.client.flower_client import client_app
from federated_core.server.flower_server import server_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Federated Learning simulation for Alzheimer's prediction"
    )
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "saved_models"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["NUM_FL_CLIENTS"] = str(args.num_clients)
    os.environ["NUM_FL_ROUNDS"] = str(args.num_rounds)
    os.environ["LOCAL_EPOCHS"] = str(args.local_epochs)
    os.environ["FL_BATCH_SIZE"] = str(args.batch_size)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "global_model.pth")
    # The centralized Flower evaluation callback receives the exact global
    # parameters after every aggregation round and persists the latest set.
    os.environ["FL_MODEL_OUTPUT_PATH"] = output_path

    print("=" * 60)
    print("  ALZHEIMER'S FEDERATED LEARNING SIMULATION")
    print("=" * 60)
    print(f"  Clients:      {args.num_clients}")
    print(f"  Rounds:       {args.num_rounds}")
    print(f"  Local Epochs: {args.local_epochs}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  Device:       {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Output:       {output_path}")
    print("=" * 60)
    print()

    start_time = time.time()

    num_gpus = torch.cuda.device_count()
    gpu_per_client = (
        min(1.0, num_gpus / args.num_clients) if num_gpus > 0 else 0.0
    )

    history = run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.num_clients,
        backend_config={
            "client_resources": {
                "num_cpus": 1,
                "num_gpus": gpu_per_client,
            }
        },
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)

    if history and getattr(history, "losses_centralized", None):
        print("\n  Round | Loss     | Accuracy")
        print("  " + "-" * 35)
        accuracy_history = getattr(history, "metrics_centralized", {}).get(
            "accuracy", []
        )
        for idx, (rd, loss) in enumerate(history.losses_centralized):
            accuracy = accuracy_history[idx][1] if idx < len(accuracy_history) else float("nan")
            print(f"  {rd:5d} | {loss:.4f}  | {accuracy:.4f}")

    if not os.path.isfile(output_path):
        raise RuntimeError(
            "Federated simulation completed but the aggregated model was not "
            f"saved to {output_path}."
        )

    print(f"\n  Aggregated model saved to: {output_path}")
    print()
    

if __name__ == "__main__":
    main()
