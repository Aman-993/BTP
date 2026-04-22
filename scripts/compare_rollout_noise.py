from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rul.data import prepare_rollout_dataset
from rul.models import build_model
from rul.rollout import evaluate_rollout
from rul.utils import count_parameters, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare clean/noisy K-step rollout robustness for all next-step models."
    )
    parser.add_argument("--models", nargs="+", default=["rnn", "lstm", "gru", "lnn"])
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--data-dir", default="data/raw/FD001")
    parser.add_argument("--output-dir", default="outputs/metrics")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rollout-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--noise-std", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_next_step_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint["args"]
    task = checkpoint.get("task", train_args.get("task", "rul"))
    if task != "next_step":
        raise ValueError(f"{checkpoint_path} is task={task!r}, but this script needs next_step checkpoints.")

    model = build_model(
        model_name=checkpoint["model_name"],
        input_size=checkpoint["input_size"],
        hidden_size=train_args["hidden_size"],
        num_layers=train_args["num_layers"],
        dropout=train_args["dropout"],
        solver_steps=train_args["solver_steps"],
        output_size=checkpoint["output_size"],
        output_activation=checkpoint["output_activation"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    nested_results = {}

    for model_name in args.models:
        checkpoint_path = Path(args.checkpoint_dir) / f"{model_name}_next_step_best.pt"
        if not checkpoint_path.exists():
            print(f"Skipping {model_name}: missing {checkpoint_path}")
            continue

        model, checkpoint = load_next_step_model(checkpoint_path, device)
        train_args = checkpoint["args"]
        set_seed(train_args.get("seed", 42))

        rollout_dataset = prepare_rollout_dataset(
            data_dir=args.data_dir,
            sequence_length=train_args["sequence_length"],
            horizon=max(args.rollout_k),
            split="test",
            seed=train_args.get("seed", 42),
        )
        rollout_loader = DataLoader(rollout_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"\n=== {model_name.upper()} ===")
        nested_results[model_name] = {
            "parameter_count": count_parameters(model),
            "noise": {},
        }

        for noise_std in args.noise_std:
            label = "clean" if noise_std == 0.0 else f"noise_std_{noise_std:g}"
            results = evaluate_rollout(
                model,
                rollout_loader,
                K_list=args.rollout_k,
                noise_std=noise_std,
            )
            nested_results[model_name]["noise"][label] = results

            print("--- CLEAN ---" if noise_std == 0.0 else f"--- NOISE (std={noise_std:g}) ---")
            for k in args.rollout_k:
                rmse = results[k]["rmse"]
                mae = results[k]["mae"]
                print(f"K={k}: RMSE={rmse:.4f}, MAE={mae:.4f}")
                rows.append(
                    {
                        "model": model_name,
                        "parameter_count": count_parameters(model),
                        "noise_std": noise_std,
                        "K": k,
                        "rmse": rmse,
                        "mae": mae,
                    }
                )

    if not rows:
        raise FileNotFoundError(
            "No next-step checkpoints were found. Train them first, for example: "
            "python -m rul.train --model gru --task next_step --epochs 30"
        )

    table = pd.DataFrame(rows)
    table_path = output_dir / "rollout_noise_comparison.csv"
    json_path = output_dir / "rollout_noise_comparison.json"
    table.to_csv(table_path, index=False)
    save_json(nested_results, json_path)

    print("\n=== SUMMARY TABLE ===")
    print(table.to_string(index=False))
    print(f"\nSaved CSV to {table_path}")
    print(f"Saved JSON to {json_path}")


if __name__ == "__main__":
    main()
