from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rul.data import prepare_datasets, prepare_rollout_dataset
from rul.models import build_model
from rul.rollout import evaluate_rollout
from rul.train import predict
from rul.utils import cmapss_score, count_parameters, get_device, mae, rmse, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RUL model on FD001 test engines.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/raw/FD001")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rollout", action="store_true", help="Also run K-step feature rollout evaluation.")
    parser.add_argument("--rollout-k", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument(
        "--noise-std",
        type=float,
        nargs="+",
        default=[0.0],
        help="Gaussian noise std values for rollout input windows. Example: --noise-std 0.0 0.01 0.02 0.05",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint["args"]
    task = checkpoint.get("task", train_args.get("task", "rul"))
    set_seed(train_args.get("seed", 42))
    device = get_device(args.device)

    prepared = prepare_datasets(
        data_dir=args.data_dir,
        sequence_length=train_args["sequence_length"],
        rul_cap=train_args["rul_cap"],
        seed=train_args.get("seed", 42),
    )
    model = build_model(
        model_name=checkpoint["model_name"],
        input_size=checkpoint["input_size"],
        hidden_size=train_args["hidden_size"],
        num_layers=train_args["num_layers"],
        dropout=train_args["dropout"],
        solver_steps=train_args["solver_steps"],
        output_size=checkpoint.get("output_size", 1),
        output_activation=checkpoint.get("output_activation", "softplus"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    output_dir = Path(args.output_dir)
    metrics = {
        "model": checkpoint["model_name"],
        "task": task,
        "parameter_count": count_parameters(model),
    }

    if task == "rul":
        test_loader = DataLoader(prepared.test_dataset, batch_size=args.batch_size, shuffle=False)
        y_true, y_pred = predict(model, test_loader, device)
        metrics.update(
            {
                "test_rmse": rmse(y_true, y_pred),
                "test_mae": mae(y_true, y_pred),
                "test_score": cmapss_score(y_true, y_pred),
            }
        )

        save_json(metrics, output_dir / "metrics" / f"{checkpoint['model_name']}_test_metrics.json")
        predictions = pd.DataFrame(
            {
                "unit": prepared.test_dataset.unit_ids,
                "true_rul": y_true,
                "predicted_rul": y_pred,
            }
        )
        pred_path = output_dir / "metrics" / f"{checkpoint['model_name']}_test_predictions.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(pred_path, index=False)

        print(metrics)
        print(f"Saved predictions to {pred_path}")

    if args.rollout:
        if task != "next_step":
            raise ValueError(
                "K-step rollout needs a checkpoint trained with --task next_step. "
                f"This checkpoint was trained with task={task!r}. Train one with: "
                f"python -m rul.train --model {checkpoint['model_name']} --task next_step"
            )
        rollout_loader = DataLoader(
            prepare_rollout_dataset(
                data_dir=args.data_dir,
                sequence_length=train_args["sequence_length"],
                horizon=max(args.rollout_k),
                split="test",
                seed=train_args.get("seed", 42),
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        rollout_by_noise = {}
        for noise_std in args.noise_std:
            label = "clean" if noise_std == 0.0 else f"noise_std_{noise_std:g}"
            rollout_results = evaluate_rollout(
                model,
                rollout_loader,
                K_list=args.rollout_k,
                noise_std=noise_std,
            )
            rollout_by_noise[label] = rollout_results

            if noise_std == 0.0:
                print("--- CLEAN ---")
            else:
                print(f"--- NOISE (std={noise_std:g}) ---")
            for k in args.rollout_k:
                values = rollout_results[k]
                print(f"K={k}: RMSE={values['rmse']:.4f}, MAE={values['mae']:.4f}")

        metrics["rollout"] = rollout_by_noise
        save_json(metrics, output_dir / "metrics" / f"{checkpoint['model_name']}_next_step_rollout_metrics.json")


if __name__ == "__main__":
    main()
