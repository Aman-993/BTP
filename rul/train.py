from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rul.data import prepare_datasets, prepare_next_step_datasets
from rul.models import build_model
from rul.utils import cmapss_score, count_parameters, get_device, mae, rmse, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNN/LNN models for C-MAPSS FD001 RUL.")
    parser.add_argument("--model", choices=["rnn", "lstm", "gru", "lnn"], required=True)
    parser.add_argument("--task", choices=["rul", "next_step"], default="rul")
    parser.add_argument("--data-dir", default="data/raw/FD001")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--rul-cap", type=int, default=125)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--solver-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy().reshape(-1)
        y_pred.append(pred)
        y_true.append(y.numpy().reshape(-1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    if args.task == "rul":
        prepared = prepare_datasets(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            rul_cap=args.rul_cap,
            seed=args.seed,
        )
        output_size = 1
        output_activation = "softplus"
    else:
        prepared = prepare_next_step_datasets(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            seed=args.seed,
        )
        output_size = prepared.input_size
        output_activation = "sigmoid"
    train_loader = DataLoader(prepared.train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(prepared.val_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        model_name=args.model,
        input_size=prepared.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        solver_steps=args.solver_steps,
        output_size=output_size,
        output_activation=output_activation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    history: list[dict[str, float]] = []
    start_time = time.time()
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"{args.model}_best.pt" if args.task == "rul" else f"{args.model}_next_step_best.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"{args.model.upper()} epoch {epoch}/{args.epochs}", leave=False)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Training produced a non-finite loss. Try a smaller learning rate, "
                    "for example --lr 0.0003, or reduce --hidden-size."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(x)
            loop.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = running_loss / len(prepared.train_dataset)
        val_true, val_pred = predict(model, val_loader, device)
        val_rmse = rmse(val_true, val_pred)
        val_mae = mae(val_true, val_pred)
        row = {"epoch": epoch, "train_mse": train_loss, "val_rmse": val_rmse, "val_mae": val_mae}
        history.append(row)
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.3f} "
            f"val_rmse={val_rmse:.3f} val_mae={val_mae:.3f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "task": args.task,
                    "input_size": prepared.input_size,
                    "output_size": output_size,
                    "output_activation": output_activation,
                    "feature_columns": prepared.feature_columns,
                    "args": vars(args),
                    "best_val_rmse": best_rmse,
                },
                checkpoint_path,
            )

    elapsed = time.time() - start_time
    metrics = {
        "model": args.model,
        "task": args.task,
        "best_val_rmse": best_rmse,
        "parameter_count": count_parameters(model),
        "training_seconds": elapsed,
        "history": history,
    }
    if args.task == "rul":
        metrics["final_val_score"] = cmapss_score(val_true, val_pred)
    metric_name = f"{args.model}_train_metrics.json" if args.task == "rul" else f"{args.model}_next_step_train_metrics.json"
    save_json(metrics, Path(args.output_dir) / "metrics" / metric_name)
    print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
