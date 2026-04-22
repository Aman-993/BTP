from __future__ import annotations

import math

import torch
from torch import nn


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def rollout_predict(model: nn.Module, input_seq: torch.Tensor, K: int) -> torch.Tensor:
    """Autoregressively predict K future feature vectors.

    Args:
        model: A sequence model that maps (batch, seq_len, features) to
            (batch, features) or (batch, 1, features).
        input_seq: Initial context window with shape (batch, seq_len, features).
        K: Number of future steps to predict.

    Returns:
        Tensor with shape (batch, K, features).

    Ground-truth future values are never fed back into the model.
    """
    if input_seq.ndim != 3:
        raise ValueError(f"input_seq must have shape (batch, seq_len, features), got {input_seq.shape}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    model.eval()
    device = _model_device(model)
    window = input_seq.to(device)
    feature_count = window.shape[-1]
    predictions = []

    with torch.no_grad():
        for _ in range(K):
            next_step = model(window)
            if next_step.ndim == 3 and next_step.shape[1] == 1:
                next_step = next_step[:, 0, :]
            if next_step.ndim != 2:
                raise ValueError(
                    "rollout_predict expected model output shape (batch, features) "
                    f"or (batch, 1, features), got {next_step.shape}"
                )
            if next_step.shape[-1] != feature_count:
                raise ValueError(
                    "rollout_predict requires the model to predict the next full feature vector. "
                    f"Input has {feature_count} features, but model output has {next_step.shape[-1]}. "
                    "Your current RUL models output a scalar, so they cannot be rolled out over "
                    "sensor features without training a next-step feature predictor."
                )

            predictions.append(next_step)
            window = torch.cat([window[:, 1:, :], next_step.unsqueeze(1)], dim=1)

    return torch.stack(predictions, dim=1)


def evaluate_rollout(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    K_list: list[int] | tuple[int, ...] = (1, 2, 3, 5),
    noise_std: float = 0.0,
) -> dict[int, dict[str, float]]:
    """Evaluate autoregressive rollout RMSE/MAE for multiple horizons.

    The dataloader must yield batches as:
        input_seq, true_future_seq

    Shapes:
        input_seq: (batch, seq_len, features)
        true_future_seq: (batch, future_len, features), where future_len >= max(K_list)
    """
    if not K_list:
        raise ValueError("K_list must contain at least one horizon.")
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0.0, got {noise_std}")

    max_k = max(K_list)
    totals = {k: {"sse": 0.0, "sae": 0.0, "count": 0} for k in K_list}
    device = _model_device(model)

    for batch in dataloader:
        if len(batch) != 2:
            raise ValueError(
                "evaluate_rollout expects dataloader batches of (input_seq, true_future_seq)."
            )

        input_seq, true_future = batch
        if true_future.ndim != 3:
            raise ValueError(
                "true_future_seq must have shape (batch, future_len, features), "
                f"got {true_future.shape}"
            )
        if true_future.shape[1] < max_k:
            raise ValueError(
                f"true_future_seq length must be >= max(K_list)={max_k}, got {true_future.shape[1]}"
            )

        if noise_std > 0.0:
            input_seq = input_seq + torch.randn_like(input_seq) * noise_std

        pred = rollout_predict(model, input_seq, max_k)
        true_future = true_future.to(device)

        for k in K_list:
            err = pred[:, :k, :] - true_future[:, :k, :]
            totals[k]["sse"] += torch.sum(err.square()).item()
            totals[k]["sae"] += torch.sum(err.abs()).item()
            totals[k]["count"] += err.numel()

    results: dict[int, dict[str, float]] = {}
    for k, values in totals.items():
        if values["count"] == 0:
            results[k] = {"rmse": math.nan, "mae": math.nan}
            continue
        results[k] = {
            "rmse": math.sqrt(values["sse"] / values["count"]),
            "mae": values["sae"] / values["count"],
        }

    return results
