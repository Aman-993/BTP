from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RecurrentRegressor(nn.Module):
    """Common wrapper for vanilla RNN, LSTM, and GRU sequence regression."""

    def __init__(
        self,
        model_type: str,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        output_activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.output_activation = output_activation
        recurrent_classes = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }
        if model_type not in recurrent_classes:
            raise ValueError(f"Unknown recurrent model: {model_type}")

        self.encoder = recurrent_classes[model_type](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        last_state = encoded[:, -1, :]
        out = self.head(last_state)
        if self.output_activation == "softplus":
            return F.softplus(out)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(out)
        if self.output_activation == "none":
            return out
        raise ValueError(f"Unknown output activation: {self.output_activation}")


class LTCCell(nn.Module):
    """A small Liquid Time-Constant style cell.

    The hidden state evolves with an Euler update:
        h(t+1) = h(t) + dt * (-h(t) + tanh(Wx + Uh)) / tau(x, h)

    tau is input-dependent and positive, giving the model liquid/adaptive
    dynamics while staying easy to understand and train for this project.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.min_tau = 1.0

        nn.init.xavier_uniform_(self.input_layer.weight, gain=0.5)
        nn.init.orthogonal_(self.hidden_layer.weight, gain=0.5)
        nn.init.xavier_uniform_(self.tau_layer.weight, gain=0.1)
        nn.init.constant_(self.tau_layer.bias, 1.0)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        candidate = torch.tanh(self.input_layer(x_t) + self.hidden_layer(h))
        tau_input = torch.cat([x_t, h], dim=-1)
        tau = F.softplus(self.tau_layer(tau_input)) + self.min_tau

        # Keep the Euler update contractive. Without this, tiny tau values can
        # make h explode to inf/NaN early in training.
        alpha = torch.clamp(dt / tau, min=0.0, max=1.0)
        h_next = h + alpha * (candidate - h)
        return self.norm(h_next)


class LiquidRegressor(nn.Module):
    """Sequence regressor using the Liquid Time-Constant style cell."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        dropout: float = 0.2,
        solver_steps: int = 1,
        output_size: int = 1,
        output_activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.output_activation = output_activation
        self.cell = LTCCell(input_size=input_size, hidden_size=hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.solver_steps = solver_steps
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.cell.hidden_size, device=x.device, dtype=x.dtype)
        dt = 1.0 / self.solver_steps

        for t in range(x.shape[1]):
            for _ in range(self.solver_steps):
                h = self.cell(x[:, t, :], h, dt=dt)

        out = self.head(self.dropout(h))
        if self.output_activation == "softplus":
            return F.softplus(out)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(out)
        if self.output_activation == "none":
            return out
        raise ValueError(f"Unknown output activation: {self.output_activation}")


def build_model(
    model_name: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    solver_steps: int,
    output_size: int = 1,
    output_activation: str = "softplus",
) -> nn.Module:
    if model_name in {"rnn", "lstm", "gru"}:
        return RecurrentRegressor(
            model_type=model_name,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
            output_activation=output_activation,
        )
    if model_name in {"lnn", "ltc"}:
        return LiquidRegressor(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            solver_steps=solver_steps,
            output_size=output_size,
            output_activation=output_activation,
        )
    raise ValueError("model_name must be one of: rnn, lstm, gru, lnn")
