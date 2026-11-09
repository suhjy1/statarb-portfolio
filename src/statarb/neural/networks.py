from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

ARCHITECTURES = ("conv_transformer", "lstm", "linear")


class _ResidualPolicy(nn.Module):
    def __init__(self, lookback: int, channels: int) -> None:
        super().__init__()
        self.lookback = lookback
        self.channels = channels

    def _validate(self, features: Tensor) -> None:
        if features.ndim != 3 or features.shape[1:] != (self.lookback, self.channels):
            raise ValueError(f"features must have shape [batch, {self.lookback}, {self.channels}]")
        if not features.is_floating_point():
            raise ValueError("features must be floating point")


class LinearPolicy(_ResidualPolicy):
    def __init__(self, lookback: int, channels: int) -> None:
        super().__init__(lookback, channels)
        self.head = nn.Linear(lookback * channels, 1, device="cpu", dtype=torch.float32)

    def forward(self, features: Tensor) -> Tensor:
        self._validate(features)
        return self.head(features.flatten(start_dim=1)).squeeze(-1).tanh()


class LstmPolicy(_ResidualPolicy):
    def __init__(self, lookback: int, channels: int, hidden_size: int) -> None:
        super().__init__(lookback, channels)
        self.recurrent = nn.LSTM(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            device="cpu",
            dtype=torch.float32,
        )
        self.head = nn.Linear(hidden_size, 1, device="cpu", dtype=torch.float32)

    def forward(self, features: Tensor) -> Tensor:
        self._validate(features)

        if features.shape[0] == 0:
            return features.new_empty((0,))

        encoded, _ = self.recurrent(features)

        return self.head(encoded[:, -1]).squeeze(-1).tanh()


def _positions(lookback: int, hidden_size: int) -> Tensor:
    positions = torch.arange(lookback, dtype=torch.float32, device="cpu").unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, hidden_size, 2, dtype=torch.float32, device="cpu")
        * (-math.log(10000.0) / hidden_size)
    )
    angles = positions * frequencies
    encoded = torch.zeros((lookback, hidden_size), dtype=torch.float32, device="cpu")
    encoded[:, 0::2] = angles.sin()
    encoded[:, 1::2] = angles[:, : hidden_size // 2].cos()

    return encoded.unsqueeze(0)


class ConvTransformerPolicy(_ResidualPolicy):
    def __init__(self, lookback: int, channels: int, hidden_size: int) -> None:
        super().__init__(lookback, channels)
        kwargs = {"device": "cpu", "dtype": torch.float32}
        self.first = nn.Conv1d(channels, hidden_size, kernel_size=2, **kwargs)
        self.second = nn.Conv1d(hidden_size, hidden_size, kernel_size=2, **kwargs)
        self.skip = nn.Conv1d(channels, hidden_size, kernel_size=1, **kwargs)
        self.normalization = nn.LayerNorm(hidden_size, **kwargs)
        heads = 4 if hidden_size % 4 == 0 else 2 if hidden_size % 2 == 0 else 1
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=2 * hidden_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
            **kwargs,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Linear(hidden_size, 1, **kwargs)
        self.register_buffer("positions", _positions(lookback, hidden_size))
        self.register_buffer(
            "causal_mask",
            torch.ones((lookback, lookback), dtype=torch.bool, device="cpu").triu(diagonal=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        self._validate(features)

        if features.shape[0] == 0:
            return features.new_empty((0,))

        channels_first = features.transpose(1, 2)
        local = functional.gelu(self.first(functional.pad(channels_first, (1, 0))))
        local = self.second(functional.pad(local, (1, 0)))
        encoded = self.normalization((local + self.skip(channels_first)).transpose(1, 2))
        encoded = self.temporal(encoded + self.positions, mask=self.causal_mask)

        return self.head(encoded[:, -1]).squeeze(-1).tanh()


def make_policy(
    architecture: str,
    lookback: int = 30,
    channels: int = 2,
    hidden_size: int = 16,
    seed: int = 42,
) -> nn.Module:
    for name, value in (
        ("lookback", lookback),
        ("channels", channels),
        ("hidden_size", hidden_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())

        if architecture == "conv_transformer":
            return ConvTransformerPolicy(lookback, channels, hidden_size)

        if architecture == "lstm":
            return LstmPolicy(lookback, channels, hidden_size)

        return LinearPolicy(lookback, channels)
