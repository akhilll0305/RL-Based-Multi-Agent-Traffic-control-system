"""
LSTM predictor for security-phase queue correction.

Model contract:
- Input sequence shape: (batch, seq_len, 4)
- Output next-step queue prediction shape: (batch, 4)

This module is intentionally standalone so it can be reused by:
- train_lstm.py (training and checkpointing)
- security_layer.py (frozen inference for correction)
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):
    """Single-layer LSTM predictor for 4-lane queue forecasting."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 4,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, 4)

        Returns:
            Tensor of shape (batch, 4) for next-step prediction.
        """
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input shape (batch, seq_len, {self.input_size}), got {tuple(x.shape)}"
            )

        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]
        prediction = self.output_layer(last_timestep)
        return prediction

    @torch.no_grad()
    def predict(self, history: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict next-step queue values from one history window.

        Args:
            history: shape (seq_len, 4) or (1, seq_len, 4)

        Returns:
            numpy array shape (4,)
        """
        self.eval()

        if isinstance(history, np.ndarray):
            history_tensor = torch.from_numpy(history).float()
        else:
            history_tensor = history.float()

        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)
        elif history_tensor.ndim != 3:
            raise ValueError(
                "history must have shape (seq_len, 4) or (batch, seq_len, 4)"
            )

        history_tensor = history_tensor.to(self.device)
        pred = self.forward(history_tensor)
        pred = pred[0].detach().cpu().numpy().astype(np.float32)
        return pred

    def save(self, path: str) -> None:
        """Save model checkpoint with architecture metadata."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "output_size": self.output_size,
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: Union[str, torch.device, None] = None,
    ) -> "TrafficLSTM":
        """
        Load a saved TrafficLSTM checkpoint and return initialized model.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_location = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(path, map_location=map_location)

        if "state_dict" not in checkpoint or "config" not in checkpoint:
            raise ValueError("Invalid checkpoint format: expected keys 'state_dict' and 'config'")

        model = cls(device=map_location, **checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
