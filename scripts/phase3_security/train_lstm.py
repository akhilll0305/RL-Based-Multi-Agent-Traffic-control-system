"""
Train TrafficLSTM on clean baseline queue data.

Inputs:
- data_security/train_states.npy
- data_security/train_metadata.npz
- data_security/val_states.npy
- data_security/val_metadata.npz

Outputs:
- checkpoints_security/lstm_predictor.pth
- results_security/lstm_training_history.csv

Training target:
Given window of N timesteps (default 20), predict next-step 4-lane queue values.
"""

from __future__ import annotations

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------


import argparse
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from traffic_rl.security.lstm import TrafficLSTM


TRAIN_STATES_PATH = "data/security/train_states.npy"
TRAIN_META_PATH = "data/security/train_metadata.npz"
VAL_STATES_PATH = "data/security/val_states.npy"
VAL_META_PATH = "data/security/val_metadata.npz"

CHECKPOINT_PATH = "outputs/checkpoints/security/lstm_predictor.pth"
HISTORY_PATH = "outputs/results/security/lstm_training_history.csv"


@dataclass
class DatasetBundle:
    states: np.ndarray
    episode_lengths: np.ndarray


def set_seed(seed: int = 42) -> None:
    """Set reproducible seeds for numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_npz_metadata(meta_path: str) -> np.ndarray:
    """Load episode_lengths from a metadata NPZ file."""
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    meta = np.load(meta_path, allow_pickle=False)
    if "episode_lengths" not in meta:
        raise KeyError(f"Missing 'episode_lengths' in {meta_path}")
    return meta["episode_lengths"].astype(np.int32)


def load_split_dataset(states_path: str, meta_path: str) -> DatasetBundle:
    """Load states and episode lengths with consistency checks."""
    if not os.path.exists(states_path):
        raise FileNotFoundError(f"States file not found: {states_path}")

    states = np.load(states_path).astype(np.float32)
    if states.ndim != 3 or states.shape[2] != 4:
        raise ValueError(
            f"Expected states shape (steps, intersections, 4), got {states.shape}"
        )

    episode_lengths = _load_npz_metadata(meta_path)
    if int(episode_lengths.sum()) != int(states.shape[0]):
        raise ValueError(
            "Episode length sum mismatch: "
            f"sum(lengths)={episode_lengths.sum()} vs steps={states.shape[0]}"
        )

    return DatasetBundle(states=states, episode_lengths=episode_lengths)


def episode_slices(episode_lengths: np.ndarray) -> List[Tuple[int, int]]:
    """Convert episode lengths into list of [start, end) slices."""
    slices: List[Tuple[int, int]] = []
    start = 0
    for length in episode_lengths:
        end = start + int(length)
        slices.append((start, end))
        start = end
    return slices


def build_windows(
    states: np.ndarray,
    lengths: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows by episode, across all intersections.

    For each episode and each intersection:
      X: states[t-window_size:t, intersection, :]  -> shape (window_size, 4)
      y: states[t, intersection, :]                -> shape (4,)

    This prevents time leakage across episode boundaries.
    """
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for ep_start, ep_end in episode_slices(lengths):
        ep_data = states[ep_start:ep_end]  # (ep_steps, 8, 4)
        ep_steps = ep_data.shape[0]
        if ep_steps <= window_size:
            continue

        for intersection_idx in range(ep_data.shape[1]):
            series = ep_data[:, intersection_idx, :]  # (ep_steps, 4)
            for t in range(window_size, ep_steps):
                x_list.append(series[t - window_size : t])
                y_list.append(series[t])

    if not x_list:
        raise ValueError(
            "No training windows generated. Increase episode length or reduce window_size."
        )

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    return x, y


class QueueWindowDataset(Dataset):
    """Torch dataset for (window, next-step) queue prediction."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


@torch.no_grad()
def evaluate(model: TrafficLSTM, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Evaluate average MSE loss and MAE over a loader."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)
        mae = torch.mean(torch.abs(pred - yb))

        batch_size = xb.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_mae += float(mae.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_mae / total_samples


def train_lstm(args: argparse.Namespace) -> None:
    """End-to-end training pipeline for TrafficLSTM."""
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_bundle = load_split_dataset(args.train_states, args.train_meta)
    val_bundle = load_split_dataset(args.val_states, args.val_meta)

    x_train, y_train = build_windows(
        train_bundle.states,
        train_bundle.episode_lengths,
        window_size=args.window_size,
    )
    x_val, y_val = build_windows(
        val_bundle.states,
        val_bundle.episode_lengths,
        window_size=args.window_size,
    )

    train_ds = QueueWindowDataset(x_train, y_train)
    val_ds = QueueWindowDataset(x_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = TrafficLSTM(
        input_size=4,
        hidden_size=args.hidden_size,
        num_layers=1,
        output_size=4,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.history_path), exist_ok=True)

    history_rows: List[dict] = []
    best_val_loss = float("inf")

    print("=" * 70)
    print("LSTM Training")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Window size     : {args.window_size}")
    print(f"Train windows   : {len(train_ds)}")
    print(f"Val windows     : {len(val_ds)}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Learning rate   : {args.learning_rate}")
    print(f"Epochs          : {args.epochs}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = xb.shape[0]
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        train_loss = running_loss / max(sample_count, 1)
        val_loss, val_mae = evaluate(model, val_loader, loss_fn, device)

        history_rows.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
                "val_mae": val_mae,
            }
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            model.save(args.checkpoint_path)

        flag = "*" if improved else " "
        print(
            f"{flag} Epoch {epoch:03d} | "
            f"train_mse={train_loss:.6f} | "
            f"val_mse={val_loss:.6f} | "
            f"val_mae={val_mae:.6f}"
        )

    pd.DataFrame(history_rows).to_csv(args.history_path, index=False)

    print("\nTraining complete.")
    print(f"Best model saved : {args.checkpoint_path}")
    print(f"Training history : {args.history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM predictor for queue correction")
    parser.add_argument("--train-states", type=str, default=TRAIN_STATES_PATH)
    parser.add_argument("--train-meta", type=str, default=TRAIN_META_PATH)
    parser.add_argument("--val-states", type=str, default=VAL_STATES_PATH)
    parser.add_argument("--val-meta", type=str, default=VAL_META_PATH)

    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint-path", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--history-path", type=str, default=HISTORY_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    train_lstm(parse_args())
