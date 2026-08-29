"""
Evaluate trained TrafficLSTM prediction quality on validation split.

Outputs:
- outputs/results/security/lstm_eval_summary.csv
- outputs/results/security/lstm_eval_per_intersection_lane.csv
- outputs/analysis/security/lstm_eval/predicted_vs_actual_sample_episode.png
- outputs/analysis/security/lstm_eval/mae_per_intersection.png
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
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from traffic_rl.security.lstm import TrafficLSTM
from train_lstm import load_split_dataset, episode_slices


VAL_STATES_PATH = "data/security/val_states.npy"
VAL_META_PATH = "data/security/val_metadata.npz"
CHECKPOINT_PATH = "outputs/checkpoints/security/lstm_predictor.pth"
RESULTS_DIR = "outputs/results/security"
ANALYSIS_DIR = "outputs/analysis/security/lstm_eval"

LANE_NAMES = ["north", "south", "east", "west"]


@dataclass
class WindowData:
    x: np.ndarray
    y: np.ndarray
    intersection_idx: np.ndarray
    episode_idx: np.ndarray
    step_idx: np.ndarray


class EvalWindowDataset(Dataset):
    """Dataset wrapper for evaluation windows."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def build_eval_windows(states: np.ndarray, lengths: np.ndarray, window_size: int) -> WindowData:
    """Build leakage-safe windows and keep source indices for detailed analysis."""
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    i_list: List[int] = []
    e_list: List[int] = []
    t_list: List[int] = []

    for ep_idx, (start, end) in enumerate(episode_slices(lengths)):
        ep_data = states[start:end]  # (steps, 8, 4)
        steps = ep_data.shape[0]
        if steps <= window_size:
            continue

        for intersection_idx in range(ep_data.shape[1]):
            series = ep_data[:, intersection_idx, :]
            for t in range(window_size, steps):
                x_list.append(series[t - window_size : t])
                y_list.append(series[t])
                i_list.append(intersection_idx)
                e_list.append(ep_idx)
                t_list.append(t)

    if not x_list:
        raise ValueError("No windows generated for evaluation.")

    return WindowData(
        x=np.stack(x_list).astype(np.float32),
        y=np.stack(y_list).astype(np.float32),
        intersection_idx=np.array(i_list, dtype=np.int32),
        episode_idx=np.array(e_list, dtype=np.int32),
        step_idx=np.array(t_list, dtype=np.int32),
    )


@torch.no_grad()
def predict_all(model: TrafficLSTM, x: np.ndarray, batch_size: int) -> np.ndarray:
    """Run batched prediction for all windows."""
    device = model.device
    dataset = EvalWindowDataset(x, np.zeros((x.shape[0], 4), dtype=np.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    preds: List[np.ndarray] = []
    model.eval()
    for xb, _ in loader:
        xb = xb.to(device)
        pred = model(xb).detach().cpu().numpy().astype(np.float32)
        preds.append(pred)
    return np.concatenate(preds, axis=0)


def compute_metrics(pred: np.ndarray, target: np.ndarray, intersection_idx: np.ndarray, tls_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute global summary and per-intersection/lane MAE metrics."""
    abs_err = np.abs(pred - target)

    summary = {
        "global_mae": float(abs_err.mean()),
        "global_mse": float(np.mean((pred - target) ** 2)),
        "global_p95_abs_error": float(np.percentile(abs_err, 95)),
    }

    rows = []
    for i, tls in enumerate(tls_ids):
        mask = intersection_idx == i
        if not np.any(mask):
            continue
        per_tls_err = abs_err[mask]
        rows.append(
            {
                "tls_id": tls,
                "mae_all_lanes": float(per_tls_err.mean()),
                "mae_north": float(per_tls_err[:, 0].mean()),
                "mae_south": float(per_tls_err[:, 1].mean()),
                "mae_east": float(per_tls_err[:, 2].mean()),
                "mae_west": float(per_tls_err[:, 3].mean()),
            }
        )

    summary_df = pd.DataFrame([summary])
    per_tls_df = pd.DataFrame(rows)
    return summary_df, per_tls_df


def plot_mae_per_intersection(per_tls_df: pd.DataFrame, output_path: str) -> None:
    """Bar chart for average MAE per intersection."""
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(per_tls_df["tls_id"], per_tls_df["mae_all_lanes"])
    ax.set_title("LSTM Validation MAE by Intersection")
    ax.set_xlabel("Intersection")
    ax.set_ylabel("MAE (all 4 lanes)")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sample_episode(pred_all: np.ndarray, target_all: np.ndarray, meta: WindowData, tls_ids: List[str], output_path: str) -> dict:
    """
    Plot predicted vs actual lane queues for one sample episode/intersection.

    Uses: first validation episode (episode 0), intersection tls_1 (index 0).
    """
    sample_episode = 0
    sample_intersection = 0
    sample_tls = tls_ids[sample_intersection]

    mask = (meta.episode_idx == sample_episode) & (meta.intersection_idx == sample_intersection)
    if not np.any(mask):
        raise ValueError("No sample windows found for selected episode/intersection.")

    # Preserve temporal order in the sample sequence.
    order = np.argsort(meta.step_idx[mask])
    actual = target_all[mask][order]
    pred = pred_all[mask][order]
    steps = np.arange(actual.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for lane_idx, ax in enumerate(axes.ravel()):
        ax.plot(steps, actual[:, lane_idx], label="actual", linewidth=1.8)
        ax.plot(steps, pred[:, lane_idx], label="predicted", linewidth=1.5)
        ax.set_title(f"{sample_tls} - {LANE_NAMES[lane_idx]}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Queue")
        ax.legend()

    fig.suptitle("Predicted vs Actual Queue (Validation Episode Sample)")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    # Jump-behavior diagnostic: predictions should not overreact to spikes.
    actual_jump = np.abs(np.diff(actual, axis=0))
    pred_jump = np.abs(np.diff(pred, axis=0))

    return {
        "sample_tls": sample_tls,
        "sample_episode": sample_episode,
        "actual_jump_p95": float(np.percentile(actual_jump, 95)),
        "pred_jump_p95": float(np.percentile(pred_jump, 95)),
        "actual_jump_max": float(np.max(actual_jump)),
        "pred_jump_max": float(np.max(pred_jump)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM predictor on validation set")
    parser.add_argument("--val-states", type=str, default=VAL_STATES_PATH)
    parser.add_argument("--val-meta", type=str, default=VAL_META_PATH)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    val_bundle = load_split_dataset(args.val_states, args.val_meta)
    val_meta_npz = np.load(args.val_meta, allow_pickle=False)
    tls_ids = [str(x) for x in val_meta_npz["tls_ids"]]

    window_data = build_eval_windows(
        states=val_bundle.states,
        lengths=val_bundle.episode_lengths,
        window_size=args.window_size,
    )

    model = TrafficLSTM.load(args.checkpoint)
    pred = predict_all(model, window_data.x, batch_size=args.batch_size)

    summary_df, per_tls_df = compute_metrics(
        pred=pred,
        target=window_data.y,
        intersection_idx=window_data.intersection_idx,
        tls_ids=tls_ids,
    )

    sample_diag = plot_sample_episode(
        pred_all=pred,
        target_all=window_data.y,
        meta=window_data,
        tls_ids=tls_ids,
        output_path=os.path.join(ANALYSIS_DIR, "predicted_vs_actual_sample_episode.png"),
    )

    for k, v in sample_diag.items():
        summary_df[k] = v

    plot_mae_per_intersection(
        per_tls_df,
        output_path=os.path.join(ANALYSIS_DIR, "mae_per_intersection.png"),
    )

    summary_path = os.path.join(RESULTS_DIR, "lstm_eval_summary.csv")
    per_tls_path = os.path.join(RESULTS_DIR, "lstm_eval_per_intersection_lane.csv")
    summary_df.to_csv(summary_path, index=False)
    per_tls_df.to_csv(per_tls_path, index=False)

    print("=" * 70)
    print("LSTM Evaluation Complete")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {per_tls_path}")
    print(f"Saved: {os.path.join(ANALYSIS_DIR, 'predicted_vs_actual_sample_episode.png')}")
    print(f"Saved: {os.path.join(ANALYSIS_DIR, 'mae_per_intersection.png')}")


if __name__ == "__main__":
    main()
