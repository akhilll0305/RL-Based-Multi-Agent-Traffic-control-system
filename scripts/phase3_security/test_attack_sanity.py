"""
Manual attack sanity test for Step 4.4.

Goal:
- Pick a clean validation sequence where a lane is around 5 cars
- Inject fake value 18 at next step
- Verify z-score trigger (> threshold)
- Predict with LSTM from clean history
- Replace attacked value with LSTM prediction

Outputs:
- results_security/lstm_attack_sanity.csv
- analysis_security/lstm_eval/attack_sanity_case.png
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
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic_rl.security.lstm import TrafficLSTM
from train_lstm import load_split_dataset, episode_slices


VAL_STATES_PATH = "data/security/val_states.npy"
VAL_META_PATH = "data/security/val_metadata.npz"
CHECKPOINT_PATH = "outputs/checkpoints/security/lstm_predictor.pth"
OUT_CSV = "outputs/results/security/lstm_attack_sanity.csv"
OUT_PLOT = "outputs/analysis/security/lstm_eval/attack_sanity_case.png"

LANE_NAMES = ["north", "south", "east", "west"]


def find_sanity_case(
    states: np.ndarray,
    episode_lengths: np.ndarray,
    window_size: int,
    target_center: float = 5.0,
    tolerance: float = 1.5,
) -> Tuple[int, int, int, int]:
    """
    Find a case with enough history where lane value near target_center.

    Returns:
        episode_idx, intersection_idx, lane_idx, t
    where t is the attacked next-step index in episode-local timeline.
    """
    for ep_idx, (start, end) in enumerate(episode_slices(episode_lengths)):
        ep = states[start:end]  # (steps, 8, 4)
        steps = ep.shape[0]
        if steps <= window_size + 2:
            continue

        for inter_idx in range(ep.shape[1]):
            for lane_idx in range(ep.shape[2]):
                series = ep[:, inter_idx, lane_idx]
                for t in range(window_size, steps):
                    prev_val = float(series[t - 1])
                    if abs(prev_val - target_center) <= tolerance:
                        return ep_idx, inter_idx, lane_idx, t

    raise ValueError("Could not find a suitable sanity case around queue~5. Try larger tolerance.")


def compute_zscore(attacked_value: float, history_lane: np.ndarray, eps: float = 1e-6) -> float:
    """Compute anomaly z-score from lane history window."""
    mean = float(np.mean(history_lane))
    std = float(np.std(history_lane))
    return abs(attacked_value - mean) / (std + eps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual LSTM attack sanity test")
    parser.add_argument("--val-states", type=str, default=VAL_STATES_PATH)
    parser.add_argument("--val-meta", type=str, default=VAL_META_PATH)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--attack-value", type=float, default=18.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)

    bundle = load_split_dataset(args.val_states, args.val_meta)
    meta = np.load(args.val_meta, allow_pickle=False)
    tls_ids = [str(x) for x in meta["tls_ids"]]

    ep_idx, inter_idx, lane_idx, t = find_sanity_case(
        states=bundle.states,
        episode_lengths=bundle.episode_lengths,
        window_size=args.window_size,
    )

    ep_start, ep_end = episode_slices(bundle.episode_lengths)[ep_idx]
    ep = bundle.states[ep_start:ep_end]

    history = ep[t - args.window_size : t, inter_idx, :]  # (window, 4)
    clean_next = ep[t, inter_idx, :].copy()

    attacked_next = clean_next.copy()
    attacked_next[lane_idx] = float(args.attack_value)

    z = compute_zscore(attacked_next[lane_idx], history[:, lane_idx])
    triggered = z > args.z_threshold

    model = TrafficLSTM.load(args.checkpoint)
    pred_next = model.predict(history)

    corrected_next = attacked_next.copy()
    if triggered:
        corrected_next[lane_idx] = pred_next[lane_idx]

    row = {
        "episode_idx": ep_idx,
        "tls_id": tls_ids[inter_idx],
        "lane": LANE_NAMES[lane_idx],
        "window_size": args.window_size,
        "z_threshold": args.z_threshold,
        "z_score": z,
        "triggered": bool(triggered),
        "clean_prev_value": float(history[-1, lane_idx]),
        "clean_next_value": float(clean_next[lane_idx]),
        "attacked_next_value": float(attacked_next[lane_idx]),
        "lstm_predicted_value": float(pred_next[lane_idx]),
        "corrected_next_value": float(corrected_next[lane_idx]),
        "abs_error_attack_vs_clean": float(abs(attacked_next[lane_idx] - clean_next[lane_idx])),
        "abs_error_corrected_vs_clean": float(abs(corrected_next[lane_idx] - clean_next[lane_idx])),
    }

    pd.DataFrame([row]).to_csv(OUT_CSV, index=False)

    # Plot lane trend around the event.
    lane_series = ep[:, inter_idx, lane_idx]
    left = max(0, t - 25)
    right = min(ep.shape[0], t + 15)
    x = np.arange(left, right)
    y = lane_series[left:right]

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.plot(x, y, label="clean lane series", linewidth=1.8)
    ax.scatter([t], [clean_next[lane_idx]], label="clean next", s=60)
    ax.scatter([t], [attacked_next[lane_idx]], label="attacked next (18)", s=70)
    ax.scatter([t], [corrected_next[lane_idx]], label="corrected next (LSTM)", s=70)
    ax.axvline(t, linestyle="--", linewidth=1.2)
    ax.set_title(f"Attack Sanity Case: {tls_ids[inter_idx]} - {LANE_NAMES[lane_idx]} (episode {ep_idx})")
    ax.set_xlabel("Episode step")
    ax.set_ylabel("Queue value")
    ax.legend()
    fig.savefig(OUT_PLOT, dpi=160)
    plt.close(fig)

    print("=" * 70)
    print("LSTM Attack Sanity Test")
    print("=" * 70)
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {OUT_PLOT}")


if __name__ == "__main__":
    main()
