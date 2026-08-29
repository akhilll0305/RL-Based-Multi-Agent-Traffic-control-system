"""
Validate collected baseline queue-state data for Security Phase.

Checks performed:
1) File existence and shape sanity
2) Numeric integrity (NaN/Inf)
3) Value range summary
4) Episode boundary consistency
5) Per-intersection statistics
6) Quick visualization outputs

Outputs:
- data_security/baseline_validation_summary.csv
- data_security/validation/queue_distributions.png
- data_security/validation/sample_episode_totals.png
- data_security/validation/sample_tls_lane_trends.png
"""

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------


import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_STATES_PATH = "data/security/baseline_states.npy"
DEFAULT_META_PATH = "data/security/baseline_metadata.npz"
DEFAULT_OUTPUT_DIR = "data/security/validation"


def _check_file_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_data(states_path: str, meta_path: str):
    _check_file_exists(states_path, "states file")
    _check_file_exists(meta_path, "metadata file")

    states = np.load(states_path)
    meta = np.load(meta_path, allow_pickle=False)

    required_keys = ["episode_lengths", "episode_start_indices", "tls_ids"]
    for key in required_keys:
        if key not in meta:
            raise KeyError(f"Missing metadata key: {key}")

    episode_lengths = meta["episode_lengths"]
    episode_start_indices = meta["episode_start_indices"]
    tls_ids = [str(x) for x in meta["tls_ids"]]

    return states, episode_lengths, episode_start_indices, tls_ids


def run_sanity_checks(
    states: np.ndarray,
    episode_lengths: np.ndarray,
    episode_start_indices: np.ndarray,
    expected_intersections: int,
    expected_features: int,
) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}

    checks["states_is_3d"] = states.ndim == 3
    checks["feature_dims_match"] = (
        states.shape[1] == expected_intersections and states.shape[2] == expected_features
        if states.ndim == 3
        else False
    )

    checks["all_finite"] = bool(np.isfinite(states).all())
    checks["no_negative_values"] = bool((states >= 0).all())

    total_steps = states.shape[0] if states.ndim == 3 else -1
    checks["episode_lengths_sum_match"] = int(episode_lengths.sum()) == int(total_steps)
    checks["episode_count_match"] = len(episode_lengths) == len(episode_start_indices)

    starts_expected = np.cumsum(np.concatenate(([0], episode_lengths[:-1]))).astype(np.int32)
    checks["episode_starts_match"] = bool(np.array_equal(episode_start_indices.astype(np.int32), starts_expected))

    return checks


def build_summary_df(states: np.ndarray, tls_ids: list[str]) -> pd.DataFrame:
    rows = []
    for idx, tls in enumerate(tls_ids):
        values = states[:, idx, :].reshape(-1)
        rows.append(
            {
                "tls_id": tls,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
            }
        )
    return pd.DataFrame(rows)


def save_visualizations(states: np.ndarray, tls_ids: list[str], episode_lengths: np.ndarray, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 1) Distribution plot: one histogram per intersection
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for i, ax in enumerate(axes.ravel()):
        values = states[:, i, :].reshape(-1)
        bins = np.arange(values.min(), values.max() + 2) - 0.5
        ax.hist(values, bins=bins, alpha=0.85)
        ax.set_title(tls_ids[i])
        ax.set_xlabel("Queue value")
        ax.set_ylabel("Count")
    fig.suptitle("Queue Value Distribution by Intersection")
    fig.savefig(os.path.join(output_dir, "queue_distributions.png"), dpi=160)
    plt.close(fig)

    # 2) Temporal trend: first episode total queue per intersection
    first_len = int(episode_lengths[0]) if len(episode_lengths) > 0 else min(200, states.shape[0])
    ep0 = states[:first_len]  # shape (steps, 8, 4)
    ep0_totals = ep0.sum(axis=2)  # shape (steps, 8)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for i, tls in enumerate(tls_ids):
        ax.plot(ep0_totals[:, i], label=tls, linewidth=1.1)
    ax.set_title("Sample Episode 1: Total Queue per Intersection")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total queue (sum of 4 lanes)")
    ax.legend(ncol=4, fontsize=8)
    fig.savefig(os.path.join(output_dir, "sample_episode_totals.png"), dpi=160)
    plt.close(fig)

    # 3) Lane-level trend for one representative intersection
    tls_index = 0
    lane_names = ["north", "south", "east", "west"]

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for lane_idx, lane_name in enumerate(lane_names):
        ax.plot(ep0[:, tls_index, lane_idx], label=lane_name, linewidth=1.3)
    ax.set_title(f"Sample Episode 1: Lane Queues for {tls_ids[tls_index]}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Queue value")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "sample_tls_lane_trends.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate baseline queue dataset")
    parser.add_argument("--states-path", type=str, default=DEFAULT_STATES_PATH)
    parser.add_argument("--meta-path", type=str, default=DEFAULT_META_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-intersections", type=int, default=8)
    parser.add_argument("--expected-features", type=int, default=4)
    args = parser.parse_args()

    states, ep_lengths, ep_starts, tls_ids = load_data(args.states_path, args.meta_path)

    checks = run_sanity_checks(
        states=states,
        episode_lengths=ep_lengths,
        episode_start_indices=ep_starts,
        expected_intersections=args.expected_intersections,
        expected_features=args.expected_features,
    )

    print("=" * 70)
    print("Baseline Data Validation")
    print("=" * 70)
    print(f"States shape: {states.shape}")
    print(f"Episodes    : {len(ep_lengths)}")
    print(f"Global min/max: {states.min():.3f} / {states.max():.3f}")
    print("\nSanity checks:")
    for key, ok in checks.items():
        print(f"  {key}: {'PASS' if ok else 'FAIL'}")

    if not all(checks.values()):
        raise ValueError("One or more sanity checks failed. Please inspect dataset and metadata.")

    summary_df = build_summary_df(states, tls_ids)

    os.makedirs(os.path.dirname(DEFAULT_STATES_PATH), exist_ok=True)
    summary_path = os.path.join(os.path.dirname(DEFAULT_STATES_PATH), "baseline_validation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    save_visualizations(states, tls_ids, ep_lengths, args.output_dir)

    print("\nSaved outputs:")
    print(f"  Summary CSV : {summary_path}")
    print(f"  Plots dir   : {args.output_dir}")
    print("Validation complete.")


if __name__ == "__main__":
    main()
