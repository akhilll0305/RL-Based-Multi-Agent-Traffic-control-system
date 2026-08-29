"""
Split baseline queue dataset into train/validation sets by episode (no leakage).

Input files:
- data_security/baseline_states.npy
- data_security/baseline_metadata.npz

Output files:
- data_security/train_states.npy
- data_security/val_states.npy
- data_security/train_metadata.npz
- data_security/val_metadata.npz
"""

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------


import argparse
import os

import numpy as np


DEFAULT_INPUT_STATES = "data/security/baseline_states.npy"
DEFAULT_INPUT_META = "data/security/baseline_metadata.npz"
DEFAULT_OUTPUT_DIR = "data/security"


def load_baseline_data(states_path: str, meta_path: str):
    """Load baseline states and required metadata."""
    if not os.path.exists(states_path):
        raise FileNotFoundError(f"States file not found: {states_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    states = np.load(states_path)
    meta = np.load(meta_path, allow_pickle=False)

    for key in ("episode_lengths", "episode_start_indices", "tls_ids"):
        if key not in meta:
            raise KeyError(f"Missing metadata key: {key}")

    episode_lengths = meta["episode_lengths"].astype(np.int32)
    episode_starts = meta["episode_start_indices"].astype(np.int32)
    tls_ids = meta["tls_ids"]

    if int(episode_lengths.sum()) != int(states.shape[0]):
        raise ValueError("Episode length sum does not match total step count in states.")

    return states, episode_lengths, episode_starts, tls_ids


def build_episode_slices(episode_lengths: np.ndarray) -> list[tuple[int, int]]:
    """Build (start, end) step slices for each episode."""
    slices = []
    start = 0
    for length in episode_lengths:
        end = start + int(length)
        slices.append((start, end))
        start = end
    return slices


def split_by_episode(
    states: np.ndarray,
    episode_lengths: np.ndarray,
    train_ratio: float,
):
    """
    Split entire dataset by episode boundaries.

    No shuffling is applied by default, preserving episode chronology.
    """
    num_episodes = len(episode_lengths)
    if num_episodes < 2:
        raise ValueError("Need at least 2 episodes to create train/val split.")

    train_episodes = int(round(num_episodes * train_ratio))
    train_episodes = max(1, min(train_episodes, num_episodes - 1))
    val_episodes = num_episodes - train_episodes

    episode_slices = build_episode_slices(episode_lengths)

    train_start = episode_slices[0][0]
    train_end = episode_slices[train_episodes - 1][1]

    val_start = episode_slices[train_episodes][0]
    val_end = episode_slices[-1][1]

    train_states = states[train_start:train_end]
    val_states = states[val_start:val_end]

    train_lengths = episode_lengths[:train_episodes].copy()
    val_lengths = episode_lengths[train_episodes:].copy()

    train_starts = np.cumsum(np.concatenate(([0], train_lengths[:-1]))).astype(np.int32)
    val_starts = np.cumsum(np.concatenate(([0], val_lengths[:-1]))).astype(np.int32)

    split_info = {
        "num_episodes": int(num_episodes),
        "train_episodes": int(train_episodes),
        "val_episodes": int(val_episodes),
        "train_steps": int(train_states.shape[0]),
        "val_steps": int(val_states.shape[0]),
    }

    return train_states, val_states, train_lengths, val_lengths, train_starts, val_starts, split_info


def save_split(
    output_dir: str,
    train_states: np.ndarray,
    val_states: np.ndarray,
    train_lengths: np.ndarray,
    val_lengths: np.ndarray,
    train_starts: np.ndarray,
    val_starts: np.ndarray,
    tls_ids: np.ndarray,
) -> None:
    """Save train/val arrays and metadata."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "train_states.npy"), train_states)
    np.save(os.path.join(output_dir, "val_states.npy"), val_states)

    np.savez(
        os.path.join(output_dir, "train_metadata.npz"),
        episode_lengths=train_lengths,
        episode_start_indices=train_starts,
        tls_ids=tls_ids,
    )

    np.savez(
        os.path.join(output_dir, "val_metadata.npz"),
        episode_lengths=val_lengths,
        episode_start_indices=val_starts,
        tls_ids=tls_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split baseline dataset by episodes (80/20 default)")
    parser.add_argument("--states-path", type=str, default=DEFAULT_INPUT_STATES)
    parser.add_argument("--meta-path", type=str, default=DEFAULT_INPUT_META)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1.")

    states, episode_lengths, _, tls_ids = load_baseline_data(args.states_path, args.meta_path)

    (
        train_states,
        val_states,
        train_lengths,
        val_lengths,
        train_starts,
        val_starts,
        info,
    ) = split_by_episode(
        states=states,
        episode_lengths=episode_lengths,
        train_ratio=args.train_ratio,
    )

    save_split(
        output_dir=args.output_dir,
        train_states=train_states,
        val_states=val_states,
        train_lengths=train_lengths,
        val_lengths=val_lengths,
        train_starts=train_starts,
        val_starts=val_starts,
        tls_ids=tls_ids,
    )

    print("=" * 70)
    print("Baseline Split Complete")
    print("=" * 70)
    print(f"Episodes total : {info['num_episodes']}")
    print(f"Train episodes : {info['train_episodes']}")
    print(f"Val episodes   : {info['val_episodes']}")
    print(f"Train states   : {train_states.shape}")
    print(f"Val states     : {val_states.shape}")
    print(f"Train steps    : {info['train_steps']}")
    print(f"Val steps      : {info['val_steps']}")
    print("Saved files:")
    print(f"  {os.path.join(args.output_dir, 'train_states.npy')}")
    print(f"  {os.path.join(args.output_dir, 'val_states.npy')}")
    print(f"  {os.path.join(args.output_dir, 'train_metadata.npz')}")
    print(f"  {os.path.join(args.output_dir, 'val_metadata.npz')}")


if __name__ == "__main__":
    main()
