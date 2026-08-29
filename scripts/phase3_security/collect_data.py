"""
Collect clean baseline queue-state sequences for Security Phase LSTM training.

This script:
1) Loads frozen trained Local-Supervisor system from checkpoints_supervisor/
2) Runs evaluation episodes with greedy policy (epsilon=0)
3) Records raw local queue values only (indices 0-3 of 6-dim state)
4) Saves dataset and episode-boundary metadata to data_security/

Output files:
- data_security/baseline_states.npy        shape: (total_steps, 8, 4)
- data_security/baseline_metadata.npz      episode lengths and start indices

Usage examples:
  python collect_baseline_data.py
  python collect_baseline_data.py --episodes 100 --num-seconds 1800
  python collect_baseline_data.py --episodes 100 --gui
"""

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------


import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from traffic_rl.core.agent import DDQNAgent
from traffic_rl.supervisor.agent import SupervisorAgent
from traffic_rl.envs.grid8_supervisor import SupervisorSumoEnvironment


CHECKPOINT_DIR_AGENTS = "outputs/checkpoints/supervisor/agents"
CHECKPOINT_DIR_SUPERVISORS = "outputs/checkpoints/supervisor/supervisors"
OUTPUT_DIR = "data/security"


def set_seed(seed: int = 42) -> None:
    """Set reproducible random seeds for numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _required_checkpoint_paths(tls_ids: List[str]) -> List[str]:
    """Return all required checkpoint file paths for baseline collection."""
    paths = [os.path.join(CHECKPOINT_DIR_AGENTS, f"{tls}_final.pth") for tls in tls_ids]
    paths.append(os.path.join(CHECKPOINT_DIR_SUPERVISORS, "supervisor_a_final.pth"))
    paths.append(os.path.join(CHECKPOINT_DIR_SUPERVISORS, "supervisor_b_final.pth"))
    return paths


def ensure_checkpoints_exist(tls_ids: List[str]) -> None:
    """Fail fast if any required trained model checkpoint is missing."""
    missing = [p for p in _required_checkpoint_paths(tls_ids) if not os.path.exists(p)]
    if missing:
        formatted = "\n".join(f"  - {m}" for m in missing)
        raise FileNotFoundError(
            "Missing required checkpoints for baseline collection:\n"
            f"{formatted}\n"
            "Please ensure Phase 2 final checkpoints exist in checkpoints_supervisor/."
        )


def load_trained_system(
    env: SupervisorSumoEnvironment,
) -> Tuple[Dict[str, DDQNAgent], SupervisorAgent, SupervisorAgent]:
    """
    Build agents/supervisors and load final frozen checkpoints.

    Returns:
        agents: dict tls_id -> DDQNAgent (7-dim input)
        supervisor_a: SupervisorAgent for Group A
        supervisor_b: SupervisorAgent for Group B
    """
    agents: Dict[str, DDQNAgent] = {}
    for tls in env.tls_ids:
        agent = DDQNAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            hidden_dim=128,
            learning_rate=1e-4,
            epsilon_start=0.0,
            epsilon_decay=1.0,
            epsilon_min=0.0,
        )
        agent.load(os.path.join(CHECKPOINT_DIR_AGENTS, f"{tls}_final.pth"))
        agent.epsilon = 0.0
        agents[tls] = agent

    supervisor_a = SupervisorAgent(
        group_tls_ids=env.group_a,
        state_dim_per_agent=env.get_local_state_dim(),
        hidden_dim=64,
        learning_rate=0.001,
        gamma=0.95,
        buffer_capacity=10_000,
        batch_size=64,
    )
    supervisor_b = SupervisorAgent(
        group_tls_ids=env.group_b,
        state_dim_per_agent=env.get_local_state_dim(),
        hidden_dim=64,
        learning_rate=0.001,
        gamma=0.95,
        buffer_capacity=10_000,
        batch_size=64,
    )

    supervisor_a.load(os.path.join(CHECKPOINT_DIR_SUPERVISORS, "supervisor_a_final.pth"))
    supervisor_b.load(os.path.join(CHECKPOINT_DIR_SUPERVISORS, "supervisor_b_final.pth"))

    return agents, supervisor_a, supervisor_b


def extract_queue_matrix(local_states: Dict[str, np.ndarray], tls_ids: List[str]) -> np.ndarray:
    """
    Convert raw local state dict to queue-only matrix.

    Args:
        local_states: {tls_id: np.array(6,)}
        tls_ids: fixed order of intersection IDs

    Returns:
        np.ndarray shape (8, 4) with queue values for [N,S,E,W].
    """
    return np.stack([local_states[tls][:4] for tls in tls_ids], axis=0).astype(np.float32)


def collect_baseline_data(
    episodes: int = 100,
    num_seconds: int = 1800,
    delta_time: int = 5,
    use_gui: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Run frozen policy and collect clean queue sequences.

    Returns:
        states: np.ndarray shape (total_steps, 8, 4)
        episode_lengths: np.ndarray shape (episodes,)
        episode_start_indices: np.ndarray shape (episodes,)
        tls_ids: ordered list of intersection IDs used in states axis-1
    """
    set_seed(seed)

    env = SupervisorSumoEnvironment(
        use_gui=use_gui,
        num_seconds=num_seconds,
        delta_time=delta_time,
    )

    ensure_checkpoints_exist(env.tls_ids)
    agents, supervisor_a, supervisor_b = load_trained_system(env)

    all_steps: List[np.ndarray] = []
    episode_lengths: List[int] = []
    episode_start_indices: List[int] = []

    try:
        for ep in tqdm(range(episodes), desc="Collecting baseline episodes"):
            local_states = env.reset()
            done = False
            steps_this_episode = 0

            while not done:
                # Record queue-only clean raw state before action selection.
                all_steps.append(extract_queue_matrix(local_states, env.tls_ids))
                steps_this_episode += 1

                signals_a = supervisor_a.get_signals({tls: local_states[tls] for tls in env.group_a})
                signals_b = supervisor_b.get_signals({tls: local_states[tls] for tls in env.group_b})
                enhanced = env.build_enhanced_states(local_states, signals_a, signals_b)

                actions = {
                    tls: agents[tls].select_action(enhanced[tls], training=False)
                    for tls in env.tls_ids
                }

                next_local, _, done, _ = env.step(actions)
                local_states = next_local

            episode_start = int(np.sum(episode_lengths)) if episode_lengths else 0
            episode_start_indices.append(episode_start)
            episode_lengths.append(steps_this_episode)

            if (ep + 1) % 10 == 0:
                print(
                    f"[Progress] Episodes: {ep + 1}/{episodes} | "
                    f"Total steps collected: {len(all_steps)}"
                )

    finally:
        env.close()

    states = np.stack(all_steps, axis=0).astype(np.float32)
    return (
        states,
        np.array(episode_lengths, dtype=np.int32),
        np.array(episode_start_indices, dtype=np.int32),
        env.tls_ids,
    )


def save_dataset(
    states: np.ndarray,
    episode_lengths: np.ndarray,
    episode_start_indices: np.ndarray,
    tls_ids: List[str],
) -> None:
    """Persist baseline data and metadata to data_security/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    states_path = os.path.join(OUTPUT_DIR, "baseline_states.npy")
    meta_path = os.path.join(OUTPUT_DIR, "baseline_metadata.npz")

    np.save(states_path, states)
    np.savez(
        meta_path,
        episode_lengths=episode_lengths,
        episode_start_indices=episode_start_indices,
        tls_ids=np.array(tls_ids),
    )

    print("\nSaved baseline dataset:")
    print(f"  States   : {states_path}")
    print(f"  Metadata : {meta_path}")
    print(f"  Shape    : {states.shape}  (steps, intersections, queue_features)")
    print(f"  Episodes : {len(episode_lengths)}")
    print(f"  Min/Max queue values: {states.min():.2f} / {states.max():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect clean baseline state sequences for security-phase LSTM training")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes to collect (default: 100)")
    parser.add_argument("--num-seconds", type=int, default=1800, help="Episode duration in SUMO seconds")
    parser.add_argument("--delta-time", type=int, default=5, help="Decision interval in seconds")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("Collecting clean baseline data for Security Phase")
    print("=" * 70)
    print(f"Episodes      : {args.episodes}")
    print(f"Episode length: {args.num_seconds}s")
    print(f"Delta-time    : {args.delta_time}s")
    print(f"Expected steps/episode ~ {args.num_seconds // args.delta_time}")

    states, episode_lengths, start_indices, tls_ids = collect_baseline_data(
        episodes=args.episodes,
        num_seconds=args.num_seconds,
        delta_time=args.delta_time,
        use_gui=args.gui,
        seed=args.seed,
    )
    save_dataset(states, episode_lengths, start_indices, tls_ids)


if __name__ == "__main__":
    main()
