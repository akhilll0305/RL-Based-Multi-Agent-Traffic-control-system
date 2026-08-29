"""
8-Intersection Grouped Cooperative Training Script
Stage 1: Two groups of 4 agents, cooperative within groups, no cross-group info.

Fine-tunes from existing 700-episode cooperative checkpoints.
Group A (tls_1-4) loads from outputs/checkpoints/cooperative/tls_{1-4}_final.pth
Group B (tls_5-8) loads from same checkpoints (mirror: tls_1→5, 2→6, 3→7, 4→8)
"""

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------


import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from traffic_rl.envs.grid8 import EightIntersectionEnv
from traffic_rl.core.agent import DDQNAgent
from traffic_rl.sumo_gen.grid8 import generate_all as generate_sumo


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CHECKPOINT_DIR = 'outputs/checkpoints/grid8'
RESULTS_DIR = 'outputs/results/grid8'

# Mapping: which cooperative checkpoint each TLS loads from
WEIGHT_MAP = {
    'tls_1': 'tls_1', 'tls_2': 'tls_2', 'tls_3': 'tls_3', 'tls_4': 'tls_4',
    'tls_5': 'tls_1', 'tls_6': 'tls_2', 'tls_7': 'tls_3', 'tls_8': 'tls_4'
}


# ────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────
def train(env, agents, num_episodes=300, target_update_freq=10,
          save_freq=20, resume_from=0):
    """Train 8 agents with grouped cooperation."""

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    if resume_from > 0:
        print(f"GROUPED COOPERATIVE TRAINING: Episodes {resume_from+1}-{resume_from+num_episodes} (Resuming)")
    else:
        print(f"GROUPED COOPERATIVE TRAINING: {num_episodes} Episodes")
    print("  Group A: tls_1-4 | Group B: tls_5-8")
    print("  Cooperation WITHIN groups only")
    print("=" * 70)

    history = {
        'episode_rewards': [],
        'group_a_rewards': [],
        'group_b_rewards': [],
        'per_intersection': {tls: [] for tls in env.tls_ids}
    }

    for episode in tqdm(range(num_episodes), desc="Training"):
        actual_ep = resume_from + episode + 1
        states = env.reset()
        ep_reward = {tls: 0.0 for tls in env.tls_ids}
        done = False

        while not done:
            actions = {}
            for tls in env.tls_ids:
                actions[tls] = agents[tls].select_action(states[tls], training=True)

            next_states, rewards, done, info = env.step(actions)

            for tls in env.tls_ids:
                agents[tls].memory.store(
                    states[tls], actions[tls], rewards[tls],
                    next_states[tls], done
                )
                agents[tls].train()
                ep_reward[tls] += rewards[tls]

            states = next_states

        # Target network updates
        if (episode + 1) % target_update_freq == 0:
            for tls in env.tls_ids:
                agents[tls].update_target_network()

        # Epsilon decay
        for tls in env.tls_ids:
            agents[tls].decay_epsilon()

        # Record metrics
        network_reward = sum(ep_reward.values())
        group_a_reward = sum(ep_reward[t] for t in env.group_a)
        group_b_reward = sum(ep_reward[t] for t in env.group_b)

        history['episode_rewards'].append(network_reward)
        history['group_a_rewards'].append(group_a_reward)
        history['group_b_rewards'].append(group_b_reward)
        for tls in env.tls_ids:
            history['per_intersection'][tls].append(ep_reward[tls])

        # Checkpoint
        if (episode + 1) % save_freq == 0:
            for tls in env.tls_ids:
                agents[tls].save(f'{CHECKPOINT_DIR}/{tls}_episode_{actual_ep}.pth')
            print(f"\n✓ Checkpoint saved at episode {actual_ep}")

    env.close()

    # Save finals
    for tls in env.tls_ids:
        agents[tls].save(f'{CHECKPOINT_DIR}/{tls}_final.pth')

    # Save CSV history
    df = pd.DataFrame({
        'Network_Reward': history['episode_rewards'],
        'Group_A_Reward': history['group_a_rewards'],
        'Group_B_Reward': history['group_b_rewards'],
    })
    for tls in env.tls_ids:
        df[tls] = history['per_intersection'][tls]
    df.to_csv(f'{RESULTS_DIR}/training_history.csv', index=False)

    print("\n✅ Training complete!")
    print(f"Final models: {CHECKPOINT_DIR}/")
    print(f"History CSV:  {RESULTS_DIR}/training_history.csv")

    return history


# ────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────
def evaluate(env, agents, num_episodes=20):
    """Evaluate 8-agent system."""

    print("\n" + "=" * 70)
    print(f"EVALUATION: {num_episodes} Episodes")
    print("=" * 70)

    ep_rewards = {tls: [] for tls in env.tls_ids}
    network_rewards = []
    group_a_rewards = []
    group_b_rewards = []
    wait_times = []

    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        states = env.reset()
        ep_reward = {tls: 0.0 for tls in env.tls_ids}
        done = False

        while not done:
            actions = {tls: agents[tls].select_action(states[tls], training=False)
                       for tls in env.tls_ids}
            next_states, rewards, done, info = env.step(actions)
            for tls in env.tls_ids:
                ep_reward[tls] += rewards[tls]
            if done:
                wait_times.append(info['avg_waiting_time'])
            states = next_states

        for tls in env.tls_ids:
            ep_rewards[tls].append(ep_reward[tls])
        net = sum(ep_reward.values())
        network_rewards.append(net)
        group_a_rewards.append(sum(ep_reward[t] for t in env.group_a))
        group_b_rewards.append(sum(ep_reward[t] for t in env.group_b))

    env.close()

    # Print results
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)

    print("\nPer-Intersection:")
    for tls in env.tls_ids:
        avg = np.mean(ep_rewards[tls])
        std = np.std(ep_rewards[tls])
        group = "A" if tls in env.group_a else "B"
        print(f"  {tls} (Group {group}): {avg:.1f} ± {std:.1f}")

    print(f"\nGroup A Avg: {np.mean(group_a_rewards)/4:.1f}")
    print(f"Group B Avg: {np.mean(group_b_rewards)/4:.1f}")
    print(f"\nNetwork Total: {np.mean(network_rewards):.1f} ± {np.std(network_rewards):.1f}")
    print(f"Avg per Intersection: {np.mean(network_rewards)/8:.1f}")
    print(f"Avg Waiting Time: {np.mean(wait_times):.2f}s")
    print("=" * 70)

    return {
        'per_intersection': ep_rewards,
        'network_rewards': network_rewards,
        'group_a_rewards': group_a_rewards,
        'group_b_rewards': group_b_rewards,
        'waiting_times': wait_times
    }


# ────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────
def plot_training(history_csv=None):
    """Plot training curves from history CSV."""
    csv_path = history_csv or f'{RESULTS_DIR}/training_history.csv'
    if not os.path.exists(csv_path):
        print("No training history found.")
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('8-Intersection Grouped Cooperative Training', fontsize=16, fontweight='bold')

    episodes = range(1, len(df) + 1)

    # 1) Network reward
    axes[0, 0].plot(episodes, df['Network_Reward'], alpha=0.3, color='blue')
    axes[0, 0].plot(episodes, df['Network_Reward'].rolling(20).mean(),
                    color='red', linewidth=2, label='MA-20')
    axes[0, 0].set_title('Network Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward (8 agents)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2) Group comparison
    axes[0, 1].plot(episodes, df['Group_A_Reward'].rolling(20).mean(),
                    label='Group A', linewidth=2)
    axes[0, 1].plot(episodes, df['Group_B_Reward'].rolling(20).mean(),
                    label='Group B', linewidth=2)
    axes[0, 1].set_title('Group A vs Group B')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Group Reward (4 agents)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3) Per-intersection (Group A)
    for tls in ['tls_1', 'tls_2', 'tls_3', 'tls_4']:
        if tls in df.columns:
            axes[1, 0].plot(episodes, df[tls].rolling(20).mean(), label=tls, linewidth=1.5)
    axes[1, 0].set_title('Group A Per-Intersection')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # 4) Per-intersection (Group B)
    for tls in ['tls_5', 'tls_6', 'tls_7', 'tls_8']:
        if tls in df.columns:
            axes[1, 1].plot(episodes, df[tls].rolling(20).mean(), label=tls, linewidth=1.5)
    axes[1, 1].set_title('Group B Per-Intersection')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f'{RESULTS_DIR}/training_curves_8int.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to {out_path}")
    plt.close()


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def main(args):
    print("=" * 70)
    print("8-Intersection Grouped Cooperative DDQN Traffic Control")
    print("=" * 70)

    set_seed(args.seed)

    # Generate SUMO files if needed
    if not os.path.exists('sumo_files/grid8/network.net.xml'):
        print("\nGenerating SUMO network files...")
        generate_sumo()

    # Create environment
    env = EightIntersectionEnv(
        use_gui=args.gui,
        num_seconds=args.num_seconds,
        delta_time=args.delta_time
    )
    print(f"  Episode duration: {args.num_seconds}s  |  Delta-time: {args.delta_time}s  |  Steps/ep: {args.num_seconds // args.delta_time}")

    state_dim = env.get_state_dim()   # 8
    action_dim = env.get_action_dim()  # 2

    print(f"\nEnvironment:")
    print(f"  Intersections: 8 (2 groups of 4)")
    print(f"  State dim: {state_dim} (6 local + 2 neighbor within group)")
    print(f"  Action dim: {action_dim}")

    # Create 8 agents
    print(f"\nInitializing 8 agents...")
    agents = {}
    for tls in env.tls_ids:
        agents[tls] = DDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=args.learning_rate,
            epsilon_start=args.epsilon,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )

    # Load weights
    if args.resume_from > 0:
        # Resume from 8-intersection checkpoint
        for tls in env.tls_ids:
            path = f'{CHECKPOINT_DIR}/{tls}_episode_{args.resume_from}.pth'
            if os.path.exists(path):
                agents[tls].load(path)
                decay = 0.995 ** args.resume_from
                agents[tls].epsilon = max(args.epsilon * decay, agents[tls].epsilon_min)
                print(f"  ✓ {tls}: Resumed from episode {args.resume_from}")
            else:
                print(f"  ❌ {tls}: Checkpoint not found at {path}")
                return

    elif args.load_final:
        # Load 8-intersection finals
        for tls in env.tls_ids:
            path = f'{CHECKPOINT_DIR}/{tls}_final.pth'
            if os.path.exists(path):
                agents[tls].load(path)
                print(f"  ✓ {tls}: Loaded final model")
            else:
                print(f"  ⚠ {tls}: Final model not found")

    elif not args.from_scratch:
        # Fine-tune from cooperative 4-agent checkpoints
        print("\n" + "="*70)
        print("  FINE-TUNING from 700-episode cooperative checkpoints")
        print("  NOT training from scratch — loaded weights carry prior knowledge")
        print("="*70)
        for tls in env.tls_ids:
            source = WEIGHT_MAP[tls]
            path = f'outputs/checkpoints/cooperative/{source}_final.pth'
            if os.path.exists(path):
                agents[tls].load(path)
                agents[tls].epsilon = args.epsilon
                group = 'A' if tls in env.group_a else 'B'
                print(f"  ✓ {tls} (Group {group}): weights ← {source}_final.pth  |  ε={args.epsilon}")
            else:
                print(f"  ⚠ {tls}: Cooperative checkpoint not found at {path}")
                print(f"       Starting with random weights")

    else:
        print("  Starting all agents from scratch (random weights)")

    # ─── Run mode ───
    if args.mode in ('train', 'all'):
        train(env, agents, num_episodes=args.episodes,
              target_update_freq=10, save_freq=20,
              resume_from=args.resume_from)
        plot_training()

    if args.mode in ('evaluate', 'all'):
        evaluate(env, agents, num_episodes=args.eval_episodes)

    print("\n" + "=" * 70)
    print("✅ 8-Intersection Experiment Complete!")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='8-Intersection Grouped Cooperative DDQN Traffic Control')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'all'],
                        help='Mode: train, evaluate, or all')
    parser.add_argument('--episodes', type=int, default=300,
                        help='Number of training episodes (default: 300)')
    parser.add_argument('--eval-episodes', type=int, default=20,
                        help='Number of evaluation episodes (default: 20)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate for fine-tuning (default: 0.0001)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Initial epsilon for fine-tuning (default: 0.1)')
    parser.add_argument('--resume-from', type=int, default=0,
                        help='Resume from episode checkpoint')
    parser.add_argument('--load-final', action='store_true',
                        help='Load final 8-intersection models')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Train from scratch (ignore cooperative checkpoints)')
    parser.add_argument('--gui', action='store_true',
                        help='Use SUMO GUI')
    parser.add_argument('--num-seconds', type=int, default=1800,
                        help='Episode duration in seconds (default: 1800 for fast fine-tuning)')
    parser.add_argument('--delta-time', type=int, default=5,
                        help='Seconds between agent decisions (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()
    main(args)
