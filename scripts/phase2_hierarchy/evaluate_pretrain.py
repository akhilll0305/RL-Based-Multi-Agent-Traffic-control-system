"""
Pre-Fine-Tuning Evaluation: 8-Intersection Grouped Cooperative System
- Runs N episodes (with optional SUMO GUI)
- Saves per-intersection results to CSV
- Compares against 4-intersection cooperative baseline
"""

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------


import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import traci.exceptions

from traffic_rl.envs.grid8 import EightIntersectionEnv
from traffic_rl.core.agent import DDQNAgent

# ── Weight mapping: Group B mirrors Group A ──────────────────────────
WEIGHT_MAP = {
    'tls_1': 'tls_1', 'tls_2': 'tls_2',
    'tls_3': 'tls_3', 'tls_4': 'tls_4',
    'tls_5': 'tls_1', 'tls_6': 'tls_2',
    'tls_7': 'tls_3', 'tls_8': 'tls_4',
}

# Known 4-intersection cooperative baseline (from 700-episode training)
BASELINE_4INT = {
    'tls_1': -585.8, 'tls_2': -585.8,
    'tls_3': -585.8, 'tls_4': -585.8,
    'avg_per_intersection': -585.8,
    'network_total': -585.8 * 4   # -2343.2
}

RESULTS_DIR = 'outputs/results/grid8'


# ────────────────────────────────────────────────────────────────────
def load_agents(env, epsilon=0.0):
    """Create 8 agents and load cooperative checkpoint weights."""
    agents = {}
    print("\nLoading cooperative weights into 8 agents...")
    for tls in env.tls_ids:
        agents[tls] = DDQNAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            hidden_dim=128,
            learning_rate=0.0001,
            epsilon_start=epsilon,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        src = WEIGHT_MAP[tls]
        path = f'outputs/checkpoints/cooperative/{src}_final.pth'
        if os.path.exists(path):
            agents[tls].load(path)
            agents[tls].epsilon = epsilon
            group = 'A' if tls in env.group_a else 'B'
            print(f"  ✓ {tls} (Group {group}) ← {path}")
        else:
            print(f"  ⚠ {tls}: checkpoint not found at {path}")
    return agents


# ────────────────────────────────────────────────────────────────────
def run_evaluation(env, agents, num_episodes=10):
    """Run evaluation episodes and return per-episode metrics."""
    print(f"\n{'='*65}")
    print(f"  Running {num_episodes} evaluation episodes")
    print(f"{'='*65}")

    ep_rewards   = {tls: [] for tls in env.tls_ids}
    network_rews = []
    group_a_rews = []
    group_b_rews = []
    wait_times   = []

    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        states = env.reset()
        ep_reward = {tls: 0.0 for tls in env.tls_ids}
        done = False

        while not done:
            actions = {
                tls: agents[tls].select_action(states[tls], training=False)
                for tls in env.tls_ids
            }
            states, rewards, done, info = env.step(actions)
            for tls in env.tls_ids:
                ep_reward[tls] += rewards[tls]
            if done:
                wait_times.append(info['avg_waiting_time'])

        for tls in env.tls_ids:
            ep_rewards[tls].append(ep_reward[tls])
        network_rews.append(sum(ep_reward.values()))
        group_a_rews.append(sum(ep_reward[t] for t in env.group_a))
        group_b_rews.append(sum(ep_reward[t] for t in env.group_b))

    env.close()

    return {
        'per_intersection': ep_rewards,
        'network':          network_rews,
        'group_a':          group_a_rews,
        'group_b':          group_b_rews,
        'wait_times':       wait_times,
    }


# ────────────────────────────────────────────────────────────────────
def print_summary(results, env):
    """Print evaluation summary to console."""
    print(f"\n{'='*65}")
    print("  EVALUATION SUMMARY  —  8-Intersection (Pre-Fine-Tuning)")
    print(f"{'='*65}")

    print("\nPer-Intersection Performance:")
    for tls in env.tls_ids:
        avg = np.mean(results['per_intersection'][tls])
        std = np.std(results['per_intersection'][tls])
        group = 'A' if tls in env.group_a else 'B'
        print(f"  {tls} (Group {group}): {avg:8.1f} ± {std:.1f}")

    avg_per_int = np.mean(results['network']) / 8
    print(f"\nGroup A Avg/intersection : {np.mean(results['group_a'])/4:.1f}")
    print(f"Group B Avg/intersection : {np.mean(results['group_b'])/4:.1f}")
    print(f"Network total reward     : {np.mean(results['network']):.1f}")
    print(f"Avg per intersection     : {avg_per_int:.1f}")
    print(f"Avg waiting time         : {np.mean(results['wait_times']):.2f}s")

    print(f"\n{'─'*65}")
    print("  vs 4-Intersection Cooperative Baseline")
    print(f"{'─'*65}")
    diff = avg_per_int - BASELINE_4INT['avg_per_intersection']
    pct  = diff / abs(BASELINE_4INT['avg_per_intersection']) * 100
    direction = '↑ better' if diff > 0 else '↓ worse'
    print(f"  4-int avg/intersection : {BASELINE_4INT['avg_per_intersection']:.1f}")
    print(f"  8-int avg/intersection : {avg_per_int:.1f}")
    print(f"  Difference             : {diff:+.1f}  ({pct:+.1f}%)  {direction}")
    print(f"{'='*65}")


# ────────────────────────────────────────────────────────────────────
def save_results(results, env, num_episodes):
    """Save per-episode results to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for ep in range(num_episodes):
        row = {
            'episode':        ep + 1,
            'network_reward': results['network'][ep],
            'group_a_reward': results['group_a'][ep],
            'group_b_reward': results['group_b'][ep],
            'avg_wait_time':  results['wait_times'][ep]
                              if ep < len(results['wait_times']) else None,
        }
        for tls in env.tls_ids:
            row[tls] = results['per_intersection'][tls][ep]
        rows.append(row)

    df = pd.DataFrame(rows)
    out = f'{RESULTS_DIR}/pretrain_eval_results.csv'
    df.to_csv(out, index=False)
    print(f"\n✅ Results saved → {out}")
    return df


# ────────────────────────────────────────────────────────────────────
def plot_comparison(results, env):
    """Generate comparison plots: 8-int (pre-FT) vs 4-int cooperative."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Per-intersection averages ──
    tls_8   = env.tls_ids
    avgs_8  = [np.mean(results['per_intersection'][t]) for t in tls_8]
    stds_8  = [np.std(results['per_intersection'][t])  for t in tls_8]

    tls_4       = ['tls_1', 'tls_2', 'tls_3', 'tls_4']
    avgs_4      = [BASELINE_4INT[t] for t in tls_4]
    avg_8_per   = np.mean(results['network']) / 8
    avg_4_per   = BASELINE_4INT['avg_per_intersection']

    colors_a = '#45B7D1'
    colors_b = '#9B59B6'
    color_4  = '#2ECC71'

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        '8-Intersection Grouped Cooperative  vs  4-Intersection Cooperative\n'
        '(Pre-Fine-Tuning, Transferred Weights)',
        fontsize=15, fontweight='bold'
    )

    # ── Plot 1: Per-intersection bar (8-int) ──
    ax = axes[0, 0]
    bar_colors = [colors_a if t in env.group_a else colors_b for t in tls_8]
    bars = ax.bar(tls_8, avgs_8, yerr=stds_8, color=bar_colors,
                  alpha=0.85, edgecolor='black', linewidth=1.2,
                  capsize=4, error_kw={'linewidth': 1.5})
    ax.axhline(avg_4_per, color=color_4, linewidth=2, linestyle='--',
               label=f'4-int baseline ({avg_4_per:.0f})')
    ax.set_title('8-Int Per-Intersection Reward', fontweight='bold')
    ax.set_ylabel('Avg Reward')
    ax.set_xlabel('Intersection')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    # colour legend patches
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=colors_a, label='Group A (tls 1-4)'),
                  Patch(facecolor=colors_b, label='Group B (tls 5-8)')]
    ax.legend(handles=legend_els + [
        plt.Line2D([0], [0], color=color_4, linestyle='--',
                   linewidth=2, label=f'4-int baseline ({avg_4_per:.0f})')
    ], fontsize=8, loc='lower right')
    for bar, val in zip(bars, avgs_8):
        ax.text(bar.get_x() + bar.get_width()/2, val - 60,
                f'{val:.0f}', ha='center', va='top',
                fontsize=7, fontweight='bold', color='white')

    # ── Plot 2: Average per-intersection comparison ──
    ax = axes[0, 1]
    systems = ['4-Int\nCooperative\n(Baseline)', '8-Int Group A\n(Pre-FT)',
               '8-Int Group B\n(Pre-FT)', '8-Int Overall\n(Pre-FT)']
    avg_a = np.mean(results['group_a']) / 4
    avg_b = np.mean(results['group_b']) / 4
    vals  = [avg_4_per, avg_a, avg_b, avg_8_per]
    clrs  = [color_4, colors_a, colors_b, '#E67E22']
    b2 = ax.bar(systems, vals, color=clrs, alpha=0.85,
                edgecolor='black', linewidth=1.2)
    ax.set_title('Avg Reward per Intersection', fontweight='bold')
    ax.set_ylabel('Avg Reward')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(b2, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val - 40,
                f'{val:.1f}', ha='center', va='top',
                fontsize=9, fontweight='bold', color='white')
    diff = avg_8_per - avg_4_per
    pct  = diff / abs(avg_4_per) * 100
    ax.text(0.5, 0.04,
            f'8-int vs 4-int: {diff:+.1f} ({pct:+.1f}%)',
            transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontweight='bold')

    # ── Plot 3: Episode-by-episode network reward ──
    ax = axes[1, 0]
    eps = range(1, len(results['network']) + 1)
    ax.plot(eps, results['network'], 'o-', color='#E67E22',
            linewidth=2, markersize=6, label='8-int network')
    ax.axhline(BASELINE_4INT['network_total'], color=color_4,
               linewidth=2, linestyle='--',
               label=f'4-int network total ({BASELINE_4INT["network_total"]:.0f})')
    ax.set_title('Network Total Reward per Episode', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Network Reward')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Plot 4: Group A vs Group B per episode ──
    ax = axes[1, 1]
    ax.plot(eps, [r/4 for r in results['group_a']], 'o-',
            color=colors_a, linewidth=2, markersize=6, label='Group A avg/int')
    ax.plot(eps, [r/4 for r in results['group_b']], 's-',
            color=colors_b, linewidth=2, markersize=6, label='Group B avg/int')
    ax.axhline(avg_4_per, color=color_4, linewidth=2, linestyle='--',
               label=f'4-int baseline ({avg_4_per:.0f})')
    ax.set_title('Group A vs Group B per Episode', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward per Intersection')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'{RESULTS_DIR}/pretrain_comparison.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved → {out}")
    plt.close()


# ────────────────────────────────────────────────────────────────────
def main(args):
    # ── GUI pass (optional visual run, short) ──
    if args.gui:
        print("\n" + "="*65)
        print("  SUMO GUI MODE  —  Visual inspection run")
        print("="*65)
        gui_env = EightIntersectionEnv(use_gui=True, num_seconds=3600, delta_time=5)
        gui_agents = load_agents(gui_env, epsilon=0.0)
        print(f"\nLaunching SUMO GUI for {args.gui_episodes} episode(s)...")
        print("Close the SUMO GUI window to advance to the next episode.\n")

        for ep in range(args.gui_episodes):
            print(f"  Episode {ep+1}/{args.gui_episodes}")
            try:
                states = gui_env.reset()
            except traci.exceptions.FatalTraCIError:
                print("  ⚠ SUMO GUI closed — skipping remaining GUI episodes.")
                break
            done = False
            ep_r = {tls: 0.0 for tls in gui_env.tls_ids}
            info = {'avg_waiting_time': 0.0}
            gui_closed = False
            while not done:
                try:
                    actions = {
                        tls: gui_agents[tls].select_action(states[tls], training=False)
                        for tls in gui_env.tls_ids
                    }
                    states, rewards, done, info = gui_env.step(actions)
                    for tls in gui_env.tls_ids:
                        ep_r[tls] += rewards[tls]
                except traci.exceptions.FatalTraCIError:
                    print(f"  ⚠ SUMO GUI window closed — ending episode {ep+1} early.")
                    gui_env.sumo_running = False
                    gui_closed = True
                    break
            print(f"    Network reward: {sum(ep_r.values()):.1f}  |  "
                  f"Avg wait: {info['avg_waiting_time']:.2f}s")
            if gui_closed:
                break

        try:
            gui_env.close()
        except Exception:
            pass
        print("\n✅ GUI run complete.")

    # ── Headless evaluation pass (save results + compare) ──
    print("\n" + "="*65)
    print("  HEADLESS EVALUATION  —  Saving results + comparison")
    print("="*65)
    eval_env = EightIntersectionEnv(use_gui=False, num_seconds=3600, delta_time=5)
    eval_agents = load_agents(eval_env, epsilon=0.0)

    results = run_evaluation(eval_env, eval_agents, num_episodes=args.episodes)
    print_summary(results, eval_env)
    save_results(results, eval_env, num_episodes=args.episodes)
    plot_comparison(results, eval_env)

    print("\n✅ All done! Files saved in outputs/results/grid8/")


# ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-FT evaluation + comparison for 8-intersection system')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of headless eval episodes (default: 10)')
    parser.add_argument('--gui', action='store_true',
                        help='Also run SUMO GUI visualisation first')
    parser.add_argument('--gui-episodes', type=int, default=2,
                        help='Number of episodes to show in GUI (default: 2)')
    args = parser.parse_args()
    main(args)
