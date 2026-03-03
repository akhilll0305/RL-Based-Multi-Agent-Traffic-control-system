"""
Evaluation & Comparison Plots for Federated Hierarchical Training

Generates visualizations comparing:
  - Training reward curves (total, per-zone)
  - Queue length progression per intersection
  - Supervisor action distribution over time
  - Waiting time trends
  - FedAvg event markers
"""

import os
import csv
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠ matplotlib not installed. Install with: pip install matplotlib")


def load_history(csv_path='results/federated/training_history.csv'):
    """Load training history from CSV"""
    history = {
        'episodes': [], 'total_reward': [], 'zone_a_reward': [], 'zone_b_reward': [],
        'avg_waiting_time': [], 'avg_queue': [], 'epsilon_local': [], 'epsilon_supervisor': [],
        'sup_a_ns': [], 'sup_a_ew': [], 'sup_a_bal': [],
        'sup_b_ns': [], 'sup_b_ew': [], 'sup_b_bal': [],
    }
    queue_keys = [f'queue_tls_{i}' for i in range(1, 9)]
    for k in queue_keys:
        history[k] = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history['episodes'].append(int(row['Episode']))
            history['total_reward'].append(float(row['Total_Reward']))
            history['zone_a_reward'].append(float(row['Zone_A_Reward']))
            history['zone_b_reward'].append(float(row['Zone_B_Reward']))
            history['avg_waiting_time'].append(float(row['Avg_Waiting_Time']))
            history['avg_queue'].append(float(row['Avg_Queue']))
            history['epsilon_local'].append(float(row['Epsilon_Local']))
            history['epsilon_supervisor'].append(float(row['Epsilon_Supervisor']))
            history['sup_a_ns'].append(float(row['Sup_A_NS']))
            history['sup_a_ew'].append(float(row['Sup_A_EW']))
            history['sup_a_bal'].append(float(row['Sup_A_BAL']))
            history['sup_b_ns'].append(float(row['Sup_B_NS']))
            history['sup_b_ew'].append(float(row['Sup_B_EW']))
            history['sup_b_bal'].append(float(row['Sup_B_BAL']))
            for i in range(1, 9):
                key = f'queue_tls_{i}'
                history[key].append(float(row[f'Queue_TLS_{i}']))

    return history


def smooth(data, window=20):
    """Smoothing for noisy RL curves"""
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(data[start:i + 1]))
    return smoothed


def plot_training_rewards(history, output_dir='results/federated'):
    """Plot total and per-zone reward curves"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    eps = history['episodes']

    # Total reward
    axes[0].plot(eps, smooth(history['total_reward']), color='blue', linewidth=1.5, label='Total Reward')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Reward (Smoothed)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-zone rewards
    axes[1].plot(eps, smooth(history['zone_a_reward']), color='green', linewidth=1.5, label='Zone A')
    axes[1].plot(eps, smooth(history['zone_b_reward']), color='orange', linewidth=1.5, label='Zone B')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Zone Reward')
    axes[1].set_title('Per-Zone Rewards (Smoothed)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_rewards.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved training_rewards.png")


def plot_queue_and_waiting(history, output_dir='results/federated'):
    """Plot average queue and waiting time"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    eps = history['episodes']

    axes[0].plot(eps, smooth(history['avg_queue']), color='red', linewidth=1.5)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Avg Queue Length')
    axes[0].set_title('Average Queue Length Across 8 Intersections')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps, smooth(history['avg_waiting_time']), color='purple', linewidth=1.5)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Avg Waiting Time (s)')
    axes[1].set_title('Average Vehicle Waiting Time')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/queue_and_waiting.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved queue_and_waiting.png")


def plot_per_intersection_queues(history, output_dir='results/federated'):
    """Plot queue length per intersection (Zone A vs Zone B)"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    eps = history['episodes']

    # Zone A intersections
    colors_a = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, i in enumerate([1, 2, 3, 4]):
        key = f'queue_tls_{i}'
        axes[0].plot(eps, smooth(history[key], 30), linewidth=1.2,
                     color=colors_a[idx], label=f'TLS {i}')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Queue Length')
    axes[0].set_title('Zone A (TLS 1-4) Queue Lengths')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zone B intersections
    colors_b = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for idx, i in enumerate([5, 6, 7, 8]):
        key = f'queue_tls_{i}'
        axes[1].plot(eps, smooth(history[key], 30), linewidth=1.2,
                     color=colors_b[idx], label=f'TLS {i}')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Queue Length')
    axes[1].set_title('Zone B (TLS 5-8) Queue Lengths')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_intersection_queues.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved per_intersection_queues.png")


def plot_supervisor_actions(history, output_dir='results/federated'):
    """Plot supervisor action distribution over training"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    eps = history['episodes']

    # Supervisor A
    axes[0].stackplot(eps,
                      smooth(history['sup_a_ns'], 30),
                      smooth(history['sup_a_ew'], 30),
                      smooth(history['sup_a_bal'], 30),
                      labels=['NS Priority', 'EW Priority', 'Balanced'],
                      colors=['#2ecc71', '#e74c3c', '#3498db'],
                      alpha=0.7)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Action Proportion')
    axes[0].set_title('Supervisor A Action Distribution')
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(0, 1)

    # Supervisor B
    axes[1].stackplot(eps,
                      smooth(history['sup_b_ns'], 30),
                      smooth(history['sup_b_ew'], 30),
                      smooth(history['sup_b_bal'], 30),
                      labels=['NS Priority', 'EW Priority', 'Balanced'],
                      colors=['#2ecc71', '#e74c3c', '#3498db'],
                      alpha=0.7)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Action Proportion')
    axes[1].set_title('Supervisor B Action Distribution')
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/supervisor_actions.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved supervisor_actions.png")


def plot_epsilon_decay(history, output_dir='results/federated'):
    """Plot epsilon decay for local agents and supervisors"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    eps = history['episodes']
    ax.plot(eps, history['epsilon_local'], color='blue', linewidth=1.5, label='Local Agents')
    ax.plot(eps, history['epsilon_supervisor'], color='red', linewidth=1.5, label='Supervisors')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/epsilon_decay.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved epsilon_decay.png")


def generate_all_plots(csv_path='results/federated/training_history.csv'):
    """Generate all evaluation plots from training history"""
    if not os.path.exists(csv_path):
        print(f"  ⚠ Training history not found at {csv_path}")
        print("  Run training first: python main_federated.py")
        return

    print("=" * 70)
    print("  GENERATING EVALUATION PLOTS")
    print("=" * 70)

    history = load_history(csv_path)
    output_dir = os.path.dirname(csv_path)

    plot_training_rewards(history, output_dir)
    plot_queue_and_waiting(history, output_dir)
    plot_per_intersection_queues(history, output_dir)
    plot_supervisor_actions(history, output_dir)
    plot_epsilon_decay(history, output_dir)

    print("\n  All plots saved to", output_dir)
    print("=" * 70)


if __name__ == '__main__':
    generate_all_plots()
