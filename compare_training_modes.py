"""
Comparison: From-Scratch (700 eps) vs Fine-Tuned (200 eps) Federated Training

Generates side-by-side comparison plots and a summary report.
"""

import os
import csv
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Install with: pip install matplotlib")


def load_history(csv_path):
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
            for k in queue_keys:
                col = k.replace('queue_tls_', 'Queue_TLS_')
                history[k].append(float(row[col]))

    return history


def smooth(data, window=10):
    """Moving average smoothing"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def generate_comparison_plots(scratch_path, finetune_path, output_dir='results/comparison'):
    """Generate all comparison plots"""
    if not HAS_MATPLOTLIB:
        print("Cannot generate plots without matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    scratch = load_history(scratch_path)
    finetune = load_history(finetune_path)

    print("=" * 70)
    print("  GENERATING COMPARISON PLOTS")
    print(f"  From-Scratch: {len(scratch['episodes'])} episodes")
    print(f"  Fine-Tuned:   {len(finetune['episodes'])} episodes")
    print("=" * 70)

    # Color scheme
    c_scratch = '#2196F3'   # Blue
    c_finetune = '#FF5722'  # Orange-Red

    # ==================== Plot 1: Reward Curves ====================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Full reward curves
    ax = axes[0]
    s_smooth = smooth(scratch['total_reward'], 20)
    f_smooth = smooth(finetune['total_reward'], 10)
    ax.plot(range(len(s_smooth)), s_smooth, color=c_scratch, linewidth=1.5, label='From Scratch (700 eps)')
    ax.plot(range(len(f_smooth)), f_smooth, color=c_finetune, linewidth=1.5, label='Fine-Tuned (200 eps)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (smoothed)')
    ax.set_title('Training Reward Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: First 200 episodes comparison
    ax = axes[1]
    s_first200 = smooth(scratch['total_reward'][:200], 10)
    f_first200 = smooth(finetune['total_reward'], 10)
    ax.plot(range(len(s_first200)), s_first200, color=c_scratch, linewidth=1.5, label='From Scratch (first 200)')
    ax.plot(range(len(f_first200)), f_first200, color=c_finetune, linewidth=1.5, label='Fine-Tuned (200 eps)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (smoothed)')
    ax.set_title('First 200 Episodes: Head-to-Head')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_reward_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [1/6] Reward comparison plot saved")

    # ==================== Plot 2: Queue Length ====================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    s_q = smooth(scratch['avg_queue'], 20)
    f_q = smooth(finetune['avg_queue'], 10)
    ax.plot(range(len(s_q)), s_q, color=c_scratch, linewidth=1.5, label='From Scratch')
    ax.plot(range(len(f_q)), f_q, color=c_finetune, linewidth=1.5, label='Fine-Tuned')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Queue Length')
    ax.set_title('Queue Length Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # First 200 comparison
    ax = axes[1]
    s_q200 = smooth(scratch['avg_queue'][:200], 10)
    f_q200 = smooth(finetune['avg_queue'], 10)
    ax.plot(range(len(s_q200)), s_q200, color=c_scratch, linewidth=1.5, label='From Scratch (first 200)')
    ax.plot(range(len(f_q200)), f_q200, color=c_finetune, linewidth=1.5, label='Fine-Tuned')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Queue Length')
    ax.set_title('First 200 Episodes: Queue Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_queue_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [2/6] Queue comparison plot saved")

    # ==================== Plot 3: Waiting Time ====================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    s_w = smooth(scratch['avg_waiting_time'], 20)
    f_w = smooth(finetune['avg_waiting_time'], 10)
    ax.plot(range(len(s_w)), s_w, color=c_scratch, linewidth=1.5, label='From Scratch')
    ax.plot(range(len(f_w)), f_w, color=c_finetune, linewidth=1.5, label='Fine-Tuned')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Waiting Time (s)')
    ax.set_title('Waiting Time Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    s_w200 = smooth(scratch['avg_waiting_time'][:200], 10)
    f_w200 = smooth(finetune['avg_waiting_time'], 10)
    ax.plot(range(len(s_w200)), s_w200, color=c_scratch, linewidth=1.5, label='From Scratch (first 200)')
    ax.plot(range(len(f_w200)), f_w200, color=c_finetune, linewidth=1.5, label='Fine-Tuned')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Waiting Time (s)')
    ax.set_title('First 200 Episodes: Waiting Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_waiting_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [3/6] Waiting time comparison plot saved")

    # ==================== Plot 4: Convergence Speed ====================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Show epsilon decay comparison
    ax.plot(scratch['episodes'], scratch['epsilon_local'], color=c_scratch,
            linewidth=1.5, linestyle='-', label='Scratch ε (local)')
    ax.plot(scratch['episodes'], scratch['epsilon_supervisor'], color=c_scratch,
            linewidth=1.5, linestyle='--', label='Scratch ε (supervisor)')
    ax.plot(finetune['episodes'], finetune['epsilon_local'], color=c_finetune,
            linewidth=1.5, linestyle='-', label='Fine-Tuned ε (local)')
    ax.plot(finetune['episodes'], finetune['epsilon_supervisor'], color=c_finetune,
            linewidth=1.5, linestyle='--', label='Fine-Tuned ε (supervisor)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate Decay: Scratch vs Fine-Tuned')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_epsilon_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [4/6] Epsilon decay comparison plot saved")

    # ==================== Plot 5: Per-Intersection Queue Heatmap ====================
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for idx, (data, title, ax) in enumerate([
        (scratch, 'From Scratch (700 eps)', axes[0]),
        (finetune, 'Fine-Tuned (200 eps)', axes[1])
    ]):
        queue_matrix = np.array([data[f'queue_tls_{i}'] for i in range(1, 9)])
        # Smooth each row
        window = 20 if idx == 0 else 10
        smoothed = np.array([smooth(row, window) for row in queue_matrix])
        im = ax.imshow(smoothed, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_yticks(range(8))
        ax.set_yticklabels([f'TLS {i}' for i in range(1, 9)])
        ax.set_xlabel('Episode')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Queue Length')

    plt.suptitle('Per-Intersection Queue Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_queue_heatmap_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [5/6] Queue heatmap comparison saved")

    # ==================== Plot 6: Summary Bar Chart ====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Compute final metrics (last 20 episodes)
    def final_avg(data, key, n=20):
        return np.mean(data[key][-n:])

    # Bar 1: Final avg queue
    labels = ['From Scratch\n(700 eps)', 'Fine-Tuned\n(200 eps)']
    s_final_q = final_avg(scratch, 'avg_queue')
    f_final_q = final_avg(finetune, 'avg_queue')
    bars = axes[0].bar(labels, [s_final_q, f_final_q], color=[c_scratch, c_finetune], width=0.5)
    axes[0].set_ylabel('Avg Queue Length')
    axes[0].set_title('Final Avg Queue (last 20 eps)')
    for bar, val in zip(bars, [s_final_q, f_final_q]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # Bar 2: Final avg waiting time
    s_final_w = final_avg(scratch, 'avg_waiting_time')
    f_final_w = final_avg(finetune, 'avg_waiting_time')
    bars = axes[1].bar(labels, [s_final_w, f_final_w], color=[c_scratch, c_finetune], width=0.5)
    axes[1].set_ylabel('Avg Waiting Time (s)')
    axes[1].set_title('Final Avg Wait (last 20 eps)')
    for bar, val in zip(bars, [s_final_w, f_final_w]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # Bar 3: Training efficiency (time to converge)
    training_hours = [10.45, 2.96]  # ~10h27m vs ~2h58m
    bars = axes[2].bar(labels, training_hours, color=[c_scratch, c_finetune], width=0.5)
    axes[2].set_ylabel('Training Time (hours)')
    axes[2].set_title('Training Time')
    for bar, val in zip(bars, training_hours):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{val:.1f}h', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Performance Summary: From Scratch vs Fine-Tuned', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_summary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [6/6] Summary comparison saved")

    print(f"\n  All plots saved to {output_dir}/")

    # ==================== Print Text Summary ====================
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<30} {'From Scratch':>15} {'Fine-Tuned':>15} {'Winner':>10}")
    print(f"  {'-' * 70}")
    print(f"  {'Episodes':.<30} {700:>15} {200:>15} {'FT':>10}")
    print(f"  {'Training Time':.<30} {'10h 27m':>15} {'2h 58m':>15} {'FT':>10}")

    s_q_final = final_avg(scratch, 'avg_queue')
    f_q_final = final_avg(finetune, 'avg_queue')
    winner_q = 'FT' if f_q_final <= s_q_final else 'Scratch'
    print(f"  {'Final Avg Queue (last 20)':.<30} {s_q_final:>15.2f} {f_q_final:>15.2f} {winner_q:>10}")

    s_w_final = final_avg(scratch, 'avg_waiting_time')
    f_w_final = final_avg(finetune, 'avg_waiting_time')
    winner_w = 'FT' if f_w_final <= s_w_final else 'Scratch'
    print(f"  {'Final Avg Wait (last 20)':.<30} {s_w_final:>15.2f}s {f_w_final:>15.2f}s {winner_w:>10}")

    s_r_final = final_avg(scratch, 'total_reward')
    f_r_final = final_avg(finetune, 'total_reward')
    winner_r = 'FT' if f_r_final >= s_r_final else 'Scratch'
    print(f"  {'Final Avg Reward (last 20)':.<30} {s_r_final:>15.1f} {f_r_final:>15.1f} {winner_r:>10}")

    # Speed to reach good performance (queue < 1.0)
    def first_below(data, key, threshold, window=10):
        vals = smooth(data[key], window)
        for i, v in enumerate(vals):
            if v < threshold:
                return i + window
        return None

    s_conv = first_below(scratch, 'avg_queue', 1.0)
    f_conv = first_below(finetune, 'avg_queue', 1.0)
    s_conv_str = str(s_conv) if s_conv else 'Never'
    f_conv_str = str(f_conv) if f_conv else 'Never'
    print(f"  {'Episode to Queue < 1.0':.<30} {s_conv_str:>15} {f_conv_str:>15} {'FT' if (f_conv or 999) < (s_conv or 999) else 'Scratch':>10}")

    # FedAvg stats
    print(f"\n  {'FedAvg Events':.<30} {'Scratch':>15} {'Fine-Tuned':>15}")
    print(f"  {'Intra-zone':.<30} {'70':>15} {'20':>15}")
    print(f"  {'Inter-zone':.<30} {'28':>15} {'8':>15}")

    speedup = 10.45 / 2.96
    print(f"\n  Training Speedup: {speedup:.1f}x faster with fine-tuning")
    print("=" * 70)

    # Save summary to text file
    summary_path = f'{output_dir}/comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("FEDERATED TRAINING COMPARISON: From Scratch vs Fine-Tuned\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"From Scratch: 700 episodes, ~10h 27min\n")
        f.write(f"Fine-Tuned:   200 episodes, ~2h 58min (from cooperative weights)\n\n")
        f.write(f"{'Metric':<35} {'Scratch':>12} {'Fine-Tuned':>12}\n")
        f.write(f"{'-' * 60}\n")
        f.write(f"{'Final Avg Queue (last 20 eps)':<35} {s_q_final:>12.2f} {f_q_final:>12.2f}\n")
        f.write(f"{'Final Avg Wait Time (last 20)':<35} {s_w_final:>12.2f}s {f_w_final:>12.2f}s\n")
        f.write(f"{'Final Avg Reward (last 20)':<35} {s_r_final:>12.1f} {f_r_final:>12.1f}\n")
        f.write(f"{'Episodes to Queue < 1.0':<35} {s_conv_str:>12} {f_conv_str:>12}\n")
        f.write(f"{'Training Time':<35} {'10h 27m':>12} {'2h 58m':>12}\n")
        f.write(f"{'Intra-zone FedAvg':<35} {'70':>12} {'20':>12}\n")
        f.write(f"{'Inter-zone FedAvg':<35} {'28':>12} {'8':>12}\n\n")
        f.write(f"Training Speedup: {speedup:.1f}x faster with fine-tuning\n\n")
        f.write("Conclusion: Fine-tuning from cooperative 4-intersection weights\n")
        f.write("provides a strong initialization, enabling the 8-intersection\n")
        f.write("federated system to converge faster with comparable or better\n")
        f.write("final performance.\n")

    print(f"  Summary saved to {summary_path}")


if __name__ == '__main__':
    generate_comparison_plots(
        scratch_path='results/federated/training_history.csv',
        finetune_path='results/federated_finetuned/training_history.csv',
        output_dir='results/comparison'
    )
