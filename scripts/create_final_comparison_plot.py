"""
Create comprehensive visualization comparing:
1. Single-Agent
2. Multi-Agent Transfer Learning
3. Multi-Agent Fine-Tuned
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('results/multiagent', exist_ok=True)

# Data from evaluations
systems = ['Single-Agent\n(1 Intersection)', 'Multi-Agent\nTransfer', 'Multi-Agent\nFine-Tuned']

# Average rewards per intersection
rewards = [-4253.5, -1363.1, -560.8]

# Waiting times
waiting_times = [8.0, 0.34, 0.00]

# Per-intersection breakdown for multi-agent systems
transfer_rewards = {'tls_1': -1766.0, 'tls_2': -1377.0, 'tls_3': -1250.5, 'tls_4': -1059.0}
finetuned_rewards = {'tls_1': -807.5, 'tls_2': -663.5, 'tls_3': -448.0, 'tls_4': -324.0}

# Create figure with subplots
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Color scheme
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
colors_improvement = ['#95E1D3', '#F38181']

# ===== Plot 1: Average Reward per Intersection =====
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(systems, rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
ax1.set_title('Reward per Intersection\n(Higher is Better)', fontsize=14, fontweight='bold')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars1, rewards):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotations
ax1.annotate('68% better', xy=(0.5, -2800), xytext=(0.5, -2200),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'),
            fontsize=11, color='green', fontweight='bold', ha='center')
ax1.annotate('58.9% better', xy=(1.5, -1000), xytext=(1.5, -1600),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'),
            fontsize=11, color='green', fontweight='bold', ha='center')

# ===== Plot 2: Waiting Time Comparison =====
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(systems, waiting_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Average Waiting Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_title('Waiting Time per Intersection\n(Lower is Better)', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars2, waiting_times):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{val:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement percentage
ax2.text(1, 4.5, '95.8%\nreduction', ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), fontweight='bold')
ax2.text(2, 4.5, '100%\nreduction', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), fontweight='bold')

# ===== Plot 3: Improvement Metrics =====
ax3 = fig.add_subplot(gs[0, 2])
improvements = [68.0, 58.9]  # Single->Transfer, Transfer->Fine-tuned
improvement_labels = ['Single-Agent\n→\nTransfer', 'Transfer\n→\nFine-Tuned']
bars3 = ax3.bar(improvement_labels, improvements, color=['#4ECDC4', '#45B7D1'], 
                alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Performance Improvement (%)', fontsize=12, fontweight='bold')
ax3.set_title('Improvement at Each Stage\n(Reward-based)', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 80)
ax3.grid(axis='y', alpha=0.3)

for bar, val in zip(bars3, improvements):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='green')

# ===== Plot 4: Per-Intersection Breakdown =====
ax4 = fig.add_subplot(gs[1, :2])
intersections = ['TLS 1\n(Top-Left)', 'TLS 2\n(Top-Right)', 'TLS 3\n(Bottom-Left)', 'TLS 4\n(Bottom-Right)']
transfer_vals = [-1766.0, -1377.0, -1250.5, -1059.0]
finetuned_vals = [-807.5, -663.5, -448.0, -324.0]

x = np.arange(len(intersections))
width = 0.35

bars4a = ax4.bar(x - width/2, transfer_vals, width, label='Transfer Learning', 
                 color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
bars4b = ax4.bar(x + width/2, finetuned_vals, width, label='Fine-Tuned',
                 color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('Reward', fontsize=12, fontweight='bold')
ax4.set_title('Per-Intersection Performance Comparison\n(Fine-Tuning vs Transfer Learning)', 
              fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(intersections, fontsize=10)
ax4.legend(fontsize=11, loc='lower right')
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.grid(axis='y', alpha=0.3)

# Add improvement percentages above bars
improvements_per_tls = [54.3, 51.8, 64.2, 69.4]
for i, (transfer_bar, finetuned_bar, improvement) in enumerate(zip(bars4a, bars4b, improvements_per_tls)):
    transfer_height = transfer_bar.get_height()
    finetuned_height = finetuned_bar.get_height()
    mid_height = (transfer_height + finetuned_height) / 2
    ax4.annotate(f'+{improvement:.1f}%', 
                xy=(i, mid_height), 
                fontsize=10, ha='center', fontweight='bold',
                color='green',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# ===== Plot 5: Training Progress Summary =====
ax5 = fig.add_subplot(gs[1, 2])
stages = ['Episode 900\n(Single)', 'Transfer to\nMulti-Agent', 'After 100\nEpisodes Fine-Tune']
stage_rewards = [-4253.5, -1363.1, -560.8]
stage_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

ax5.plot(stages, stage_rewards, 'o-', linewidth=3, markersize=12, color='#2C3E50')
for i, (stage, reward, color) in enumerate(zip(stages, stage_rewards, stage_colors)):
    ax5.scatter(i, reward, s=200, color=color, edgecolor='black', linewidth=2, zorder=5)
    ax5.text(i, reward - 250, f'{reward:.1f}', ha='center', va='top', 
             fontsize=10, fontweight='bold')

ax5.set_ylabel('Avg Reward per Intersection', fontsize=11, fontweight='bold')
ax5.set_title('Training Progression\n(Performance Evolution)', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_xticks(range(len(stages)))
ax5.set_xticklabels(stages, fontsize=9)

# Add overall title
fig.suptitle('Complete System Performance Comparison\nSingle-Agent vs Multi-Agent (Transfer vs Fine-Tuned)', 
             fontsize=18, fontweight='bold', y=0.98)

# Add footer with key metrics
footer_text = (
    'Key Results: ✓ Transfer Learning: 68% improvement over single-agent  '
    '✓ Fine-Tuning: Additional 58.9% improvement  '
    '✓ Total: 86.8% better than single-agent  '
    '✓ Waiting time reduced to 0.0s'
)
fig.text(0.5, 0.02, footer_text, ha='center', fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontweight='bold')

plt.savefig('results/multiagent/final_comparison_visualization.png', dpi=300, bbox_inches='tight')
print('✅ Final comparison visualization saved to: results/multiagent/final_comparison_visualization.png')

plt.show()
