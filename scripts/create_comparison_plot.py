"""
Create comparison visualization for Single-Agent vs Multi-Agent
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
single_agent_reward = -4253.5
multi_agent_rewards = {
    'Int 1': -1766.0,
    'Int 2': -1377.0,
    'Int 3': -1250.5,
    'Int 4': -1059.0
}
multi_agent_avg = -1363.1

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Single-Agent vs Multi-Agent Traffic Control Comparison\nEpisode 900 Performance', 
             fontsize=16, fontweight='bold')

# 1. Reward Comparison
ax1.bar(['Single\nIntersection'], [single_agent_reward], color='#3498db', alpha=0.7, label='Single-Agent')
ax1.bar(['Multi-Agent\n(Avg per Int)'], [multi_agent_avg], color='#2ecc71', alpha=0.7, label='Multi-Agent')
ax1.set_ylabel('Average Reward (higher is better)', fontsize=11)
ax1.set_title('Reward Per Intersection Comparison', fontweight='bold')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels
ax1.text(0, single_agent_reward - 200, f'{single_agent_reward:.0f}', ha='center', fontweight='bold')
ax1.text(1, multi_agent_avg - 200, f'{multi_agent_avg:.1f}', ha='center', fontweight='bold')

# 2. Multi-Agent Performance Distribution
intersections = list(multi_agent_rewards.keys())
rewards = list(multi_agent_rewards.values())
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
bars = ax2.bar(intersections, rewards, color=colors, alpha=0.7)
ax2.set_ylabel('Reward (higher is better)', fontsize=11)
ax2.set_title('Per-Intersection Performance (Multi-Agent)', fontweight='bold')
ax2.axhline(y=multi_agent_avg, color='red', linestyle='--', linewidth=2, label=f'Average: {multi_agent_avg:.1f}')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, reward) in enumerate(zip(bars, rewards)):
    ax2.text(i, reward - 80, f'{reward:.0f}', ha='center', fontweight='bold')

# 3. Scalability Demonstration
scenarios = ['Single-Agent\n(1 Int)', 'Multi-Agent\n(4 Int)']
total_rewards = [single_agent_reward, sum(multi_agent_rewards.values())]
intersections_count = [1, 4]

ax3_twin = ax3.twinx()
ax3.bar(scenarios, [abs(r) for r in total_rewards], color=['#3498db', '#2ecc71'], alpha=0.7)
ax3.set_ylabel('Total Network Reward (magnitude)', fontsize=11, color='black')
ax3.set_title('Network-Wide Performance & Scalability', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

ax3_twin.plot(scenarios, intersections_count, 'ro-', linewidth=3, markersize=15, label='# Intersections')
ax3_twin.set_ylabel('Number of Intersections Controlled', fontsize=11, color='red')
ax3_twin.legend(loc='upper right')
ax3_twin.tick_params(axis='y', labelcolor='red')

# Add value labels
for i, (scenario, reward, count) in enumerate(zip(scenarios, total_rewards, intersections_count)):
    ax3.text(i, abs(reward) + 200, f'{abs(reward):.0f}', ha='center', fontweight='bold')
    ax3_twin.text(i, count + 0.15, f'{count}', ha='center', fontweight='bold', color='red')

# 4. Key Metrics Table
metrics_data = [
    ['Metric', 'Single-Agent', 'Multi-Agent'],
    ['Avg Reward per Int', f'{single_agent_reward:.1f}', f'{multi_agent_avg:.1f}'],
    ['Best Performance', f'{single_agent_reward:.1f}', f'{max(multi_agent_rewards.values()):.1f}'],
    ['Worst Performance', f'{single_agent_reward:.1f}', f'{min(multi_agent_rewards.values()):.1f}'],
    ['# Intersections', '1', '4'],
    ['Scalability', '❌ Limited', '✅ Proven'],
    ['Transfer Learning', 'N/A', '✅ Excellent'],
    ['Training Required', '900 episodes', '0 episodes']
]

ax4.axis('tight')
ax4.axis('off')
table = ax4.table(cellText=metrics_data, cellLoc='left', loc='center',
                  colWidths=[0.35, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(metrics_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('#ffffff')

ax4.set_title('Comparison Summary', fontweight='bold', pad=20)

# Add improvement note
improvement = ((single_agent_reward - multi_agent_avg) / abs(single_agent_reward)) * 100
fig.text(0.5, 0.02, f'✅ Multi-Agent shows {improvement:.1f}% better reward per intersection (due to distributed traffic load)', 
         ha='center', fontsize=12, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('results/multiagent/comparison_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Comparison visualization saved to: results/multiagent/comparison_visualization.png")
plt.show()
