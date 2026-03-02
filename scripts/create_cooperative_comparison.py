"""
Compare Independent vs Cooperative Multi-Agent Performance
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('results/cooperative', exist_ok=True)

# Data from evaluations (50 episodes each)
systems = ['Independent\nFine-Tuned', 'Cooperative\nFrom Scratch']

# Per-intersection results
independent_rewards = {'tls_1': -807.5, 'tls_2': -663.5, 'tls_3': -448.0, 'tls_4': -324.0}
cooperative_rewards = {'tls_1': -585.8, 'tls_2': -585.8, 'tls_3': -585.8, 'tls_4': -585.8}

# Average metrics
independent_avg = -560.8
cooperative_avg = -585.8

# Waiting times
independent_wait = 0.00
cooperative_wait = 0.00

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Color scheme
colors_systems = ['#45B7D1', '#9B59B6']
colors_intersections = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D']

# ===== Plot 1: Average Performance Comparison =====
ax1 = fig.add_subplot(gs[0, 0])
avg_rewards = [independent_avg, cooperative_avg]
bars1 = ax1.bar(systems, avg_rewards, color=colors_systems, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Average Reward per Intersection', fontsize=12, fontweight='bold')
ax1.set_title('Overall Performance\n(Higher is Better)', fontsize=14, fontweight='bold')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.grid(axis='y', alpha=0.3)

for bar, val in zip(bars1, avg_rewards):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add performance note
diff = cooperative_avg - independent_avg
diff_pct = (diff / independent_avg) * 100
if diff > 0:
    label = f'{abs(diff_pct):.1f}% worse'
    color = 'red'
else:
    label = f'{abs(diff_pct):.1f}% better'
    color = 'green'

ax1.text(0.5, min(avg_rewards) - 80, label, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat' if diff > 0 else 'lightgreen', alpha=0.7),
         fontweight='bold', color=color)

# ===== Plot 2: Per-Intersection Breakdown =====
ax2 = fig.add_subplot(gs[0, 1:])
intersections = ['TLS 1\n(Top-Left)', 'TLS 2\n(Top-Right)', 'TLS 3\n(Bottom-Left)', 'TLS 4\n(Bottom-Right)']
independent_vals = [-807.5, -663.5, -448.0, -324.0]
cooperative_vals = [-585.8, -585.8, -585.8, -585.8]

x = np.arange(len(intersections))
width = 0.35

bars2a = ax2.bar(x - width/2, independent_vals, width, label='Independent (Fine-Tuned)', 
                 color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2b = ax2.bar(x + width/2, cooperative_vals, width, label='Cooperative (From Scratch)',
                 color='#9B59B6', alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_ylabel('Reward', fontsize=12, fontweight='bold')
ax2.set_title('Per-Intersection Performance Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(intersections, fontsize=10)
ax2.legend(fontsize=11, loc='lower right')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axhline(y=-560.8, color='#45B7D1', linestyle=':', alpha=0.5, label='Independent Avg')
ax2.axhline(y=-585.8, color='#9B59B6', linestyle=':', alpha=0.5, label='Cooperative Avg')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars2a, bars2b]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height - 30,
                f'{height:.1f}', ha='center', va='top', fontsize=9, fontweight='bold')

# ===== Plot 3: Performance Variance =====
ax3 = fig.add_subplot(gs[1, 0])
independent_std = np.std(independent_vals)
cooperative_std = np.std(cooperative_vals)
variance_data = [independent_std, cooperative_std]

bars3 = ax3.bar(systems, variance_data, color=colors_systems, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
ax3.set_title('Performance Consistency\n(Lower is Better)', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for bar, val in zip(bars3, variance_data):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add interpretation
if cooperative_std < independent_std:
    improvement_pct = ((independent_std - cooperative_std) / independent_std) * 100
    ax3.text(0.5, max(variance_data) * 0.5, f'Cooperative is\n{improvement_pct:.1f}% more balanced',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            fontweight='bold')

# ===== Plot 4: Training Approach Comparison =====
ax4 = fig.add_subplot(gs[1, 1])
approaches = ['Independent', 'Cooperative']
training_details = {
    'Independent': {
        'Method': 'Transfer + Fine-tune',
        'Episodes': 100,
        'Initial': 'Episode 900 (pretrained)',
        'State': '6 features',
        'Result': -560.8
    },
    'Cooperative': {
        'Method': 'From Scratch',
        'Episodes': 700,
        'Initial': 'Random weights',
        'State': '8 features (+ neighbors)',
        'Result': -585.8
    }
}

ax4.axis('off')
table_data = [
    ['Attribute', 'Independent', 'Cooperative'],
    ['Training Method', 'Transfer + Fine-tune', 'From Scratch'],
    ['Episodes Trained', '100', '700'],
    ['Initial Weights', 'Episode 900', 'Random'],
    ['State Dimension', '6 features', '8 features'],
    ['Neighbor Info', '❌ No', '✅ Yes'],
    ['Avg Reward', '-560.8', '-585.8'],
    ['Balance (Std Dev)', '197.0', '0.0'],
]

table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                  colWidths=[0.3, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')

ax4.set_title('Training Details Comparison', fontsize=14, fontweight='bold', pad=20)

# ===== Plot 5: Key Insights =====
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')

insights_text = """
KEY FINDINGS:

✅ COOPERATIVE ADVANTAGES:
  • Perfect load balancing
    All intersections: -585.8
  • Zero variance (std = 0.0)
  • Sees neighbor traffic patterns
  • Network-level optimization

⚠️ COOPERATIVE CHALLENGES:
  • 4.5% worse avg performance
    (-585.8 vs -560.8)
  • Required 700 episodes training
    (vs 100 for fine-tuning)
  • ~5 hours training time
    (vs 40 min fine-tuning)

💡 INDEPENDENT ADVANTAGES:
  • Better avg performance (-560.8)
  • Fast fine-tuning (100 episodes)
  • Transfer learning from Episode 900
  • Lower training cost

⚠️ INDEPENDENT ISSUES:
  • Unbalanced load:
    TLS 4: -324 (best)
    TLS 1: -807 (worst)
  • High variance (197.0)
  • No explicit coordination
"""

ax5.text(0.1, 0.95, insights_text, transform=ax5.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Overall title
fig.suptitle('Multi-Agent Traffic Control: Independent vs Cooperative Comparison\n(50 Episodes Each)', 
             fontsize=18, fontweight='bold', y=0.98)

# Footer
footer_text = (
    '📊 Summary: Independent has 4.5% better avg performance but unbalanced load. '
    'Cooperative has perfectly balanced load but required 7× more training. '
    'Both achieve 0.0s waiting time.'
)
fig.text(0.5, 0.02, footer_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
         fontweight='bold', wrap=True)

plt.savefig('results/cooperative/cooperative_vs_independent_comparison.png', dpi=300, bbox_inches='tight')
print('✅ Comparison visualization saved to: results/cooperative/cooperative_vs_independent_comparison.png')

plt.show()
