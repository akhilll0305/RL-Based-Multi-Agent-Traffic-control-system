"""
Global Supervisor System — Visualization & Validation
===================================================
Generates plots and validation specifically for the Global Supervisor (Step 2)
Output folder: outputs/analysis/global_supervisor/
"""

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = 'outputs/analysis/global_supervisor'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BASELINE  = '#2ECC71'   # green  — 8-intersection baseline
C_SUP_LOCAL = '#E74C3C'   # red    — local supervisor
C_SUP_GLOB  = '#8E44AD'   # purple — global supervisor
C_GROUP_A   = '#9B59B6'   # purple
C_GROUP_B   = '#F39C12'   # orange

BASELINE_8INT  = -197.0
LOCAL_SUP_AVG  = -187.9

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
def load_global_data():
    data = {}
    
    path = 'outputs/results/global_supervisor/training_history.csv'
    if os.path.exists(path):
        data['train'] = pd.read_csv(path)
    else:
        print(f"✗ Missing: {path}")
        data['train'] = None

    path = 'outputs/results/global_supervisor/eval_results.csv'
    if os.path.exists(path):
        data['eval'] = pd.read_csv(path)
    else:
        print(f"✗ Missing: {path}")
        data['eval'] = None

    return data

# ─────────────────────────────────────────────────────────────────────────────
# Validation checks
# ─────────────────────────────────────────────────────────────────────────────
def validate_data(data):
    print("\n" + "="*60)
    print("  GLOBAL SUPERVISOR - DATA VALIDATION")
    print("="*60)
    errors = []

    if data.get('train') is not None:
        df = data['train']
        print(f"\n[Global Supervisor Training]")
        print(f"  Episodes        : {len(df)}")
        print(f"  Network reward  : min={df['Network_Reward'].min():.1f}, "
              f"max={df['Network_Reward'].max():.1f}, "
              f"final={df['Network_Reward'].iloc[-1]:.1f}")
        print(f"  Sup A loss      : mean={df['Sup_A_Loss'].mean():.4f}")
        
        first_half  = df['Network_Reward'].iloc[:len(df)//2].mean()
        second_half = df['Network_Reward'].iloc[len(df)//2:].mean()
        if second_half > first_half:
            print(f"  ✅ Reward improved: {first_half:.1f} → {second_half:.1f}")
        else:
            print(f"  ⚠️  Reward did NOT improve: {first_half:.1f} → {second_half:.1f}")
            errors.append("Training reward did not improve over time. The 28-dim space might be unstable.")

        tls_ids = [c for c in df.columns if c.startswith('tls_')]
        if tls_ids:
            final_rewards = [df[t].iloc[-50:].mean() for t in tls_ids]
            if len(set([round(r, 1) for r in final_rewards])) > 1:
                print(f"  ✅ Per-intersection rewards are differentiated")
            else:
                print(f"  ⚠️  All per-intersection rewards are identical")
                errors.append("Per-intersection rewards identical")

    if data.get('eval') is not None:
        df = data['eval']
        print(f"\n[Global Supervisor Evaluation]")
        print(f"  Episodes        : {len(df)}")
        rewards = df['network_reward'].values
        print(f"  Avg reward/ep   : {rewards.mean():.1f} ± {rewards.std():.1f}")
        print(f"  Avg/intersection: {rewards.mean()/8:.1f}")

        if rewards.std() > 1.0:
            print(f"  ✅ Evaluation has healthy variation (std={rewards.std():.1f})")
        else:
            print(f"  ⚠️  Evaluation results are suspiciously deterministic")
            errors.append("Evaluation results are deterministic")

    print("\n" + "="*60)
    if errors:
        print(f"  ⚠️  {len(errors)} issue(s) found:")
        for e in errors:
            print(f"     • {e}")
    else:
        print("  ✅ All checks passed — global supervisor data looks healthy!")
    print("="*60)
    return errors

# ─────────────────────────────────────────────────────────────────────────────
# Polt 1: Training Convergence
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_convergence(data):
    if data.get('train') is None: return

    df = data['train']
    eps = np.arange(1, len(df) + 1)
    window = 30

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Global Supervisor Training Convergence (900 Episodes)', fontsize=16, fontweight='bold', y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1a: Network reward
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(eps, df['Network_Reward'], alpha=0.25, color=C_SUP_GLOB, linewidth=0.8)
    smooth = df['Network_Reward'].rolling(window, min_periods=1).mean()
    ax.plot(eps, smooth, color=C_SUP_GLOB, linewidth=2.5, label='Global Sup (smoothed)')
    ax.axhline(BASELINE_8INT * 8, color=C_BASELINE, linestyle='--', linewidth=2, label=f'8-int baseline ({BASELINE_8INT * 8:.0f})')
    ax.axhline(LOCAL_SUP_AVG * 8, color=C_SUP_LOCAL, linestyle=':', linewidth=2, label=f'Local Sup avg ({LOCAL_SUP_AVG * 8:.0f})')
    ax.set_title('Network Total Reward per Episode', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 1b: Group rewards
    ax = fig.add_subplot(gs[0, 2])
    smooth_a = df['Group_A_Reward'].rolling(window, min_periods=1).mean() / 4
    smooth_b = df['Group_B_Reward'].rolling(window, min_periods=1).mean() / 4
    ax.plot(eps, smooth_a, color=C_GROUP_A, linewidth=2, label='Group A avg/int')
    ax.plot(eps, smooth_b, color=C_GROUP_B, linewidth=2, label='Group B avg/int')
    ax.axhline(BASELINE_8INT, color=C_BASELINE, linestyle='--', linewidth=1.5, label=f'Baseline ({BASELINE_8INT})')
    ax.set_title('Group Rewards (smoothed)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1c: Losses
    ax = fig.add_subplot(gs[1, 2])
    smooth_la = df['Sup_A_Loss'].rolling(window, min_periods=1).mean()
    smooth_lb = df['Sup_B_Loss'].rolling(window, min_periods=1).mean()
    ax.plot(eps, smooth_la, color=C_GROUP_A, linewidth=2, label='Supervisor A loss')
    ax.plot(eps, smooth_lb, color=C_GROUP_B, linewidth=2, label='Supervisor B loss')
    ax.set_title('Global Supervisor TD Loss (smoothed)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.savefig(f'{OUTPUT_DIR}/01_global_training_convergence.png', dpi=200, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Evaluation Breakdown
# ─────────────────────────────────────────────────────────────────────────────
def plot_eval_breakdown(data):
    if data.get('eval') is None: return

    df = data['eval']
    tls_ids = [f'tls_{i}' for i in range(1, 9)]
    avgs = [df[t].mean() for t in tls_ids]
    stds = [df[t].std() for t in tls_ids]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Global Supervisor — Per-Intersection Evaluation Performance', fontsize=15, fontweight='bold')

    ax = axes[0]
    colors = [C_GROUP_A if i < 4 else C_GROUP_B for i in range(len(tls_ids))]
    bars = ax.bar(tls_ids, avgs, yerr=stds, color=colors, alpha=0.85, edgecolor='black', capsize=5)
    ax.axhline(BASELINE_8INT, color=C_BASELINE, linewidth=2.5, linestyle='--', label=f'Baseline ({BASELINE_8INT})')
    
    global_avg = df['network_reward'].mean() / 8
    ax.axhline(global_avg, color=C_SUP_GLOB, linewidth=2, linestyle='-', label=f'Global avg ({global_avg:.1f})')
    ax.axhline(LOCAL_SUP_AVG, color=C_SUP_LOCAL, linewidth=2, linestyle=':', label=f'Local avg ({LOCAL_SUP_AVG})')
    
    ax.set_title('Avg Reward per Intersection (±std)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    for bar, val, std in zip(bars, avgs, stds):
        ax.text(bar.get_x() + bar.get_width()/2, val - std - 15, f'{val:.0f}', ha='center', va='top', fontsize=8, color='white')

    ax = axes[1]
    episodes = df['episode'].values
    net_per_int = df['network_reward'].values / 8
    ax.plot(episodes, net_per_int, 'o-', color=C_SUP_GLOB, linewidth=2, markersize=5, label='Global Supervisor avg/int')
    ax.axhline(BASELINE_8INT, color=C_BASELINE, linewidth=2, linestyle='--', label=f'Baseline ({BASELINE_8INT})')
    ax.axhline(LOCAL_SUP_AVG, color=C_SUP_LOCAL, linewidth=2, linestyle=':', label=f'Local Supervisor ({LOCAL_SUP_AVG})')
    
    ax.set_title('Episode-by-Episode Consistency', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_global_eval_per_intersection.png', dpi=200, bbox_inches='tight')
    plt.close()

def main():
    data = load_global_data()
    errors = validate_data(data)
    
    plot_training_convergence(data)
    plot_eval_breakdown(data)
    
    print(f"\n✅ All global supervisor plots saved to: {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
