"""
Supervisor System — Comprehensive Visualization & Validation
============================================================
Generates plots comparing:
  - Local Supervisor vs 8-Intersection Baseline vs 4-Intersection Cooperative
  - Training convergence curves (supervisor losses + agent rewards)
  - Per-intersection performance breakdown
  - Episode-by-episode evaluation consistency

Output folder: outputs/analysis/supervisor/
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

OUTPUT_DIR = 'outputs/analysis/supervisor'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BASELINE  = '#2ECC71'   # green  — 8-intersection baseline
C_SUP       = '#E74C3C'   # red    — supervisor system
C_4INT      = '#3498DB'   # blue   — 4-intersection cooperative
C_GROUP_A   = '#9B59B6'   # purple — Group A
C_GROUP_B   = '#F39C12'   # orange — Group B
C_SUP_A     = '#E74C3C'   # red
C_SUP_B     = '#C0392B'   # dark red

# Known baselines
BASELINE_4INT  = -585.8       # per intersection (from 4-agent cooperative run)
BASELINE_8INT  = -197.0       # per intersection (from 800-ep retrain)
SUPERVISOR_AVG = -187.9       # per intersection (our local result)

# We'll compute GLOBAL_SUPERVISOR_AVG dynamically if the file exists
GLOBAL_SUPERVISOR_AVG = None


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    data = {}

    # Supervisor training history
    path = 'outputs/results/supervisor/training_history.csv'
    if os.path.exists(path):
        data['sup_train'] = pd.read_csv(path)
        print(f"✓ Loaded supervisor training history ({len(data['sup_train'])} episodes)")
    else:
        print(f"✗ Missing: {path}")
        data['sup_train'] = None

    # Supervisor eval results
    path = 'outputs/results/supervisor/eval_results.csv'
    if os.path.exists(path):
        data['sup_eval'] = pd.read_csv(path)
        print(f"✓ Loaded local supervisor eval results ({len(data['sup_eval'])} episodes)")
    else:
        print(f"✗ Missing: {path}")
        data['sup_eval'] = None

    # Global Supervisor training history
    path = 'outputs/results/global_supervisor/training_history.csv'
    if os.path.exists(path):
        data['global_train'] = pd.read_csv(path)
        print(f"✓ Loaded GLOBAL supervisor training history ({len(data['global_train'])} episodes)")
    else:
        print(f"✗ Missing: {path}")
        data['global_train'] = None

    # Global Supervisor eval results
    path = 'outputs/results/global_supervisor/eval_results.csv'
    if os.path.exists(path):
        data['global_eval'] = pd.read_csv(path)
        print(f"✓ Loaded GLOBAL supervisor eval results ({len(data['global_eval'])} episodes)")
        global GLOBAL_SUPERVISOR_AVG
        GLOBAL_SUPERVISOR_AVG = data['global_eval']['network_reward'].mean() / 8
    else:
        print(f"✗ Missing: {path}")
        data['global_eval'] = None

    # 8-intersection baseline training history
    path = 'outputs/results/grid8/training_history.csv'
    if os.path.exists(path):
        data['base_train'] = pd.read_csv(path)
        print(f"✓ Loaded 8-intersection training history ({len(data['base_train'])} episodes)")
    else:
        print(f"✗ Missing: {path}")
        data['base_train'] = None

    # 8-intersection eval results
    for fname in ['finetuned_eval_results.csv', 'pretrain_eval_results.csv']:
        path = f'outputs/results/grid8/{fname}'
        if os.path.exists(path):
            data['base_eval'] = pd.read_csv(path)
            print(f"✓ Loaded baseline eval results ← {fname}")
            break

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Validation checks
# ─────────────────────────────────────────────────────────────────────────────
def validate_data(data):
    print("\n" + "="*60)
    print("  DATA VALIDATION")
    print("="*60)
    errors = []

    if data.get('sup_train') is not None:
        df = data['sup_train']
        print(f"\n[Supervisor Training]")
        print(f"  Episodes        : {len(df)}")
        print(f"  Columns         : {list(df.columns)}")
        print(f"  Network reward  : min={df['Network_Reward'].min():.1f}, "
              f"max={df['Network_Reward'].max():.1f}, "
              f"final={df['Network_Reward'].iloc[-1]:.1f}")
        print(f"  Sup A loss      : mean={df['Sup_A_Loss'].mean():.4f}")
        print(f"  Sup B loss      : mean={df['Sup_B_Loss'].mean():.4f}")

        # Check: rewards should improve over time
        first_half  = df['Network_Reward'].iloc[:len(df)//2].mean()
        second_half = df['Network_Reward'].iloc[len(df)//2:].mean()
        if second_half > first_half:
            print(f"  ✅ Reward improved: {first_half:.1f} → {second_half:.1f}")
        else:
            print(f"  ⚠️  Reward did NOT improve: {first_half:.1f} → {second_half:.1f}")
            errors.append("Training reward did not improve over time")

        # Check: all per-intersection rewards should vary (not identical)
        tls_ids = [c for c in df.columns if c.startswith('tls_')]
        if tls_ids:
            final_rewards = [df[t].iloc[-50:].mean() for t in tls_ids]
            if len(set([round(r, 1) for r in final_rewards])) > 1:
                print(f"  ✅ Per-intersection rewards are differentiated")
            else:
                print(f"  ⚠️  All per-intersection rewards are identical — supervisor may not be working")
                errors.append("Per-intersection rewards identical (group avg reward problem)")

    if data.get('sup_eval') is not None:
        df = data['sup_eval']
        print(f"\n[Supervisor Evaluation]")
        print(f"  Episodes        : {len(df)}")
        rewards = df['network_reward'].values
        print(f"  Avg reward/ep   : {rewards.mean():.1f} ± {rewards.std():.1f}")
        print(f"  Avg/intersection: {rewards.mean()/8:.1f}")

        # Check: results should vary (not deterministic)
        if rewards.std() > 1.0:
            print(f"  ✅ Evaluation has variation (std={rewards.std():.1f}) — randomness working")
        else:
            print(f"  ⚠️  Evaluation results are nearly identical — --random flag may not be working")
            errors.append("Evaluation results are deterministic")

        # Check: per-intersection are differentiated
        tls_ids = [c for c in df.columns if c.startswith('tls_')]
        if tls_ids:
            per_int_avgs = [df[t].mean() for t in tls_ids]
            per_int_stds = [df[t].std()  for t in tls_ids]
            print(f"\n  Per-intersection averages:")
            for t, avg, std in zip(tls_ids, per_int_avgs, per_int_stds):
                flag = "✅" if std > 1.0 else "⚠️ "
                print(f"    {t}: {avg:.1f} ± {std:.1f}  {flag}")

    print("\n" + "="*60)
    if errors:
        print(f"  ⚠️  {len(errors)} issue(s) found:")
        for e in errors:
            print(f"     • {e}")
    else:
        print("  ✅ All checks passed — data looks healthy!")
    print("="*60)
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Training Convergence
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_convergence(data):
    if data.get('sup_train') is None:
        print("Skipping: no supervisor training history")
        return

    df   = data['sup_train']
    eps  = np.arange(1, len(df) + 1)
    window = 20

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Supervisor Training Convergence (900 Episodes)',
                 fontsize=16, fontweight='bold', y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1a: Network reward ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(eps, df['Network_Reward'], alpha=0.25, color=C_SUP, linewidth=0.8)
    smooth = df['Network_Reward'].rolling(window, min_periods=1).mean()
    ax.plot(eps, smooth, color=C_SUP, linewidth=2.5, label='Supervisor (smoothed)')
    ax.axhline(BASELINE_8INT * 8, color=C_BASELINE, linestyle='--', linewidth=2,
               label=f'8-int baseline ({BASELINE_8INT * 8:.0f})')
    ax.set_title('Network Total Reward per Episode', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Network Reward')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 1b: Group rewards ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    smooth_a = df['Group_A_Reward'].rolling(window, min_periods=1).mean() / 4
    smooth_b = df['Group_B_Reward'].rolling(window, min_periods=1).mean() / 4
    ax.plot(eps, smooth_a, color=C_GROUP_A, linewidth=2, label='Group A avg/int')
    ax.plot(eps, smooth_b, color=C_GROUP_B, linewidth=2, label='Group B avg/int')
    ax.axhline(BASELINE_8INT, color=C_BASELINE, linestyle='--', linewidth=1.5,
               label=f'Baseline ({BASELINE_8INT})')
    ax.set_title('Group Rewards (smoothed)', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward/Intersection')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 1c: Per-intersection rewards ─────────────────────────────
    tls_ids = [c for c in df.columns if c.startswith('tls_')]
    ax = fig.add_subplot(gs[1, :2])
    colors_a = ['#9B59B6', '#8E44AD', '#7D3C98', '#6C3483']
    colors_b = ['#F39C12', '#D68910', '#B9770E', '#9A7D0A']
    for i, tls in enumerate([t for t in tls_ids if t in ['tls_1','tls_2','tls_3','tls_4']]):
        s = df[tls].rolling(window, min_periods=1).mean()
        ax.plot(eps, s, color=colors_a[i], linewidth=1.5, label=tls, alpha=0.85)
    for i, tls in enumerate([t for t in tls_ids if t in ['tls_5','tls_6','tls_7','tls_8']]):
        s = df[tls].rolling(window, min_periods=1).mean()
        ax.plot(eps, s, color=colors_b[i], linewidth=1.5, label=tls, alpha=0.85, linestyle='--')
    ax.axhline(BASELINE_8INT, color=C_BASELINE, linestyle=':', linewidth=1.5,
               label='8-int baseline')
    ax.set_title('Per-Intersection Reward (smoothed)', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # ── 1d: Supervisor losses ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    if 'Sup_A_Loss' in df.columns:
        smooth_la = df['Sup_A_Loss'].rolling(window, min_periods=1).mean()
        smooth_lb = df['Sup_B_Loss'].rolling(window, min_periods=1).mean()
        ax.plot(eps, smooth_la, color=C_GROUP_A, linewidth=2, label='Supervisor A loss')
        ax.plot(eps, smooth_lb, color=C_GROUP_B, linewidth=2, label='Supervisor B loss')
        ax.set_title('Supervisor TD Loss (smoothed)', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('MSE Loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.savefig(f'{OUTPUT_DIR}/01_training_convergence.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/01_training_convergence.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Evaluation — Per-Intersection Breakdown
# ─────────────────────────────────────────────────────────────────────────────
def plot_eval_per_intersection(data):
    if data.get('sup_eval') is None:
        print("Skipping: no supervisor eval data")
        return

    df     = data['sup_eval']
    tls_ids = [f'tls_{i}' for i in range(1, 9)]
    avgs   = [df[t].mean() for t in tls_ids if t in df.columns]
    stds   = [df[t].std()  for t in tls_ids if t in df.columns]
    tls_ids = [t for t in tls_ids if t in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Local Supervisor — Per-Intersection Performance',
                 fontsize=15, fontweight='bold')

    # ── 2a: Bar chart ──────────────────────────────────────────
    ax = axes[0]
    colors = [C_GROUP_A if i < 4 else C_GROUP_B for i in range(len(tls_ids))]
    bars = ax.bar(tls_ids, avgs, yerr=stds, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1.2, capsize=5,
                  error_kw={'linewidth': 2})
    ax.axhline(BASELINE_8INT, color=C_BASELINE, linewidth=2.5, linestyle='--',
               label=f'8-int baseline ({BASELINE_8INT})')
    ax.axhline(SUPERVISOR_AVG, color=C_SUP, linewidth=2, linestyle=':',
               label=f'Supervisor avg ({SUPERVISOR_AVG})')
    ax.set_title('Avg Reward per Intersection (±std over 20 eval episodes)',
                 fontweight='bold')
    ax.set_ylabel('Avg Reward')
    ax.set_xlabel('Intersection')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    legend_els = [
        mpatches.Patch(facecolor=C_GROUP_A, label='Group A (tls 1-4)'),
        mpatches.Patch(facecolor=C_GROUP_B, label='Group B (tls 5-8)'),
        plt.Line2D([0], [0], color=C_BASELINE, linestyle='--',
                   linewidth=2, label=f'8-int baseline ({BASELINE_8INT})'),
    ]
    ax.legend(handles=legend_els, fontsize=9)

    for bar, val, std in zip(bars, avgs, stds):
        ax.text(bar.get_x() + bar.get_width()/2, val - std - 15,
                f'{val:.0f}', ha='center', va='top',
                fontsize=8, fontweight='bold', color='white')

    # ── 2b: Episode-by-episode consistency ────────────────────
    ax = axes[1]
    episodes = df['episode'].values
    net_per_int = df['network_reward'].values / 8
    ax.plot(episodes, net_per_int, 'o-', color=C_SUP, linewidth=2,
            markersize=5, label='Supervisor avg/int')
    ax.axhline(BASELINE_8INT, color=C_BASELINE, linewidth=2, linestyle='--',
               label=f'8-int baseline ({BASELINE_8INT})')
    ax.axhline(SUPERVISOR_AVG, color=C_SUP, linewidth=1.5, linestyle=':',
               label=f'Mean ({SUPERVISOR_AVG})')
    ax.fill_between(episodes,
                    [SUPERVISOR_AVG - df['network_reward'].std()/8]*len(episodes),
                    [SUPERVISOR_AVG + df['network_reward'].std()/8]*len(episodes),
                    alpha=0.15, color=C_SUP)
    ax.set_title('Episode-by-Episode Avg Reward/Intersection', fontweight='bold')
    ax.set_xlabel('Evaluation Episode')
    ax.set_ylabel('Avg Reward per Intersection')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_eval_per_intersection.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/02_eval_per_intersection.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: System Comparison — All 3 Stages
# ─────────────────────────────────────────────────────────────────────────────
def plot_system_comparison(data):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Traffic Control System Comparison — All Stages',
                 fontsize=15, fontweight='bold')

    # ── 3a: Per-intersection avg across systems ─────────────────
    ax = axes[0]
    systems = ['4-Int\nCooperative\n(Baseline)', '8-Int\nNo Supervisor\n(Baseline)',
               '8-Int\nLocal Supervisor\n(Ours)']
    vals  = [BASELINE_4INT, BASELINE_8INT, SUPERVISOR_AVG]
    clrs  = [C_4INT, C_BASELINE, C_SUP]

    if GLOBAL_SUPERVISOR_AVG is not None:
        systems.append('8-Int\nGlobal Supervisor\n(Ours)')
        vals.append(GLOBAL_SUPERVISOR_AVG)
        clrs.append('#8E44AD') # Purple for global

    bars  = ax.bar(systems, vals, color=clrs, alpha=0.85, edgecolor='black',
                   linewidth=1.5, width=0.5)
    ax.set_title('Avg Reward per Intersection — System Comparison',
                 fontweight='bold')
    ax.set_ylabel('Avg Reward per Intersection')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val - 15,
                f'{val:.1f}', ha='center', va='top',
                fontsize=11, fontweight='bold', color='white')

    # Improvement arrows
    ax.annotate('', xy=(1, BASELINE_8INT), xytext=(0, BASELINE_4INT),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(2, SUPERVISOR_AVG), xytext=(1, BASELINE_8INT),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    diff_sup = SUPERVISOR_AVG - BASELINE_8INT
    pct_sup  = diff_sup / abs(BASELINE_8INT) * 100
    ax.text(1.5, (SUPERVISOR_AVG + BASELINE_8INT) / 2,
            f'{diff_sup:+.1f}\n({pct_sup:+.1f}%)',
            ha='center', fontsize=10, color='green', fontweight='bold')

    # ── 3b: Group-level comparison ──────────────────────────────
    ax = axes[1]
    if data.get('sup_eval') is not None:
        df = data['sup_eval']
        sup_a = df['group_a_reward'].mean() / 4
        sup_b = df['group_b_reward'].mean() / 4

        categories = ['Group A\n8-int baseline', 'Group A\nSupervisor',
                      'Group B\n8-int baseline', 'Group B\nSupervisor']
        # Using known baseline group results from earlier eval
        base_a = -200.9   # from 8-intersection eval run 1
        base_b = -189.7
        group_vals  = [base_a, sup_a, base_b, sup_b]
        group_clrs  = [C_BASELINE, C_GROUP_A, C_BASELINE, C_GROUP_B]
        group_alpha = [0.6, 0.9, 0.6, 0.9]

        bars2 = ax.bar(categories, group_vals, color=group_clrs,
                       alpha=0.85, edgecolor='black', linewidth=1.2, width=0.5)
        ax.set_title('Group-Level Comparison: Baseline vs Supervisor', fontweight='bold')
        ax.set_ylabel('Avg Reward per Intersection')
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars2, group_vals):
            ax.text(bar.get_x() + bar.get_width()/2, val - 8,
                    f'{val:.1f}', ha='center', va='top',
                    fontsize=10, fontweight='bold', color='white')

        legend_els = [
            mpatches.Patch(facecolor=C_BASELINE, alpha=0.6, label='Baseline'),
            mpatches.Patch(facecolor=C_GROUP_A,  alpha=0.9, label='Supervisor Group A'),
            mpatches.Patch(facecolor=C_GROUP_B,  alpha=0.9, label='Supervisor Group B'),
        ]
        ax.legend(handles=legend_els, fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_system_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/03_system_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Supervisor Signal Analysis (from training history)
# ─────────────────────────────────────────────────────────────────────────────
def plot_reward_distribution(data):
    if data.get('sup_eval') is None:
        return

    df      = data['sup_eval']
    tls_ids = [f'tls_{i}' for i in range(1, 9) if f'tls_{i}' in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Reward Distribution per Intersection — 20 Eval Episodes',
                 fontsize=14, fontweight='bold')

    for i, tls in enumerate(tls_ids):
        ax    = axes[i // 4][i % 4]
        vals  = df[tls].values
        color = C_GROUP_A if i < 4 else C_GROUP_B
        grp   = 'A' if i < 4 else 'B'

        ax.hist(vals, bins=10, color=color, alpha=0.75, edgecolor='black', linewidth=0.8)
        ax.axvline(vals.mean(), color='black', linewidth=2, linestyle='--',
                   label=f'mean={vals.mean():.0f}')
        ax.axvline(BASELINE_8INT, color=C_BASELINE, linewidth=1.5, linestyle=':',
                   label=f'baseline={BASELINE_8INT}')
        ax.set_title(f'{tls} (Group {grp})\n{vals.mean():.1f} ± {vals.std():.1f}',
                     fontweight='bold', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Episode Reward')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_reward_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/04_reward_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Supervisor System — Visualization & Validation")
    print("=" * 60)

    data   = load_data()
    errors = validate_data(data)

    print(f"\nGenerating visualizations → {OUTPUT_DIR}/")

    plot_training_convergence(data)
    plot_eval_per_intersection(data)
    plot_system_comparison(data)
    plot_reward_distribution(data)

    print(f"\n{'='*60}")
    print(f"  All plots saved to: {OUTPUT_DIR}/")
    print(f"  Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(f'{OUTPUT_DIR}/{f}') // 1024
        print(f"    {f}  ({size} KB)")
    print(f"  Summary:")
    print(f"    4-Int Cooperative baseline : {BASELINE_4INT:.1f}/intersection")
    print(f"    8-Int No-Supervisor        : {BASELINE_8INT:.1f}/intersection")
    print(f"    8-Int Local Supervisor     : {SUPERVISOR_AVG:.1f}/intersection")
    diff = SUPERVISOR_AVG - BASELINE_8INT
    print(f"    Local Improvement          : {diff:+.1f} ({diff/abs(BASELINE_8INT)*100:+.1f}%)")

    if GLOBAL_SUPERVISOR_AVG is not None:
        print(f"    8-Int GLOBAL Supervisor    : {GLOBAL_SUPERVISOR_AVG:.1f}/intersection")
        g_diff = GLOBAL_SUPERVISOR_AVG - BASELINE_8INT
        print(f"    Global Improvement         : {g_diff:+.1f} ({g_diff/abs(BASELINE_8INT)*100:+.1f}%)")
        v_local = GLOBAL_SUPERVISOR_AVG - SUPERVISOR_AVG
        print(f"    Global vs Local            : {v_local:+.1f} ({v_local/abs(SUPERVISOR_AVG)*100:+.1f}%)")
    if errors:
        print(f"\n  ⚠️  {len(errors)} validation issue(s) to review:")
        for e in errors:
            print(f"     • {e}")
    else:
        print(f"\n  ✅ All validation checks passed!")
    print("="*60)


if __name__ == '__main__':
    main()
