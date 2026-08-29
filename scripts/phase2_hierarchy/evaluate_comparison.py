"""
Final Evaluation & Presentation Comparison
  System A : 4×1  — 4-intersection cooperative (700 episodes)
  System B : 4×2 pre-FT — 8-intersection transferred weights (no fine-tuning)
  System C : 4×2 fine-tuned — 8-intersection after 300 fine-tuning episodes
"""

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------


import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from traffic_rl.envs.grid8 import EightIntersectionEnv
from traffic_rl.core.agent import DDQNAgent

OUT_DIR  = 'outputs/results/grid8/final_comparison'
W_MAP    = {
    'tls_1':'tls_1','tls_2':'tls_2','tls_3':'tls_3','tls_4':'tls_4',
    'tls_5':'tls_1','tls_6':'tls_2','tls_7':'tls_3','tls_8':'tls_4',
}
C4 = '#2ECC71';  CA = '#45B7D1';  CB = '#9B59B6';  CFT = '#E74C3C';  CNG = '#E67E22'

# ── 4-int final performance (from 700-ep training history) ──────────
def get_4int_stats():
    df = pd.read_csv('outputs/results/cooperative/training_history.csv')
    df.columns = ['network','tls1','tls2','tls3','tls4']
    last = df.tail(50)
    return dict(
        per_int_avg = last['tls1'].mean(),
        per_int_std = last['tls1'].std(),
        network_avg = last['network'].mean(),
        label       = '4×1 Cooperative\n(4 intersections)',
        history_df  = df,
    )

# ── Pre-FT stats (already measured) ─────────────────────────────────
def get_preft_stats():
    df = pd.read_csv('outputs/results/grid8/pretrain_eval_results.csv')
    tls_cols = [c for c in df.columns if c.startswith('tls_')]
    per_int  = df[tls_cols].values.flatten()
    ga = df[['tls_1','tls_2','tls_3','tls_4']].mean().mean()
    gb = df[['tls_5','tls_6','tls_7','tls_8']].mean().mean()
    return dict(
        per_int_avg = per_int.mean(),
        per_int_std = per_int.std(),
        network_avg = df['network_reward'].mean(),
        group_a_avg = ga,
        group_b_avg = gb,
        label       = '4×2 Pre-Fine-Tune\n(8 intersections, transferred)',
        df          = df,
    )

# ── Run evaluation on fine-tuned model ──────────────────────────────
def evaluate_finetuned(n_episodes=10):
    print(f"\n{'='*65}")
    print(f"  Evaluating FINE-TUNED 4×2 model  ({n_episodes} episodes)")
    print(f"{'='*65}")

    env = EightIntersectionEnv(use_gui=False, num_seconds=3600, delta_time=5)
    agents = {}
    for tls in env.tls_ids:
        agents[tls] = DDQNAgent(
            state_dim=env.get_state_dim(), action_dim=env.get_action_dim(),
            hidden_dim=128, learning_rate=0.0001,
            epsilon_start=0.0, epsilon_decay=0.995, epsilon_min=0.01
        )
        ck = f'outputs/checkpoints/grid8/{tls}_final.pth'
        agents[tls].load(ck)
        agents[tls].epsilon = 0.0
        print(f"  ✓ {tls} ← {ck}")

    rows = []
    for ep in tqdm(range(n_episodes), desc="Fine-tuned eval"):
        states = env.reset()
        ep_r   = {t: 0.0 for t in env.tls_ids}
        done   = False
        info   = {'avg_waiting_time': 0.0}
        while not done:
            actions = {t: agents[t].select_action(states[t], training=False)
                       for t in env.tls_ids}
            states, rewards, done, info = env.step(actions)
            for t in env.tls_ids:
                ep_r[t] += rewards[t]
        row = {'episode': ep+1,
               'network_reward': sum(ep_r.values()),
               'group_a_reward': sum(ep_r[t] for t in env.group_a),
               'group_b_reward': sum(ep_r[t] for t in env.group_b),
               'avg_wait_time':  info['avg_waiting_time']}
        for t in env.tls_ids:
            row[t] = ep_r[t]
        rows.append(row)
    env.close()

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f'{OUT_DIR}/finetuned_eval_results.csv', index=False)
    print(f"  ✅ Saved → {OUT_DIR}/finetuned_eval_results.csv")

    tls_cols = [c for c in df.columns if c.startswith('tls_')]
    per_int  = df[tls_cols].values.flatten()
    ga = df[['tls_1','tls_2','tls_3','tls_4']].mean().mean()
    gb = df[['tls_5','tls_6','tls_7','tls_8']].mean().mean()
    return dict(
        per_int_avg = per_int.mean(),
        per_int_std = per_int.std(),
        network_avg = df['network_reward'].mean(),
        group_a_avg = ga,
        group_b_avg = gb,
        label       = '4×2 Fine-Tuned\n(8 intersections, 300 ep)',
        df          = df,
    )

# ── Print console summary ────────────────────────────────────────────
def print_summary(s4, preft, ft):
    print(f"\n{'='*65}")
    print("  FINAL COMPARISON SUMMARY")
    print(f"{'='*65}")
    print(f"{'System':<35} {'Avg/Int':>10} {'Net Total':>12} {'vs 4×1':>10}")
    print(f"{'─'*65}")
    for s, col in [(s4,'4×1'), (preft,'Pre-FT'), (ft,'Fine-Tuned')]:
        diff = s['per_int_avg'] - s4['per_int_avg']
        pct  = diff / abs(s4['per_int_avg']) * 100
        tag  = f"{diff:+.0f} ({pct:+.1f}%)" if col != '4×1' else '—'
        n    = int(8 if col != '4×1' else 4)
        print(f"  {col+' ('+str(n)+' intersections)':<33} {s['per_int_avg']:>10.1f} "
              f"{s['network_avg']:>12.1f} {tag:>10}")
    print(f"{'='*65}")

# ── FIGURE 1: Three-way bar comparison ──────────────────────────────
def fig_bar_comparison(s4, preft, ft):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle('Traffic Control System Comparison\n'
                 '4×1 Cooperative  |  4×2 Pre-Fine-Tune  |  4×2 Fine-Tuned',
                 fontsize=15, fontweight='bold')
    fig.patch.set_facecolor('#F8F9FA')
    for ax in axes:
        ax.set_facecolor('#F8F9FA')
        ax.grid(axis='y', alpha=0.35, color='#DEE2E6')

    systems = ['4×1\nCooperative\n(4 agents)', '4×2\nPre-Fine-Tune\n(8 agents)', '4×2\nFine-Tuned\n(8 agents)']
    colors  = [C4, CNG, CFT]
    avgs    = [s4['per_int_avg'], preft['per_int_avg'], ft['per_int_avg']]
    stds    = [s4['per_int_std'], preft['per_int_std'], ft['per_int_std']]
    nets    = [s4['network_avg'], preft['network_avg'], ft['network_avg']]

    # ── Plot 1: Avg reward per intersection ──
    ax = axes[0]
    bars = ax.bar(systems, avgs, yerr=stds, color=colors, alpha=0.88,
                  edgecolor='black', linewidth=1.2, capsize=6,
                  error_kw={'linewidth': 2})
    ax.set_title('Avg Reward per Intersection', fontweight='bold', fontsize=12)
    ax.set_ylabel('Avg Reward')
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x()+bar.get_width()/2, v-30, f'{v:.1f}',
                ha='center', va='top', fontsize=11, fontweight='bold', color='white')
    # improvement arrows
    for i, s in enumerate([preft, ft]):
        diff = s['per_int_avg'] - s4['per_int_avg']
        pct  = diff / abs(s4['per_int_avg']) * 100
        col  = '#27AE60' if diff > 0 else '#E74C3C'
        sym  = '▲' if diff > 0 else '▼'
        ax.text(i+1, avgs[i+1]+stds[i+1]+20, f'{sym}{abs(pct):.1f}%',
                ha='center', fontsize=10, color=col, fontweight='bold')

    # ── Plot 2: Group A vs Group B per intersection ──
    ax = axes[1]
    x = np.arange(2)
    w = 0.28
    ga_vals = [s4['per_int_avg'], preft.get('group_a_avg', preft['per_int_avg']),
               ft.get('group_a_avg', ft['per_int_avg'])]
    gb_vals = [s4['per_int_avg'], preft.get('group_b_avg', preft['per_int_avg']),
               ft.get('group_b_avg', ft['per_int_avg'])]

    grp_labels = ['Pre-Fine-Tune', 'Fine-Tuned']
    ax.bar(x - w/2, [ga_vals[1], ga_vals[2]], w, color=CA, alpha=0.88,
           edgecolor='black', linewidth=1.1, label='Group A (I1–I4)')
    ax.bar(x + w/2, [gb_vals[1], gb_vals[2]], w, color=CB, alpha=0.88,
           edgecolor='black', linewidth=1.1, label='Group B (I5–I8)')
    ax.axhline(s4['per_int_avg'], color=C4, linestyle='--', linewidth=2,
               label=f'4×1 baseline ({s4["per_int_avg"]:.0f})')
    ax.set_xticks(x); ax.set_xticklabels(grp_labels)
    ax.set_title('Group A vs Group B\nper Intersection', fontweight='bold', fontsize=12)
    ax.set_ylabel('Avg Reward per Intersection')
    ax.legend(fontsize=9)
    for xi, va, vb in zip(x, [ga_vals[1], ga_vals[2]], [gb_vals[1], gb_vals[2]]):
        ax.text(xi-w/2, va-25, f'{va:.0f}', ha='center', va='top',
                fontsize=9, fontweight='bold', color='white')
        ax.text(xi+w/2, vb-25, f'{vb:.0f}', ha='center', va='top',
                fontsize=9, fontweight='bold', color='white')

    # ── Plot 3: Network total ──
    ax = axes[2]
    bars3 = ax.bar(systems, nets, color=colors, alpha=0.88,
                   edgecolor='black', linewidth=1.2)
    ax.set_title('Total Network Reward\n(sum over all intersections)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Network Total Reward')
    for bar, v in zip(bars3, nets):
        ax.text(bar.get_x()+bar.get_width()/2, v-80, f'{v:.0f}',
                ha='center', va='top', fontsize=11, fontweight='bold', color='white')
    ax.text(0.5, 0.03,
            f'8-int fine-tuned total = {nets[2]:.0f}  vs  4-int×2 = {nets[0]*2:.0f}',
            transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.9))

    plt.tight_layout()
    out = f'{OUT_DIR}/three_way_bar_comparison.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='#F8F9FA')
    print(f"✅ {out}")
    plt.close()

# ── FIGURE 2: Training progression (fine-tuning curve) ──────────────
def fig_training_progression(s4):
    ft_hist = pd.read_csv('outputs/results/grid8/training_history.csv')
    coop_hist = s4['history_df']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Progression: 4×1 (700 ep)  vs  4×2 Fine-Tune (300 ep)',
                 fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('#F8F9FA')
    for ax in axes:
        ax.set_facecolor('#F8F9FA')
        ax.grid(True, alpha=0.3, color='#DEE2E6')

    # — per-intersection reward over training —
    ax = axes[0]
    ep4 = range(1, len(coop_hist)+1)
    smooth4 = coop_hist['tls1'].rolling(20, min_periods=1).mean()
    ax.plot(ep4, coop_hist['tls1'], color=C4, alpha=0.15, linewidth=1)
    ax.plot(ep4, smooth4, color=C4, linewidth=2.5, label='4×1 per-intersection (MA20)')

    ep8 = range(1, len(ft_hist)+1)
    for tls, col, label in [
        ('tls_1', CA, 'Group A avg (MA20)'),
        ('tls_5', CB, 'Group B avg (MA20)'),
    ]:
        if tls in ft_hist.columns:
            sm = ft_hist[tls].rolling(20, min_periods=1).mean()
            ax.plot(ep8, ft_hist[tls], color=col, alpha=0.12, linewidth=1)
            ax.plot(ep8, sm, color=col, linewidth=2, label=f'4×2 FT {label}')

    ax.set_title('Per-Intersection Reward During Training', fontweight='bold')
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Reward per Intersection')
    ax.legend(fontsize=9)

    # — network reward comparison —
    ax = axes[1]
    net_smooth4 = (coop_hist['network'] / 4).rolling(20, min_periods=1).mean()
    ax.plot(ep4, coop_hist['network']/4, color=C4, alpha=0.15, linewidth=1)
    ax.plot(ep4, net_smooth4, color=C4, linewidth=2.5, label='4×1 network/4 (MA20)')

    if 'Network_Reward' in ft_hist.columns:
        net8_per = ft_hist['Network_Reward'] / 8
        sm8 = net8_per.rolling(20, min_periods=1).mean()
        ax.plot(ep8, net8_per, color=CFT, alpha=0.15, linewidth=1)
        ax.plot(ep8, sm8, color=CFT, linewidth=2.5, label='4×2 FT network/8 (MA20)')

    ax.set_title('Network Reward ÷ Intersections\n(normalised for fair comparison)', fontweight='bold')
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Avg Reward per Intersection')
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = f'{OUT_DIR}/training_progression.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='#F8F9FA')
    print(f"✅ {out}")
    plt.close()

# ── FIGURE 3: Presentation poster (dark theme) ──────────────────────
def fig_presentation_poster(s4, preft, ft):
    fig = plt.figure(figsize=(20, 11))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.35)
    fig.patch.set_facecolor('#0d1117')

    def sax(ax, title=''):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        ax.grid(True, color='#21262d', linewidth=0.8)
        if title: ax.set_title(title, fontsize=10.5, fontweight='bold', color='#f0f6fc')

    systems = ['4×1\nCooper.', '4×2\nPre-FT', '4×2\nFine-Tuned']
    avgs    = [s4['per_int_avg'], preft['per_int_avg'], ft['per_int_avg']]
    nets    = [s4['network_avg'], preft['network_avg'], ft['network_avg']]
    colors  = [C4, CNG, CFT]

    # ── [0,0]: per-intersection bars ──
    ax = fig.add_subplot(gs[0, 0])
    sax(ax, 'Avg Reward / Intersection')
    bars = ax.bar(systems, avgs, color=colors, alpha=0.9, edgecolor='#30363d', linewidth=1)
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x()+bar.get_width()/2, v-20, f'{v:.0f}',
                ha='center', va='top', fontsize=9, fontweight='bold', color='white')

    # ── [0,1]: network total ──
    ax = fig.add_subplot(gs[0, 1])
    sax(ax, 'Total Network Reward')
    ax.bar(systems, nets, color=colors, alpha=0.9, edgecolor='#30363d', linewidth=1)
    for i, v in enumerate(nets):
        ax.text(i, v - 60, f'{v:.0f}', ha='center', va='top',
                fontsize=9, fontweight='bold', color='white')

    # ── [0,2]: group A vs B ──
    ax = fig.add_subplot(gs[0, 2])
    sax(ax, 'Group A vs Group B (8-int)')
    labels = ['Pre-FT', 'Fine-Tuned']
    x = np.arange(2); w = 0.3
    ga = [preft.get('group_a_avg', preft['per_int_avg']),
          ft.get('group_a_avg', ft['per_int_avg'])]
    gb = [preft.get('group_b_avg', preft['per_int_avg']),
          ft.get('group_b_avg', ft['per_int_avg'])]
    ax.bar(x-w/2, ga, w, color=CA, alpha=0.9, label='Group A', edgecolor='#30363d')
    ax.bar(x+w/2, gb, w, color=CB, alpha=0.9, label='Group B', edgecolor='#30363d')
    ax.axhline(s4['per_int_avg'], color=C4, linestyle='--', linewidth=1.8,
               label=f'4×1 ({s4["per_int_avg"]:.0f})')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#c9d1d9', edgecolor='#30363d')

    # ── [0,3]: improvement % card ──
    ax = fig.add_subplot(gs[0, 3])
    ax.set_facecolor('#161b22')
    ax.axis('off')
    metrics = [
        ('Pre-FT vs 4×1',       preft['per_int_avg'], CNG),
        ('Fine-Tuned vs 4×1',   ft['per_int_avg'],    CFT),
        ('Fine-Tuned vs Pre-FT', ft['per_int_avg'],   CA),
    ]
    baselines = [s4['per_int_avg'], s4['per_int_avg'], preft['per_int_avg']]
    y = 0.92
    ax.text(0.5, 1.0, 'Improvement Summary', ha='center', va='top',
            transform=ax.transAxes, color='#f0f6fc', fontsize=11, fontweight='bold')
    for (label, val, col), base in zip(metrics, baselines):
        diff = val - base
        pct  = diff / abs(base) * 100
        sym  = '▲' if diff > 0 else '▼'
        ax.text(0.05, y, label, transform=ax.transAxes, color='#8b949e', fontsize=9, va='top')
        ax.text(0.05, y-0.09, f'  {sym} {abs(pct):.1f}%  ({diff:+.0f} reward)',
                transform=ax.transAxes, color=col, fontsize=11, fontweight='bold', va='top')
        y -= 0.25

    # ── [1,0:2]: fine-tuning curve ──
    ax = fig.add_subplot(gs[1, :2])
    sax(ax, 'Fine-Tuning Progress (4×2, 300 episodes)')
    ft_hist = pd.read_csv('outputs/results/grid8/training_history.csv')
    ep8 = range(1, len(ft_hist)+1)
    if 'Network_Reward' in ft_hist.columns:
        per8 = ft_hist['Network_Reward'] / 8
        ax.plot(ep8, per8, color=CFT, alpha=0.2, linewidth=1)
        ax.plot(ep8, per8.rolling(20, min_periods=1).mean(),
                color=CFT, linewidth=2.5, label='Fine-tuned avg/int (MA20)')
    ax.axhline(s4['per_int_avg'], color=C4, linestyle='--', linewidth=1.8,
               label=f'4×1 baseline ({s4["per_int_avg"]:.0f})')
    ax.axhline(preft['per_int_avg'], color=CNG, linestyle=':', linewidth=1.8,
               label=f'Pre-FT ({preft["per_int_avg"]:.0f})')
    ax.set_xlabel('Fine-Tuning Episode', color='#c9d1d9')
    ax.set_ylabel('Avg Reward per Intersection', color='#c9d1d9')
    ax.legend(fontsize=9, facecolor='#161b22', labelcolor='#c9d1d9', edgecolor='#30363d')

    # ── [1,2:4]: per-intersection breakdown ──
    ax = fig.add_subplot(gs[1, 2:])
    sax(ax, 'Per-Intersection Breakdown — Fine-Tuned Model')
    ft_eval = pd.read_csv(f'{OUT_DIR}/finetuned_eval_results.csv')
    tls_ids = [f'tls_{i}' for i in range(1, 9)]
    per_vals = [ft_eval[t].mean() for t in tls_ids]
    bar_cols = [CA]*4 + [CB]*4
    bars = ax.bar(tls_ids, per_vals, color=bar_cols, alpha=0.9, edgecolor='#30363d', linewidth=1)
    ax.axhline(s4['per_int_avg'], color=C4, linestyle='--', linewidth=1.8,
               label=f'4×1 baseline ({s4["per_int_avg"]:.0f})')
    for bar, v in zip(bars, per_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v-20, f'{v:.0f}',
                ha='center', va='top', fontsize=8, fontweight='bold', color='white')
    legend_els = [mpatches.Patch(facecolor=CA, label='Group A (I1-I4)'),
                  mpatches.Patch(facecolor=CB, label='Group B (I5-I8)'),
                  plt.Line2D([0],[0], color=C4, linestyle='--', linewidth=1.8, label='4×1 baseline')]
    ax.legend(handles=legend_els, fontsize=8.5,
              facecolor='#161b22', labelcolor='#c9d1d9', edgecolor='#30363d')
    ax.set_xlabel('Intersection', color='#c9d1d9')
    ax.set_ylabel('Avg Reward', color='#c9d1d9')

    plt.suptitle('Multi-Agent DDQN Traffic Control — Final Results\n'
                 '4-Intersection Cooperative  vs  8-Intersection Grouped Cooperative',
                 fontsize=16, fontweight='bold', color='#f0f6fc', y=1.01)
    out = f'{OUT_DIR}/presentation_poster.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    print(f"✅ {out}")
    plt.close()

# ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load 4-int baseline stats
    print("\n[1/5] Loading 4-intersection baseline stats...")
    s4 = get_4int_stats()

    # 2. Load pre-FT stats
    print("[2/5] Loading pre-fine-tuning eval stats...")
    preft = get_preft_stats()

    # 3. Evaluate fine-tuned model
    print("[3/5] Evaluating fine-tuned 8-intersection model...")
    ft = evaluate_finetuned(n_episodes=10)

    # 4. Print summary
    print_summary(s4, preft, ft)

    # 5. Generate all figures
    print("\n[4/5] Generating comparison figures...")
    fig_bar_comparison(s4, preft, ft)
    fig_training_progression(s4)

    print("[5/5] Generating presentation poster...")
    fig_presentation_poster(s4, preft, ft)

    print(f"\n{'='*65}")
    print(f"  All outputs saved to: {OUT_DIR}/")
    print(f"    three_way_bar_comparison.png")
    print(f"    training_progression.png")
    print(f"    presentation_poster.png")
    print(f"    finetuned_eval_results.csv")
    print(f"{'='*65}")

if __name__ == '__main__':
    main()
