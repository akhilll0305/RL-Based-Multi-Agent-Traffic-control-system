"""
Main Training Script — 8-Intersection with Local Supervisors
=============================================================
Hierarchy:
  - 2 Supervisor Agents (one per group of 4)
  - 8 Local DDQN Agents (7-dim state: 6 local + 1 supervisor signal)

Training flow per step:
  1. Get 6-dim local states from SUMO
  2. Supervisors generate coordination signals (one per agent)
  3. Build 7-dim enhanced states
  4. Agents pick actions
  5. SUMO executes, returns individual rewards
  6. Store + train agents (individual reward)
  7. Store + train supervisors (group average reward)

Usage:
  python scripts/phase2_hierarchy/local_supervisor.py --mode train --episodes 500
  python scripts/phase2_hierarchy/local_supervisor.py --mode evaluate --load-final
  python scripts/phase2_hierarchy/local_supervisor.py --mode train --episodes 300 --resume-from 200
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
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from traffic_rl.envs.grid8_supervisor import SupervisorSumoEnvironment
from traffic_rl.supervisor.agent import SupervisorAgent
from traffic_rl.core.agent import DDQNAgent
from traffic_rl.sumo_gen.grid8 import generate_all as generate_sumo

# ─── Directories ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR_AGENTS      = 'outputs/checkpoints/supervisor/agents'
CHECKPOINT_DIR_SUPERVISORS = 'outputs/checkpoints/supervisor/supervisors'
RESULTS_DIR                = 'outputs/results/supervisor'

# ─── Weight mapping: Group B mirrors Group A (for partial transfer) ───────────
WEIGHT_MAP = {
    'tls_1': 'tls_1', 'tls_2': 'tls_2', 'tls_3': 'tls_3', 'tls_4': 'tls_4',
    'tls_5': 'tls_1', 'tls_6': 'tls_2', 'tls_7': 'tls_3', 'tls_8': 'tls_4',
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: set random seeds
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Helper: partial weight transfer (8-dim → 7-dim)
# ─────────────────────────────────────────────────────────────────────────────
def partial_transfer(agent_7dim, checkpoint_path):
    """
    Load an 8-dim checkpoint into a 7-dim agent.

    DDQNAgent saves with keys:
      'online_network_state_dict', 'target_network_state_dict'
    Layer names: 'network.0.weight', 'network.2.weight', 'network.4.weight'

    Strategy for input layer (network.0.weight):
      Old: shape [128, 8] — columns 0-5 = local features, 6-7 = neighbor queues
      New: shape [128, 7] — columns 0-5 copied, column 6 is new random
                            (will learn supervisor signal mapping)

    All other layers copied as-is.
    """
    if not os.path.exists(checkpoint_path):
        print(f"    ⚠  No checkpoint at {checkpoint_path} — using random weights")
        return

    device = agent_7dim.device
    ckpt   = torch.load(checkpoint_path, map_location=device)

    # ── Online network ───────────────────────────────────────────
    old_state = ckpt['online_network_state_dict']
    new_state = agent_7dim.online_network.state_dict()

    # Input layer: take first 6 columns only
    old_w = old_state['network.0.weight']          # [128, 8]
    new_w = new_state['network.0.weight'].clone()  # [128, 7]
    new_w[:, :6] = old_w[:, :6]                   # copy local feature weights
    # column 6 (supervisor signal weight) stays random
    new_state['network.0.weight'] = new_w

    # Copy bias + all other layers as-is
    for key in ['network.0.bias', 'network.2.weight', 'network.2.bias',
                'network.4.weight', 'network.4.bias']:
        if key in old_state:
            new_state[key] = old_state[key]

    agent_7dim.online_network.load_state_dict(new_state)

    # ── Target network: same transfer ───────────────────────────
    old_target = ckpt['target_network_state_dict']
    new_target = agent_7dim.target_network.state_dict()

    old_tw = old_target['network.0.weight']
    new_tw = new_target['network.0.weight'].clone()
    new_tw[:, :6] = old_tw[:, :6]
    new_target['network.0.weight'] = new_tw

    for key in ['network.0.bias', 'network.2.weight', 'network.2.bias',
                'network.4.weight', 'network.4.bias']:
        if key in old_target:
            new_target[key] = old_target[key]

    agent_7dim.target_network.load_state_dict(new_target)

    print(f"    ✓ Partial transfer done ← {checkpoint_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(env, agents, supervisor_a, supervisor_b,
          num_episodes=500, target_update_freq=10,
          save_freq=20, resume_from=0):
    """
    Hierarchical training loop.

    Every simulation step:
      Local states → Supervisor signals → Enhanced states →
      Agent actions → SUMO step → Individual rewards →
      Train agents + supervisors
    """
    os.makedirs(CHECKPOINT_DIR_AGENTS,      exist_ok=True)
    os.makedirs(CHECKPOINT_DIR_SUPERVISORS, exist_ok=True)
    os.makedirs(RESULTS_DIR,                exist_ok=True)

    print("\n" + "=" * 70)
    print("  SUPERVISOR TRAINING — 8 Intersections, 2 Supervisors")
    print(f"  Episodes: {resume_from + 1} → {resume_from + num_episodes}")
    print("=" * 70)

    history = {
        'episode_rewards'   : [],
        'group_a_rewards'   : [],
        'group_b_rewards'   : [],
        'sup_a_loss'        : [],
        'sup_b_loss'        : [],
        'per_intersection'  : {tls: [] for tls in env.tls_ids},
    }

    for episode in tqdm(range(num_episodes), desc="Training"):
        actual_ep = resume_from + episode + 1

        # ── Episode reset ──────────────────────────────────────────
        local_states = env.reset()                                # {tls: 6-dim}

        # Initial supervisor signals
        signals_a = supervisor_a.get_signals(
            {tls: local_states[tls] for tls in env.group_a})
        signals_b = supervisor_b.get_signals(
            {tls: local_states[tls] for tls in env.group_b})

        enhanced_states = env.build_enhanced_states(
            local_states, signals_a, signals_b)                   # {tls: 7-dim}

        ep_reward = {tls: 0.0 for tls in env.tls_ids}
        sup_losses_a, sup_losses_b = [], []
        done = False

        # ── Episode loop ───────────────────────────────────────────
        while not done:
            # 1. Agents pick actions from 7-dim enhanced state
            actions = {
                tls: agents[tls].select_action(enhanced_states[tls], training=True)
                for tls in env.tls_ids
            }

            # 2. Environment step → individual rewards
            next_local_states, rewards, done, info = env.step(actions)

            # 3. Next supervisor signals
            next_signals_a = supervisor_a.get_signals(
                {tls: next_local_states[tls] for tls in env.group_a})
            next_signals_b = supervisor_b.get_signals(
                {tls: next_local_states[tls] for tls in env.group_b})

            next_enhanced = env.build_enhanced_states(
                next_local_states, next_signals_a, next_signals_b)

            # 4. Group average rewards (for supervisors)
            group_a_reward = env.get_group_avg_reward(rewards, env.group_a)
            group_b_reward = env.get_group_avg_reward(rewards, env.group_b)

            # 5. Store + train local agents (individual reward)
            for tls in env.tls_ids:
                agents[tls].memory.store(
                    enhanced_states[tls], actions[tls],
                    rewards[tls], next_enhanced[tls], done
                )
                agents[tls].train()
                ep_reward[tls] += rewards[tls]

            # 6. Store + train supervisors (group avg reward)
            supervisor_a.store(
                {tls: local_states[tls]      for tls in env.group_a},
                group_a_reward,
                {tls: next_local_states[tls] for tls in env.group_a},
                done
            )
            loss_a = supervisor_a.train()
            if loss_a is not None:
                sup_losses_a.append(loss_a)

            supervisor_b.store(
                {tls: local_states[tls]      for tls in env.group_b},
                group_b_reward,
                {tls: next_local_states[tls] for tls in env.group_b},
                done
            )
            loss_b = supervisor_b.train()
            if loss_b is not None:
                sup_losses_b.append(loss_b)

            # 7. Advance state
            local_states   = next_local_states
            enhanced_states = next_enhanced

        # ── Post-episode updates ───────────────────────────────────
        if (episode + 1) % target_update_freq == 0:
            for tls in env.tls_ids:
                agents[tls].update_target_network()
            supervisor_a.update_target_network()
            supervisor_b.update_target_network()

        for tls in env.tls_ids:
            agents[tls].decay_epsilon()

        # ── Record metrics ─────────────────────────────────────────
        network_reward = sum(ep_reward.values())
        group_a_tot    = sum(ep_reward[t] for t in env.group_a)
        group_b_tot    = sum(ep_reward[t] for t in env.group_b)

        history['episode_rewards'].append(network_reward)
        history['group_a_rewards'].append(group_a_tot)
        history['group_b_rewards'].append(group_b_tot)
        history['sup_a_loss'].append(np.mean(sup_losses_a) if sup_losses_a else 0)
        history['sup_b_loss'].append(np.mean(sup_losses_b) if sup_losses_b else 0)
        for tls in env.tls_ids:
            history['per_intersection'][tls].append(ep_reward[tls])

        # ── Checkpoints ────────────────────────────────────────────
        if (episode + 1) % save_freq == 0:
            for tls in env.tls_ids:
                agents[tls].save(f'{CHECKPOINT_DIR_AGENTS}/{tls}_episode_{actual_ep}.pth')
            supervisor_a.save(f'{CHECKPOINT_DIR_SUPERVISORS}/supervisor_a_episode_{actual_ep}.pth')
            supervisor_b.save(f'{CHECKPOINT_DIR_SUPERVISORS}/supervisor_b_episode_{actual_ep}.pth')
            print(f"\n  ✓ Checkpoint saved at episode {actual_ep}")

    env.close()

    # ── Save final models ──────────────────────────────────────────
    for tls in env.tls_ids:
        agents[tls].save(f'{CHECKPOINT_DIR_AGENTS}/{tls}_final.pth')
    supervisor_a.save(f'{CHECKPOINT_DIR_SUPERVISORS}/supervisor_a_final.pth')
    supervisor_b.save(f'{CHECKPOINT_DIR_SUPERVISORS}/supervisor_b_final.pth')

    # ── Save CSV ───────────────────────────────────────────────────
    df = pd.DataFrame({
        'Network_Reward' : history['episode_rewards'],
        'Group_A_Reward' : history['group_a_rewards'],
        'Group_B_Reward' : history['group_b_rewards'],
        'Sup_A_Loss'     : history['sup_a_loss'],
        'Sup_B_Loss'     : history['sup_b_loss'],
    })
    for tls in env.tls_ids:
        df[tls] = history['per_intersection'][tls]
    df.to_csv(f'{RESULTS_DIR}/training_history.csv', index=False)

    print("\n✅ Training complete!")
    print(f"   Agents     → {CHECKPOINT_DIR_AGENTS}/")
    print(f"   Supervisors→ {CHECKPOINT_DIR_SUPERVISORS}/")
    print(f"   History    → {RESULTS_DIR}/training_history.csv")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(env, agents, supervisor_a, supervisor_b, num_episodes=20):
    """Evaluate the full supervisor system (greedy policy)."""
    print("\n" + "=" * 70)
    print(f"  EVALUATION: {num_episodes} Episodes")
    print("=" * 70)

    ep_rewards     = {tls: [] for tls in env.tls_ids}
    network_rewards, group_a_rewards, group_b_rewards, wait_times = [], [], [], []

    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        local_states = env.reset()
        signals_a    = supervisor_a.get_signals({tls: local_states[tls] for tls in env.group_a})
        signals_b    = supervisor_b.get_signals({tls: local_states[tls] for tls in env.group_b})
        enhanced     = env.build_enhanced_states(local_states, signals_a, signals_b)

        ep_reward = {tls: 0.0 for tls in env.tls_ids}
        done = False

        while not done:
            actions = {
                tls: agents[tls].select_action(enhanced[tls], training=False)
                for tls in env.tls_ids
            }
            next_local, rewards, done, info = env.step(actions)

            next_sig_a = supervisor_a.get_signals({tls: next_local[tls] for tls in env.group_a})
            next_sig_b = supervisor_b.get_signals({tls: next_local[tls] for tls in env.group_b})
            enhanced   = env.build_enhanced_states(next_local, next_sig_a, next_sig_b)
            local_states = next_local

            for tls in env.tls_ids:
                ep_reward[tls] += rewards[tls]
            if done:
                wait_times.append(info['avg_waiting_time'])

        for tls in env.tls_ids:
            ep_rewards[tls].append(ep_reward[tls])
        net = sum(ep_reward.values())
        network_rewards.append(net)
        group_a_rewards.append(sum(ep_reward[t] for t in env.group_a))
        group_b_rewards.append(sum(ep_reward[t] for t in env.group_b))

    env.close()

    # ── Print results ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    print("\nPer-Intersection:")
    for tls in env.tls_ids:
        avg = np.mean(ep_rewards[tls])
        std = np.std(ep_rewards[tls])
        grp = "A" if tls in env.group_a else "B"
        print(f"  {tls} (Group {grp}): {avg:.1f} ± {std:.1f}")

    print(f"\nGroup A Avg/intersection : {np.mean(group_a_rewards)/4:.1f}")
    print(f"Group B Avg/intersection : {np.mean(group_b_rewards)/4:.1f}")
    print(f"Network Total            : {np.mean(network_rewards):.1f} ± {np.std(network_rewards):.1f}")
    print(f"Avg per Intersection     : {np.mean(network_rewards)/8:.1f}")
    print(f"Avg Waiting Time         : {np.mean(wait_times):.3f}s")
    print(f"\n  8-Intersection Baseline : ~-197 avg/intersection")
    diff = np.mean(network_rewards)/8 - (-197)
    pct  = diff / 197 * 100
    direction = "↑ better" if diff > 0 else "↓ worse"
    print(f"  Supervisor vs Baseline  : {diff:+.1f} ({pct:+.1f}%) {direction}")
    print("=" * 70)

    # ── Save CSV ─────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for ep in range(num_episodes):
        row = {
            'episode'        : ep + 1,
            'network_reward' : network_rewards[ep],
            'group_a_reward' : group_a_rewards[ep],
            'group_b_reward' : group_b_rewards[ep],
            'avg_wait_time'  : wait_times[ep] if ep < len(wait_times) else None,
        }
        for tls in env.tls_ids:
            row[tls] = ep_rewards[tls][ep]
        rows.append(row)
    pd.DataFrame(rows).to_csv(f'{RESULTS_DIR}/eval_results.csv', index=False)
    print(f"✅ Eval results saved → {RESULTS_DIR}/eval_results.csv")

    return {'per_intersection': ep_rewards, 'network': network_rewards, 'wait_times': wait_times}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    print("=" * 70)
    print("  8-Intersection DDQN + Local Supervisors")
    print("=" * 70)

    set_seed(args.seed)

    # ── Generate SUMO files if needed ─────────────────────────────
    if not os.path.exists('sumo_files/grid8/network.net.xml'):
        print("\nGenerating SUMO network files...")
        generate_sumo()

    # ── Environment ───────────────────────────────────────────────
    env = SupervisorSumoEnvironment(
        use_gui     = args.gui,
        num_seconds = args.num_seconds,
        delta_time  = args.delta_time,
    )
    print(f"\n  Episode duration : {args.num_seconds}s")
    print(f"  Delta-time       : {args.delta_time}s")
    print(f"  Steps/episode    : {args.num_seconds // args.delta_time}")
    print(f"  Agent state dim  : {env.get_state_dim()} (6 local + 1 supervisor signal)")
    print(f"  Supervisor input : {env.get_local_state_dim() * 4} (4 × 6-dim states)")

    # ── Local agents (7-dim) ──────────────────────────────────────
    print("\nInitialising 8 local agents (7-dim input)...")
    agents = {}
    for tls in env.tls_ids:
        agents[tls] = DDQNAgent(
            state_dim      = env.get_state_dim(),     # 7
            action_dim     = env.get_action_dim(),    # 2
            hidden_dim     = 128,
            learning_rate  = args.learning_rate,
            epsilon_start  = args.epsilon,
            epsilon_decay  = 0.995,
            epsilon_min    = 0.01,
        )

    # ── Supervisors (24-dim input) ────────────────────────────────
    print("\nInitialising 2 supervisor agents (24-dim input)...")
    supervisor_a = SupervisorAgent(
        group_tls_ids       = env.group_a,
        state_dim_per_agent = env.get_local_state_dim(),   # 6
        hidden_dim          = 64,
        learning_rate       = args.sup_lr,
        gamma               = 0.95,
        buffer_capacity     = 10_000,
        batch_size          = 64,
    )
    supervisor_b = SupervisorAgent(
        group_tls_ids       = env.group_b,
        state_dim_per_agent = env.get_local_state_dim(),
        hidden_dim          = 64,
        learning_rate       = args.sup_lr,
        gamma               = 0.95,
        buffer_capacity     = 10_000,
        batch_size          = 64,
    )

    # ── Load weights ──────────────────────────────────────────────
    if args.resume_from > 0:
        print(f"\nResuming from episode {args.resume_from}...")
        for tls in env.tls_ids:
            path = f'{CHECKPOINT_DIR_AGENTS}/{tls}_episode_{args.resume_from}.pth'
            if os.path.exists(path):
                agents[tls].load(path)
                agents[tls].epsilon = max(args.epsilon * (0.995 ** args.resume_from), 0.01)
        supervisor_a.load(f'{CHECKPOINT_DIR_SUPERVISORS}/supervisor_a_episode_{args.resume_from}.pth')
        supervisor_b.load(f'{CHECKPOINT_DIR_SUPERVISORS}/supervisor_b_episode_{args.resume_from}.pth')

    elif args.load_final:
        print("\nLoading final supervisor models...")
        for tls in env.tls_ids:
            agents[tls].load(f'{CHECKPOINT_DIR_AGENTS}/{tls}_final.pth')
        supervisor_a.load(f'{CHECKPOINT_DIR_SUPERVISORS}/supervisor_a_final.pth')
        supervisor_b.load(f'{CHECKPOINT_DIR_SUPERVISORS}/supervisor_b_final.pth')

    elif not args.from_scratch:
        print("\nPartial weight transfer from 8-intersection checkpoints...")
        print("  (7-dim agents: columns 0-5 from 8-dim checkpoint, col-6 is new)")
        for tls in env.tls_ids:
            src  = WEIGHT_MAP[tls]
            path = f'outputs/checkpoints/grid8/{src}_final.pth'
            partial_transfer(agents[tls], path)
            agents[tls].epsilon = args.epsilon
            grp = 'A' if tls in env.group_a else 'B'
            print(f"  {tls} (Group {grp}) ← {src}_final.pth")
    else:
        print("\n  Training all agents from scratch (random weights)")

    # ── Run mode ──────────────────────────────────────────────────
    if args.mode in ('train', 'all'):
        train(env, agents, supervisor_a, supervisor_b,
              num_episodes     = args.episodes,
              target_update_freq = 10,
              save_freq        = args.save_freq,
              resume_from      = args.resume_from)

    if args.mode in ('evaluate', 'all'):
        env2 = SupervisorSumoEnvironment(
            use_gui=args.gui, num_seconds=args.num_seconds, delta_time=args.delta_time)
        evaluate(env2, agents, supervisor_a, supervisor_b, args.eval_episodes)

    print("\n" + "=" * 70)
    print("✅ Supervisor Experiment Complete!")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='8-Intersection with Local Supervisors')

    parser.add_argument('--mode',          type=str,   default='train',
                        choices=['train', 'evaluate', 'all'])
    parser.add_argument('--episodes',      type=int,   default=500)
    parser.add_argument('--eval-episodes', type=int,   default=20)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--sup-lr',        type=float, default=0.001,
                        help='Supervisor learning rate (default: 0.001)')
    parser.add_argument('--epsilon',       type=float, default=0.3,
                        help='Initial epsilon (default: 0.3 — lower since partial transfer)')
    parser.add_argument('--save-freq',     type=int,   default=20)
    parser.add_argument('--resume-from',   type=int,   default=0)
    parser.add_argument('--load-final',    action='store_true')
    parser.add_argument('--from-scratch',  action='store_true')
    parser.add_argument('--gui',           action='store_true')
    parser.add_argument('--num-seconds',   type=int,   default=1800)
    parser.add_argument('--delta-time',    type=int,   default=5)
    parser.add_argument('--seed',          type=int,   default=42)

    args = parser.parse_args()
    main(args)
