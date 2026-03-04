"""
Hierarchical Federated Training Loop for 8-Intersection Traffic Control

Training Architecture:
  - 8 Local DDQN Agents (one per intersection, state_dim=8, action_dim=2)
  - 2 Supervisor DDQN Agents (one per zone, state_dim=24, action_dim=3)
  - FedAvg Coordinator (intra-zone every 10 eps, inter-zone every 25 eps)

Training Flow per step:
  1. Supervisors observe zone states, select coordination actions
  2. Local agents observe local states, select traffic actions
  3. Environment executes all 8 actions simultaneously
  4. Supervisor reward modifiers adjust local rewards
  5. All agents store experience and train
  6. FedAvg aggregates weights periodically
"""

import numpy as np
import os
import csv
from tqdm import tqdm

from sumo_environment_federated import FederatedSumoEnvironment
from agent import DDQNAgent
from supervisor_agent import SupervisorAgent
from federated_learning import FederatedCoordinator


def train_federated(num_episodes=700,
                    target_update_freq=10,
                    save_freq=20,
                    log_freq=10,
                    use_gui=False,
                    finetune=False):
    """
    Train the full federated hierarchical system.

    Args:
        num_episodes: Total training episodes
        target_update_freq: Update target networks every N episodes
        save_freq: Save checkpoints every N episodes
        log_freq: Print detailed log every N episodes
        use_gui: Launch SUMO GUI for visual debugging
        finetune: If True, load cooperative 4-intersection weights + use lower lr/epsilon

    Returns:
        history: Dict with all training metrics
    """
    # ==================== Setup ====================
    mode = "FINE-TUNING (from cooperative 4-intersection)" if finetune else "FROM SCRATCH"
    print("=" * 70)
    print("  FEDERATED HIERARCHICAL MULTI-AGENT TRAINING")
    print(f"  Mode: {mode}")
    print("  8 Intersections | 2 Zones | 2 Supervisors | FedAvg")
    print("=" * 70)

    # Output directories
    checkpoint_dir = 'checkpoints/federated_finetuned' if finetune else 'checkpoints/federated'
    results_dir = 'results/federated_finetuned' if finetune else 'results/federated'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Initialize environment
    env = FederatedSumoEnvironment(use_gui=use_gui)
    local_state_dim = env.get_local_state_dim()    # 8
    zone_state_dim = env.get_zone_state_dim()       # 12
    action_dim = env.get_action_dim()               # 2

    print(f"\n  Local Agent: state_dim={local_state_dim}, action_dim={action_dim}")
    print(f"  Supervisor:  state_dim={zone_state_dim * 2}, action_dim=3")
    print()

    # Hyperparameters (adjusted for fine-tuning)
    if finetune:
        local_lr = 0.0001       # Gentle learning rate
        local_epsilon = 0.1     # Low exploration (pretrained knowledge)
        local_eps_decay = 0.998 # Slow decay
        sup_lr = 0.0003
        sup_epsilon = 0.3
        sup_eps_decay = 0.998
    else:
        local_lr = 0.001
        local_epsilon = 1.0
        local_eps_decay = 0.995
        sup_lr = 0.0005
        sup_epsilon = 1.0
        sup_eps_decay = 0.997

    # Initialize 8 local DDQN agents
    local_agents = {}
    for tls_id in env.tls_ids:
        local_agents[tls_id] = DDQNAgent(
            state_dim=local_state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=local_lr,
            gamma=0.95,
            epsilon_start=local_epsilon,
            epsilon_decay=local_eps_decay,
            epsilon_min=0.01,
            batch_size=64,
            buffer_capacity=10000
        )

    # Load pretrained cooperative weights if fine-tuning
    if finetune:
        cooperative_dir = 'checkpoints/cooperative'
        # Map: TLS 1-4 get their own cooperative weights, TLS 5-8 get copies
        weight_map = {
            'tls_1': 'tls_1_final.pth',
            'tls_2': 'tls_2_final.pth',
            'tls_3': 'tls_3_final.pth',
            'tls_4': 'tls_4_final.pth',
            'tls_5': 'tls_1_final.pth',  # Zone B gets Zone A weights as starting point
            'tls_6': 'tls_2_final.pth',
            'tls_7': 'tls_3_final.pth',
            'tls_8': 'tls_4_final.pth',
        }
        print("\n  Loading cooperative 4-intersection weights:")
        for tls_id, ckpt_name in weight_map.items():
            ckpt_path = os.path.join(cooperative_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                local_agents[tls_id].load(ckpt_path)
                src = ckpt_name.replace('_final.pth', '')
                print(f"    ✓ {tls_id} ← {src} weights")
            else:
                print(f"    ⚠ {ckpt_path} not found, using random init")
        print()

    print(f"  ✓ Initialized {len(local_agents)} local agents")

    # Initialize 2 supervisor agents
    supervisor_a = SupervisorAgent(
        zone_name='zone_a',
        state_dim=zone_state_dim * 2,   # 24 (own zone + neighbor zone)
        action_dim=3,
        hidden_dim=256,
        learning_rate=sup_lr,
        gamma=0.95,
        epsilon_start=sup_epsilon,
        epsilon_decay=sup_eps_decay,
        epsilon_min=0.05,
        batch_size=32,
        buffer_capacity=5000,
        decision_interval=3
    )

    supervisor_b = SupervisorAgent(
        zone_name='zone_b',
        state_dim=zone_state_dim * 2,
        action_dim=3,
        hidden_dim=256,
        learning_rate=sup_lr,
        gamma=0.95,
        epsilon_start=sup_epsilon,
        epsilon_decay=sup_eps_decay,
        epsilon_min=0.05,
        batch_size=32,
        buffer_capacity=5000,
        decision_interval=3
    )
    print("  ✓ Initialized 2 supervisor agents")

    # Initialize FedAvg coordinator
    coordinator = FederatedCoordinator(
        local_agents=local_agents,
        supervisor_a=supervisor_a,
        supervisor_b=supervisor_b,
        intra_zone_interval=10,
        inter_zone_interval=25,
        intra_zone_alpha=0.8,
        inter_zone_alpha=0.5
    )
    print("  ✓ Initialized FedAvg coordinator")

    # ==================== Training History ====================
    history = {
        'episode_rewards': [],          # Total reward per episode
        'zone_a_rewards': [],           # Zone A cumulative reward
        'zone_b_rewards': [],           # Zone B cumulative reward
        'avg_waiting_time': [],
        'avg_queue_length': [],
        'supervisor_a_actions': [],     # Coordination action distribution
        'supervisor_b_actions': [],
        'epsilon_local': [],
        'epsilon_supervisor': [],
        'fedavg_events': [],
        'per_intersection_queues': [],
    }

    print(f"\n  Starting training: {num_episodes} episodes")
    print("-" * 70)

    # ==================== Training Loop ====================
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Reset environment
        states = env.reset()
        episode_reward = 0
        zone_a_reward = 0
        zone_b_reward = 0
        episode_losses = {tls: [] for tls in env.tls_ids}
        sup_a_losses = []
        sup_b_losses = []
        sup_a_actions_ep = []
        sup_b_actions_ep = []

        # Get initial zone states for supervisors
        zone_state_a = env.get_zone_state('zone_a')
        zone_state_b = env.get_zone_state('zone_b')

        # Initial supervisor decisions
        sup_action_a = supervisor_a.select_action(zone_state_a, zone_state_b, training=True)
        sup_action_b = supervisor_b.select_action(zone_state_b, zone_state_a, training=True)
        sup_a_actions_ep.append(sup_action_a)
        sup_b_actions_ep.append(sup_action_b)

        done = False
        step = 0

        while not done:
            step += 1

            # --- Supervisor decisions (every N local steps) ---
            prev_zone_a = zone_state_a.copy()
            prev_zone_b = zone_state_b.copy()
            prev_sup_a = sup_action_a
            prev_sup_b = sup_action_b

            if supervisor_a.should_decide():
                zone_state_a = env.get_zone_state('zone_a')
                zone_state_b = env.get_zone_state('zone_b')
                sup_action_a = supervisor_a.select_action(zone_state_a, zone_state_b, training=True)
                sup_action_b = supervisor_b.select_action(zone_state_b, zone_state_a, training=True)
                sup_a_actions_ep.append(sup_action_a)
                sup_b_actions_ep.append(sup_action_b)

            # --- Local agent actions ---
            actions = {}
            for tls_id in env.tls_ids:
                actions[tls_id] = local_agents[tls_id].select_action(
                    states[tls_id], training=True
                )

            # --- Environment step ---
            next_states, rewards, done, info = env.step(actions)

            # --- Apply supervisor reward modifiers ---
            for tls_id in env.tls_ids:
                zone = env.tls_to_zone[tls_id]
                supervisor = supervisor_a if zone == 'zone_a' else supervisor_b
                modifier = supervisor.get_reward_modifier(
                    tls_id,
                    actions[tls_id],
                    env.current_phases[tls_id]
                )
                rewards[tls_id] += modifier

            # --- Store experience and train local agents ---
            for tls_id in env.tls_ids:
                local_agents[tls_id].memory.store(
                    states[tls_id],
                    actions[tls_id],
                    rewards[tls_id],
                    next_states[tls_id],
                    done
                )
                loss = local_agents[tls_id].train()
                if loss is not None:
                    episode_losses[tls_id].append(loss)

            # --- Store supervisor experience and train ---
            if step % supervisor_a.decision_interval == 0:
                new_zone_a = env.get_zone_state('zone_a')
                new_zone_b = env.get_zone_state('zone_b')
                zone_reward_a = env.get_zone_reward('zone_a')
                zone_reward_b = env.get_zone_reward('zone_b')

                supervisor_a.store_experience(
                    prev_zone_a, prev_zone_b, prev_sup_a,
                    zone_reward_a,
                    new_zone_a, new_zone_b, done
                )
                supervisor_b.store_experience(
                    prev_zone_b, prev_zone_a, prev_sup_b,
                    zone_reward_b,
                    new_zone_b, new_zone_a, done
                )

                sup_loss_a = supervisor_a.train()
                sup_loss_b = supervisor_b.train()
                if sup_loss_a is not None:
                    sup_a_losses.append(sup_loss_a)
                if sup_loss_b is not None:
                    sup_b_losses.append(sup_loss_b)

            # Accumulate rewards
            for tls_id in env.tls_ids:
                episode_reward += rewards[tls_id]
                if env.tls_to_zone[tls_id] == 'zone_a':
                    zone_a_reward += rewards[tls_id]
                else:
                    zone_b_reward += rewards[tls_id]

            states = next_states

        # ==================== End of Episode ====================

        # Update target networks
        if (episode + 1) % target_update_freq == 0:
            for agent in local_agents.values():
                agent.update_target_network()
            supervisor_a.update_target_network()
            supervisor_b.update_target_network()

        # Decay epsilon
        for agent in local_agents.values():
            agent.decay_epsilon()
        supervisor_a.decay_epsilon()
        supervisor_b.decay_epsilon()

        # FedAvg aggregation
        fedavg_msg = coordinator.maybe_aggregate(episode + 1)
        if fedavg_msg:
            history['fedavg_events'].append((episode + 1, fedavg_msg))

        # Collect metrics
        final_metrics = info
        history['episode_rewards'].append(episode_reward)
        history['zone_a_rewards'].append(zone_a_reward)
        history['zone_b_rewards'].append(zone_b_reward)
        history['avg_waiting_time'].append(final_metrics['avg_waiting_time'])
        history['avg_queue_length'].append(
            sum(final_metrics['per_intersection'][t]['queue'] for t in env.tls_ids) / 8
        )
        history['epsilon_local'].append(local_agents['tls_1'].epsilon)
        history['epsilon_supervisor'].append(supervisor_a.epsilon)

        # Supervisor action distribution
        sup_a_dist = [sup_a_actions_ep.count(i) / max(len(sup_a_actions_ep), 1) for i in range(3)]
        sup_b_dist = [sup_b_actions_ep.count(i) / max(len(sup_b_actions_ep), 1) for i in range(3)]
        history['supervisor_a_actions'].append(sup_a_dist)
        history['supervisor_b_actions'].append(sup_b_dist)

        # Per-intersection queues
        per_int_q = {t: final_metrics['per_intersection'][t]['queue'] for t in env.tls_ids}
        history['per_intersection_queues'].append(per_int_q)

        # ==================== Logging ====================
        if (episode + 1) % log_freq == 0:
            recent = history['episode_rewards'][-log_freq:]
            avg_r = np.mean(recent)
            print(f"\n  Episode {episode + 1}/{num_episodes}")
            print(f"    Avg Reward (last {log_freq}): {avg_r:.1f}")
            print(f"    Zone A reward: {zone_a_reward:.1f} | Zone B reward: {zone_b_reward:.1f}")
            print(f"    Avg Wait: {final_metrics['avg_waiting_time']:.1f}s")
            print(f"    Avg Queue: {history['avg_queue_length'][-1]:.1f}")
            print(f"    ε local: {local_agents['tls_1'].epsilon:.3f} | ε sup: {supervisor_a.epsilon:.3f}")
            print(f"    Sup A: NS={sup_a_dist[0]:.0%} EW={sup_a_dist[1]:.0%} BAL={sup_a_dist[2]:.0%}")
            print(f"    Sup B: NS={sup_b_dist[0]:.0%} EW={sup_b_dist[1]:.0%} BAL={sup_b_dist[2]:.0%}")
            if fedavg_msg:
                print(f"    FedAvg: {fedavg_msg}")

        # ==================== Checkpointing ====================
        if (episode + 1) % save_freq == 0:
            _save_checkpoints(local_agents, supervisor_a, supervisor_b, episode + 1, checkpoint_dir)

    # ==================== Final Save ====================
    env.close()
    _save_checkpoints(local_agents, supervisor_a, supervisor_b, 'final', checkpoint_dir)
    _save_history(history, results_dir)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print(f"  FedAvg Stats: {coordinator.get_stats()['intra_zone_aggregations']} intra-zone, "
          f"{coordinator.get_stats()['inter_zone_aggregations']} inter-zone")
    print("=" * 70)

    return history


def _save_checkpoints(local_agents, supervisor_a, supervisor_b, tag, checkpoint_dir='checkpoints/federated'):
    """Save all agent checkpoints"""
    for tls_id, agent in local_agents.items():
        path = f'{checkpoint_dir}/{tls_id}_episode_{tag}.pth'
        agent.save(path)

    supervisor_a.save(f'{checkpoint_dir}/supervisor_a_episode_{tag}.pth')
    supervisor_b.save(f'{checkpoint_dir}/supervisor_b_episode_{tag}.pth')


def _save_history(history, results_dir='results/federated'):
    """Save training history to CSV"""
    csv_path = f'{results_dir}/training_history.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'Episode', 'Total_Reward', 'Zone_A_Reward', 'Zone_B_Reward',
            'Avg_Waiting_Time', 'Avg_Queue', 'Epsilon_Local', 'Epsilon_Supervisor',
            'Sup_A_NS', 'Sup_A_EW', 'Sup_A_BAL',
            'Sup_B_NS', 'Sup_B_EW', 'Sup_B_BAL'
        ]
        # Add per-intersection queue columns
        for i in range(1, 9):
            header.append(f'Queue_TLS_{i}')
        writer.writerow(header)

        for i in range(len(history['episode_rewards'])):
            row = [
                i + 1,
                history['episode_rewards'][i],
                history['zone_a_rewards'][i],
                history['zone_b_rewards'][i],
                history['avg_waiting_time'][i],
                history['avg_queue_length'][i],
                history['epsilon_local'][i],
                history['epsilon_supervisor'][i],
            ]
            # Supervisor action distributions
            row.extend(history['supervisor_a_actions'][i])
            row.extend(history['supervisor_b_actions'][i])
            # Per-intersection queues
            for j in range(1, 9):
                tls = f'tls_{j}'
                row.append(history['per_intersection_queues'][i].get(tls, 0))

            writer.writerow(row)

    print(f"  Training history saved to {csv_path}")


if __name__ == '__main__':
    train_federated(num_episodes=700, use_gui=False)
