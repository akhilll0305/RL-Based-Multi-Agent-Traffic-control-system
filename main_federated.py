"""
Main Entry Point for Federated Hierarchical Multi-Agent Traffic Control

Usage:
  python main_federated.py              # Train the system (700 episodes)
  python main_federated.py --evaluate   # Evaluate trained model
  python main_federated.py --gui        # Train with SUMO GUI visualization
  python main_federated.py --episodes 500  # Custom episode count
"""

import argparse
import os
import sys
import numpy as np
import csv

from sumo_environment_federated import FederatedSumoEnvironment
from agent import DDQNAgent
from supervisor_agent import SupervisorAgent
from train_federated import train_federated


def evaluate_federated(use_gui=True, num_episodes=5):
    """
    Evaluate trained federated model.

    Args:
        use_gui: Whether to visualize in SUMO GUI
        num_episodes: Number of evaluation episodes
    """
    print("=" * 70)
    print("  EVALUATING FEDERATED HIERARCHICAL MODEL")
    print("=" * 70)

    env = FederatedSumoEnvironment(use_gui=use_gui)
    local_state_dim = env.get_local_state_dim()
    action_dim = env.get_action_dim()
    zone_state_dim = env.get_zone_state_dim()

    # Load local agents
    local_agents = {}
    for tls_id in env.tls_ids:
        agent = DDQNAgent(state_dim=local_state_dim, action_dim=action_dim)
        checkpoint = f'checkpoints/federated/{tls_id}_episode_final.pth'
        if os.path.exists(checkpoint):
            agent.load(checkpoint)
            print(f"  ✓ Loaded {tls_id}")
        else:
            print(f"  ⚠ No checkpoint for {tls_id}, using untrained weights")
        local_agents[tls_id] = agent

    # Load supervisors
    supervisor_a = SupervisorAgent('zone_a', state_dim=zone_state_dim * 2, action_dim=3)
    supervisor_b = SupervisorAgent('zone_b', state_dim=zone_state_dim * 2, action_dim=3)

    if os.path.exists('checkpoints/federated/supervisor_a_episode_final.pth'):
        supervisor_a.load('checkpoints/federated/supervisor_a_episode_final.pth')
        print("  ✓ Loaded Supervisor A")
    if os.path.exists('checkpoints/federated/supervisor_b_episode_final.pth'):
        supervisor_b.load('checkpoints/federated/supervisor_b_episode_final.pth')
        print("  ✓ Loaded Supervisor B")

    # Evaluate
    all_metrics = []

    for ep in range(num_episodes):
        print(f"\n  --- Evaluation Episode {ep + 1}/{num_episodes} ---")
        states = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        # Get supervisor actions
        zone_a_state = env.get_zone_state('zone_a')
        zone_b_state = env.get_zone_state('zone_b')
        sup_a_action = supervisor_a.select_action(zone_a_state, zone_b_state, training=False)
        sup_b_action = supervisor_b.select_action(zone_b_state, zone_a_state, training=False)

        while not done:
            step_count += 1

            # Update supervisor decisions periodically
            if step_count % 3 == 0:
                zone_a_state = env.get_zone_state('zone_a')
                zone_b_state = env.get_zone_state('zone_b')
                sup_a_action = supervisor_a.select_action(zone_a_state, zone_b_state, training=False)
                sup_b_action = supervisor_b.select_action(zone_b_state, zone_a_state, training=False)

            # Local agents act greedily
            actions = {}
            for tls_id in env.tls_ids:
                actions[tls_id] = local_agents[tls_id].select_action(
                    states[tls_id], training=False
                )

            next_states, rewards, done, info = env.step(actions)
            total_reward += sum(rewards.values())
            states = next_states

        # Collect final metrics
        metrics = {
            'episode': ep + 1,
            'total_reward': total_reward,
            'avg_waiting_time': info['avg_waiting_time'],
            'total_vehicles': info['total_vehicles'],
            'network_waiting_time': info['network_waiting_time'],
            'zone_a_queue': info['per_zone']['zone_a']['total_queue'],
            'zone_b_queue': info['per_zone']['zone_b']['total_queue'],
        }
        all_metrics.append(metrics)

        print(f"    Reward:       {total_reward:.1f}")
        print(f"    Avg Wait:     {info['avg_waiting_time']:.1f}s")
        print(f"    Zone A Queue: {info['per_zone']['zone_a']['total_queue']}")
        print(f"    Zone B Queue: {info['per_zone']['zone_b']['total_queue']}")
        print(f"    Vehicles:     {info['total_vehicles']}")

    env.close()

    # Summary
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    avg_reward = np.mean([m['total_reward'] for m in all_metrics])
    avg_wait = np.mean([m['avg_waiting_time'] for m in all_metrics])
    avg_za_q = np.mean([m['zone_a_queue'] for m in all_metrics])
    avg_zb_q = np.mean([m['zone_b_queue'] for m in all_metrics])
    print(f"  Avg Reward:     {avg_reward:.1f}")
    print(f"  Avg Wait Time:  {avg_wait:.1f}s")
    print(f"  Avg Zone A Q:   {avg_za_q:.1f}")
    print(f"  Avg Zone B Q:   {avg_zb_q:.1f}")
    print("=" * 70)

    # Save eval results
    os.makedirs('results/federated', exist_ok=True)
    with open('results/federated/evaluation_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"  Results saved to results/federated/evaluation_results.csv")


def main():
    parser = argparse.ArgumentParser(
        description='Federated Hierarchical Multi-Agent Traffic Control'
    )
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate trained model instead of training')
    parser.add_argument('--gui', action='store_true',
                        help='Use SUMO GUI for visualization')
    parser.add_argument('--episodes', type=int, default=700,
                        help='Number of training episodes (default: 700)')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Number of evaluation episodes (default: 5)')

    args = parser.parse_args()

    if args.evaluate:
        evaluate_federated(use_gui=args.gui, num_episodes=args.eval_episodes)
    else:
        train_federated(num_episodes=args.episodes, use_gui=args.gui)


if __name__ == '__main__':
    main()
