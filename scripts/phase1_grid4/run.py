"""
Multi-Agent DDQN Traffic Light Control - Main Script
Tests and trains 4 agents controlling 4 intersections
"""

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------


import os
import random
import numpy as np
import torch
import argparse
from traffic_rl.envs.grid4 import MultiAgentSumoEnvironment
from traffic_rl.core.agent import DDQNAgent
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt



def load_widened(agent, path):
    """Load a 6-input checkpoint into a wider (cooperative) network.

    The extra neighbour inputs get zero weights, so the agent starts out acting exactly like
    the pretrained single-intersection policy and learns to use its neighbours from there.
    """
    ckpt = torch.load(path, map_location=agent.device, weights_only=False)
    src = ckpt['online_network_state_dict']
    for net in (agent.online_network, agent.target_network):
        dst = net.state_dict()
        for k, v in src.items():
            if dst[k].shape == v.shape:
                dst[k] = v.clone()
            else:                                   # first layer: [hidden, 6] -> [hidden, 8]
                w = torch.zeros_like(dst[k])
                w[:, :v.shape[1]] = v
                dst[k] = w
        net.load_state_dict(dst)


def test_transfer(env, agents, num_episodes=10):
    """
    Test how well Episode 900 transfers to multi-agent setup (NO TRAINING)
    
    Returns:
        Dict with performance metrics
    """
    print("\n" + "="*70)
    print("PHASE 0: Testing Episode 900 Transfer (Frozen Weights)")
    print("="*70)
    print(f"Testing {num_episodes} episodes without any training...")
    print("This tells us if Episode 900 works well for multi-agent.\n")
    
    episode_rewards = {tls: [] for tls in env.tls_ids}
    network_rewards = []
    
    for episode in tqdm(range(num_episodes), desc="Transfer Test"):
        states = env.reset()
        episode_reward = {tls: 0 for tls in env.tls_ids}
        done = False
        
        while not done:
            # Each agent selects action independently (NO EXPLORATION)
            actions = {}
            for tls in env.tls_ids:
                actions[tls] = agents[tls].select_action(states[tls], training=False)
            
            # Step environment
            next_states, rewards, done, info = env.step(actions)
            
            # Accumulate rewards
            for tls in env.tls_ids:
                episode_reward[tls] += rewards[tls]
            
            states = next_states
        
        # Store results
        for tls in env.tls_ids:
            episode_rewards[tls].append(episode_reward[tls])
        network_rewards.append(sum(episode_reward.values()))
    
    env.save_episode_log(f"outputs/results/{'cooperative' if env.cooperative else 'grid4'}/transfer_trips.csv")   # trip metrics, one row per episode
    env.close()
    
    # Calculate statistics
    results = {
        'per_intersection': {},
        'network_total': np.mean(network_rewards),
        'network_std': np.std(network_rewards)
    }
    
    print("\n" + "="*70)
    print("📊 TRANSFER TEST RESULTS")
    print("="*70)
    
    for tls in env.tls_ids:
        avg = np.mean(episode_rewards[tls])
        std = np.std(episode_rewards[tls])
        results['per_intersection'][tls] = {'avg': avg, 'std': std}
        print(f"{tls}: {avg:.1f} ± {std:.1f}")
    
    print(f"\nNetwork Total: {results['network_total']:.1f} ± {results['network_std']:.1f}")
    print(f"Avg per Intersection: {results['network_total']/4:.1f}")
    
    # Interpret results
    avg_per_intersection = results['network_total'] / 4
    print("\n" + "="*70)
    print("📈 EVALUATION")
    print("="*70)
    
    if avg_per_intersection > -6000:
        print("✅ EXCELLENT! Episode 900 transfers very well to multi-agent")
        print("   Recommendation: Proceed with light fine-tuning (50-100 episodes)")
        strategy = "excellent"
    elif avg_per_intersection > -7000:
        print("✅ GOOD! Episode 900 works decently for multi-agent")
        print("   Recommendation: Moderate fine-tuning (200 episodes)")
        strategy = "good"
    elif avg_per_intersection > -8000:
        print("⚠️  OK: Episode 900 needs adaptation")
        print("   Recommendation: Longer fine-tuning (300-500 episodes)")
        strategy = "ok"
    else:
        print("❌ POOR: Episode 900 doesn't transfer well")
        print("   Recommendation: Use hybrid approach or train from scratch")
        strategy = "poor"
    
    print("="*70)
    
    results['strategy'] = strategy
    return results


def fine_tune(env, agents, num_episodes=100, target_update_freq=10, save_freq=20, cooperative=False, resume_from=0):
    """
    Train/fine-tune agents for multi-agent system
    """
    mode_name = "COOPERATIVE" if cooperative else "INDEPENDENT"
    checkpoint_dir = 'outputs/checkpoints/cooperative' if cooperative else 'outputs/checkpoints/grid4'
    history_file = 'outputs/results/cooperative/training_history.csv' if cooperative else 'outputs/results/grid4/training_history.csv'
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('outputs/results/cooperative' if cooperative else 'outputs/results/grid4', exist_ok=True)
    
    print("\n" + "="*70)
    if resume_from > 0:
        print(f"{mode_name} TRAINING: Episodes {resume_from+1}-{resume_from+num_episodes} (Resuming)")
    else:
        print(f"{mode_name} TRAINING: {num_episodes} Episodes")
    print("="*70)
    
    training_history = {
        'episode_rewards': [],
        'per_intersection': {tls: [] for tls in env.tls_ids}
    }
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        actual_episode = resume_from + episode + 1  # Actual episode number for saving
        states = env.reset()
        episode_reward = {tls: 0 for tls in env.tls_ids}
        done = False
        step = 0
        
        while not done:
            # Each agent selects action with low exploration
            actions = {}
            for tls in env.tls_ids:
                actions[tls] = agents[tls].select_action(states[tls], training=True)
            
            # Step environment
            next_states, rewards, done, info = env.step(actions)
            
            # Store experiences and train each agent
            for tls in env.tls_ids:
                agents[tls].memory.store(
                    states[tls], actions[tls], rewards[tls],
                    next_states[tls], done
                )
                agents[tls].train()
                episode_reward[tls] += rewards[tls]
            
            states = next_states
            step += 1
        
        # Update target networks
        if (episode + 1) % target_update_freq == 0:
            for tls in env.tls_ids:
                agents[tls].update_target_network()
        
        # Decay epsilon for all agents
        for tls in env.tls_ids:
            agents[tls].decay_epsilon()
        
        # Store results
        network_reward = sum(episode_reward.values())
        training_history['episode_rewards'].append(network_reward)
        for tls in env.tls_ids:
            training_history['per_intersection'][tls].append(episode_reward[tls])
        
        # Save checkpoints
        if (episode + 1) % save_freq == 0:
            for tls in env.tls_ids:
                agents[tls].save(f'{checkpoint_dir}/{tls}_episode_{actual_episode}.pth')
            print(f"\n✓ Checkpoint saved at episode {actual_episode}")
    
    env.save_episode_log(os.path.join(os.path.dirname(history_file), 'training_trips.csv'))   # trip metrics, one row per episode
    env.close()
    
    # Save final models
    for tls in env.tls_ids:
        agents[tls].save(f'{checkpoint_dir}/{tls}_final.pth')
    
    # Save training history
    df = pd.DataFrame(training_history['episode_rewards'], columns=['Network_Reward'])
    for tls in env.tls_ids:
        df[tls] = training_history['per_intersection'][tls]
    df.to_csv(history_file, index=False)
    
    print("\n✅ Training complete!")
    print(f"Final models saved in: {checkpoint_dir}/")
    
    return training_history


def evaluate_multiagent(env, agents, num_episodes=20):
    """Evaluate multi-agent performance"""
    print("\n" + "="*70)
    print(f"EVALUATION: {num_episodes} Episodes")
    print("="*70)
    
    episode_rewards = {tls: [] for tls in env.tls_ids}
    network_rewards = []
    network_waiting_times = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        states = env.reset()
        episode_reward = {tls: 0 for tls in env.tls_ids}
        done = False
        
        while not done:
            actions = {}
            for tls in env.tls_ids:
                actions[tls] = agents[tls].select_action(states[tls], training=False)
            
            next_states, rewards, done, info = env.step(actions)
            
            for tls in env.tls_ids:
                episode_reward[tls] += rewards[tls]
            
            if done:
                network_waiting_times.append(info['avg_waiting_time'])
            
            states = next_states
        
        for tls in env.tls_ids:
            episode_rewards[tls].append(episode_reward[tls])
        network_rewards.append(sum(episode_reward.values()))
    
    env.save_episode_log(f"outputs/results/{'cooperative' if env.cooperative else 'grid4'}/eval_trips.csv")   # trip metrics, one row per episode
    env.close()
    
    # Print results
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    print("\nPer-Intersection Performance:")
    for tls in env.tls_ids:
        avg = np.mean(episode_rewards[tls])
        std = np.std(episode_rewards[tls])
        print(f"  {tls}: {avg:.1f} ± {std:.1f}")
    
    print(f"\nNetwork Performance:")
    print(f"  Total Reward: {np.mean(network_rewards):.1f} ± {np.std(network_rewards):.1f}")
    print(f"  Avg per Intersection: {np.mean(network_rewards)/4:.1f}")
    env.print_trip_summary(f"outputs/results/{'cooperative' if env.cooperative else 'grid4'}/eval_trips.csv")
    print("="*70)
    
    return {
        'per_intersection': episode_rewards,
        'network_rewards': network_rewards,
        'waiting_times': network_waiting_times
    }


def main(args):
    print("="*70)
    print("Multi-Agent DDQN Traffic Light Control (4 Intersections)")
    print("="*70)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment (calibrated demand is the env default; SUMO seed seed+k
    # for episode k, disjoint from the evaluation seeds starting at 100000)
    env = MultiAgentSumoEnvironment(
        use_gui=args.gui,
        num_seconds=3600,
        delta_time=5,
        cooperative=args.cooperative,
        seed=args.seed,
        neighbor_timing=args.neighbor_timing,
        cycle_clock_s=args.cycle_clock,
        **({'route_file': args.route_file} if args.route_file else {})
    )
    
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    
    print(f"\nEnvironment:")
    print(f"  Intersections: 4")
    print(f"  State dimension: {state_dim} {'(with neighbor info)' if args.cooperative else '(independent)'}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Cooperative: {args.cooperative}")
    
    # Create 4 agents
    print(f"\nInitializing 4 agents...")
    agents = {}
    for tls in env.tls_ids:
        agents[tls] = DDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=args.learning_rate,
            epsilon_start=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=0.01,
            gamma=args.gamma,
            reward_scale=args.reward_scale,
            loss=args.loss
        )
        
        # Load weights: resume, fine-tuned, or pretrained model
        # Skip loading if cooperative (dimension mismatch: 6 vs 8 features)
        if args.resume_from > 0:
            # Resume from specific episode checkpoint
            if args.cooperative:
                resume_path = f'outputs/checkpoints/cooperative/{tls}_episode_{args.resume_from}.pth'
            else:
                resume_path = f'outputs/checkpoints/grid4/{tls}_episode_{args.resume_from}.pth'
            
            if os.path.exists(resume_path):
                agents[tls].load(resume_path)
                # Adjust epsilon to the value it would have at this episode
                decay_factor = args.epsilon_decay ** args.resume_from
                agents[tls].epsilon = max(args.epsilon * decay_factor, agents[tls].epsilon_min)
                print(f"  ✓ {tls}: Resumed from episode {args.resume_from} (epsilon: {agents[tls].epsilon:.4f})")
            else:
                print(f"  ❌ {tls}: Resume checkpoint not found at {resume_path}")
                return
        elif (args.cooperative or state_dim != 6) and not args.load_finetuned \
                and os.path.exists(args.pretrained_model):
            load_widened(agents[tls], args.pretrained_model)
            print(f"  ✓ {tls}: Cooperative mode - {args.pretrained_model} widened to {state_dim} inputs "
                  f"(neighbour weights start at 0)")
        elif args.load_finetuned:
            # Load individual fine-tuned models (check cooperative dir first, then independent)
            if args.cooperative:
                finetuned_path = f'outputs/checkpoints/cooperative/{tls}_final.pth'
            else:
                finetuned_path = f'outputs/checkpoints/grid4/{tls}_final.pth'
            
            if os.path.exists(finetuned_path):
                agents[tls].load(finetuned_path)
                print(f"  ✓ {tls}: Loaded fine-tuned model {finetuned_path}")
            else:
                print(f"  ⚠ {tls}: Fine-tuned model not found at {finetuned_path}")
        elif os.path.exists(args.pretrained_model):
            # Load shared pretrained model (like Episode 900)
            agents[tls].load(args.pretrained_model)
            # load() also restores the checkpoint's epsilon (~0.01 after 900 episodes) and the
            # optimizer's lr (1e-3); fine-tuning is meant to use --epsilon and --learning-rate
            agents[tls].epsilon = args.epsilon
            for g in agents[tls].optimizer.param_groups:
                g['lr'] = args.learning_rate
            print(f"  ✓ {tls}: Loaded {args.pretrained_model} (epsilon {args.epsilon}, lr {args.learning_rate})")
        else:
            print(f"  ⚠ {tls}: Starting with random weights")
    
    # Phase 0: Test transfer (if using pretrained and not cooperative)
    if args.mode == 'test' or args.mode == 'all':
        if args.cooperative:
            print("\n⚠ Cooperative mode - skipping transfer test (dimension mismatch with pretrained model)")
        elif os.path.exists(args.pretrained_model):
            test_results = test_transfer(env, agents, num_episodes=args.test_episodes)
        else:
            print("\n⚠ No pretrained model found. Skipping transfer test.")
            test_results = {'strategy': 'scratch'}
    
    # Phase 1: Train (if requested)
    if args.mode == 'train' or args.mode == 'all':
        fine_tune(env, agents, num_episodes=args.episodes, target_update_freq=10, save_freq=20, cooperative=args.cooperative, resume_from=args.resume_from)
    
    # Phase 2: Evaluate
    if args.mode == 'evaluate' or args.mode == 'all':
        env.seed, env.episode_index = args.eval_seed, 0     # shared evaluation seeds
        evaluate_multiagent(env, agents, num_episodes=args.eval_episodes)
    
    print("\n" + "="*70)
    print("✅ Multi-Agent Experiment Complete!")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Agent DDQN Traffic Control')
    
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'train', 'evaluate', 'all'],
                       help='Mode: test transfer, train, evaluate, or all')
    parser.add_argument('--cooperative', action='store_true',
                       help='Enable cooperation (shared observations and rewards)')
    parser.add_argument('--pretrained-model', type=str, default='outputs/checkpoints/single/ddqn_episode_900.pth',
                       help='Path to pretrained single-agent model')
    parser.add_argument('--load-finetuned', action='store_true',
                       help='Load fine-tuned models from outputs/checkpoints/grid4/ (ignores --pretrained-model)')
    parser.add_argument('--resume-from', type=int, default=0,
                       help='Resume training from episode checkpoint (e.g., 380)')
    parser.add_argument('--test-episodes', type=int, default=10,
                       help='Number of episodes for transfer test')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of fine-tuning episodes')
    parser.add_argument('--eval-episodes', type=int, default=20,
                       help='Number of evaluation episodes')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate for fine-tuning (default: 0.0001)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Initial epsilon for fine-tuning (default: 0.1)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                       help='Per-episode epsilon decay (default: 0.995)')
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--neighbor-timing', action='store_true',
                       help='Cooperative: add neighbour phase and time since switch to the state')
    parser.add_argument('--cycle-clock', type=int, default=None,
                       help='Add a shared background-cycle clock (sin/cos, period in s) to every state')
    parser.add_argument('--route-file', default=None,
                       help='Route file (default: the env default, sumo_files/grid4/calibrated.rou.xml)')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                       help='Multiplies the reward in the TD target (default: 1.0)')
    parser.add_argument('--loss', choices=['mse', 'huber'], default='mse')
    parser.add_argument('--eval-seed', type=int, default=100_000,
                       help='First SUMO seed of the evaluation episodes (test seeds: 100000; '
                            'use another range, e.g. 200000, for validation)')
    parser.add_argument('--gui', action='store_true',
                       help='Use SUMO GUI for visualization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed for torch/numpy/random and the SUMO episode seeds')
    
    args = parser.parse_args()
    
    main(args)
