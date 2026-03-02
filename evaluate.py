"""
Evaluation Script for DDQN Traffic Light Control
"""

import numpy as np
import matplotlib.pyplot as plt
from sumo_environment import SumoEnvironment
from agent import DDQNAgent
import pandas as pd
from tqdm import tqdm


def evaluate_agent(env, agent, num_episodes=100, render=False):
    """
    Evaluate trained DDQN agent
    
    Args:
        env: SumoEnvironment instance
        agent: Trained DDQNAgent instance
        num_episodes: Number of evaluation episodes
        render: Whether to use SUMO GUI
    
    Returns:
        metrics: Dict with evaluation metrics
    """
    print(f"Evaluating agent over {num_episodes} episodes...")
    
    total_rewards = []
    total_waiting_times = []
    total_queues = []
    total_vehicles = []
    phase_switches = []
    
    for episode in tqdm(range(num_episodes), desc="DDQN Agent Evaluation"):
        state = env.reset()
        episode_reward = 0
        switches = 0
        done = False
        
        while not done:
            # Select action (greedy, no exploration)
            action = agent.select_action(state, training=False)
            
            # Track phase switches
            if action == 1:
                switches += 1
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
        
        # Collect metrics
        metrics = env.get_traffic_metrics()
        total_rewards.append(episode_reward)
        total_waiting_times.append(metrics['total_waiting_time'])
        total_queues.append(metrics['total_queue'])
        total_vehicles.append(metrics['total_vehicles'])
        phase_switches.append(switches)
    
    env.close()
    
    # Calculate statistics
    results = {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_waiting_time': np.mean(total_waiting_times),
        'std_waiting_time': np.std(total_waiting_times),
        'avg_queue': np.mean(total_queues),
        'std_queue': np.std(total_queues),
        'avg_vehicles': np.mean(total_vehicles),
        'avg_switches': np.mean(phase_switches)
    }
    
    print("\n=== Evaluation Results ===")
    print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Average Waiting Time: {results['avg_waiting_time']:.2f} ± {results['std_waiting_time']:.2f}s")
    print(f"Average Queue Length: {results['avg_queue']:.2f} ± {results['std_queue']:.2f}")
    print(f"Average Vehicles: {results['avg_vehicles']:.2f}")
    print(f"Average Phase Switches: {results['avg_switches']:.2f}")
    
    return results


def evaluate_fixed_time(env, num_episodes=100, green_duration=30):
    """Evaluate fixed-time traffic light controller"""
    print(f"Evaluating fixed-time controller (green duration: {green_duration}s)...")
    
    total_waiting_times = []
    total_queues = []
    
    for episode in tqdm(range(num_episodes), desc="Fixed-Time Evaluation"):
        env.reset()
        done = False
        step = 0
        
        while not done:
            # Fixed-time logic: switch every green_duration seconds
            if step % (green_duration // env.delta_time) == 0:
                action = 1  # Switch
            else:
                action = 0  # Keep
            
            _, _, done, _ = env.step(action)
            step += 1
        
        metrics = env.get_traffic_metrics()
        total_waiting_times.append(metrics['total_waiting_time'])
        total_queues.append(metrics['total_queue'])
    
    env.close()
    
    results = {
        'avg_waiting_time': np.mean(total_waiting_times),
        'std_waiting_time': np.std(total_waiting_times),
        'avg_queue': np.mean(total_queues),
        'std_queue': np.std(total_queues)
    }
    
    print(f"Fixed-Time Avg Waiting Time: {results['avg_waiting_time']:.2f} ± {results['std_waiting_time']:.2f}s")
    print(f"Fixed-Time Avg Queue: {results['avg_queue']:.2f} ± {results['std_queue']:.2f}")
    
    return results


def evaluate_random(env, num_episodes=100):
    """Evaluate random policy"""
    print("Evaluating random policy...")
    
    total_waiting_times = []
    total_queues = []
    
    for episode in tqdm(range(num_episodes), desc="Random Policy Evaluation"):
        env.reset()
        done = False
        
        while not done:
            action = np.random.randint(0, 2)  # Random action
            _, _, done, _ = env.step(action)
        
        metrics = env.get_traffic_metrics()
        total_waiting_times.append(metrics['total_waiting_time'])
        total_queues.append(metrics['total_queue'])
    
    env.close()
    
    results = {
        'avg_waiting_time': np.mean(total_waiting_times),
        'avg_queue': np.mean(total_queues)
    }
    
    print(f"Random Avg Waiting Time: {results['avg_waiting_time']:.2f}s")
    print(f"Random Avg Queue: {results['avg_queue']:.2f}")
    
    return results


def plot_training_curves(history_csv='results/single_agent/training_history.csv'):
    """Plot training curves from saved history"""
    df = pd.read_csv(history_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(df['Episode'], df['Reward'], alpha=0.3, color='blue')
    axes[0, 0].plot(df['Episode'], df['Reward'].rolling(window=50).mean(), 
                    color='red', linewidth=2, label='Moving Avg (50)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Average Waiting Time
    axes[0, 1].plot(df['Episode'], df['Avg_Waiting_Time'])
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Avg Waiting Time (s)')
    axes[0, 1].set_title('Average Waiting Time per Episode')
    axes[0, 1].grid(True)
    
    # Plot 3: Average Queue Length
    axes[1, 0].plot(df['Episode'], df['Avg_Queue'])
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Avg Queue Length')
    axes[1, 0].set_title('Average Queue Length per Episode')
    axes[1, 0].grid(True)
    
    # Plot 4: Epsilon Decay
    axes[1, 1].plot(df['Episode'], df['Epsilon'])
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].set_title('Exploration Rate (Epsilon)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/single_agent/training_curves.png', dpi=300)
    print("Training curves saved to results/single_agent/training_curves.png")
    plt.close()


def plot_comparison(ddqn_results, fixed_results, random_results):
    """Plot comparison between DDQN and baseline policies"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    policies = ['DDQN', 'Fixed-Time', 'Random']
    waiting_times = [
        ddqn_results['avg_waiting_time'],
        fixed_results['avg_waiting_time'],
        random_results['avg_waiting_time']
    ]
    queues = [
        ddqn_results['avg_queue'],
        fixed_results['avg_queue'],
        random_results['avg_queue']
    ]
    
    # Waiting Time Comparison
    axes[0].bar(policies, waiting_times, color=['green', 'orange', 'red'])
    axes[0].set_ylabel('Average Waiting Time (s)')
    axes[0].set_title('Average Waiting Time Comparison')
    axes[0].grid(axis='y')
    
    # Queue Length Comparison
    axes[1].bar(policies, queues, color=['green', 'orange', 'red'])
    axes[1].set_ylabel('Average Queue Length')
    axes[1].set_title('Average Queue Length Comparison')
    axes[1].grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('results/single_agent/comparison.png', dpi=300)
    print("Comparison plot saved to results/single_agent/comparison.png")
    plt.close()
    
    # Calculate improvement
    improvement_waiting = ((fixed_results['avg_waiting_time'] - ddqn_results['avg_waiting_time']) 
                          / fixed_results['avg_waiting_time'] * 100)
    improvement_queue = ((fixed_results['avg_queue'] - ddqn_results['avg_queue']) 
                        / fixed_results['avg_queue'] * 100)
    
    print(f"\nDDQN Improvement over Fixed-Time:")
    print(f"  Waiting Time: {improvement_waiting:.1f}% reduction")
    print(f"  Queue Length: {improvement_queue:.1f}% reduction")
