"""
Training Script for DDQN Traffic Light Control
"""

import numpy as np
import os
import csv
from tqdm import tqdm
from sumo_environment import SumoEnvironment
from agent import DDQNAgent


def train_ddqn(env, agent, num_episodes=1000, target_update_freq=10, save_freq=100):
    """
    Train DDQN agent
    
    Args:
        env: SumoEnvironment instance
        agent: DDQNAgent instance
        num_episodes: Number of training episodes
        target_update_freq: Frequency of target network updates (in episodes)
        save_freq: Frequency of model checkpointing (in episodes)
    
    Returns:
        training_history: Dict with training metrics
    """
    # Create directories
    os.makedirs('checkpoints/single_agent', exist_ok=True)
    os.makedirs('results/single_agent', exist_ok=True)
    
    # Training history
    training_history = {
        'episode_rewards': [],
        'episode_avg_waiting_time': [],
        'episode_avg_queue': [],
        'epsilon_history': [],
        'loss_history': []
    }
    
    print("Starting DDQN Training...")
    print(f"Total Episodes: {num_episodes}")
    print(f"Target Update Frequency: Every {target_update_freq} episodes")
    print("-" * 50)
    
    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        step_count = 0
        
        done = False
        while not done:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.memory.store(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state
            state = next_state
            episode_reward += reward
            step_count += 1
        
        # Update target network
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Get episode metrics
        metrics = env.get_traffic_metrics()
        avg_waiting_time = metrics['total_waiting_time'] / max(metrics['total_vehicles'], 1)
        avg_queue = metrics['total_queue'] / 4  # Average across 4 directions
        
        # Store metrics
        training_history['episode_rewards'].append(episode_reward)
        training_history['episode_avg_waiting_time'].append(avg_waiting_time)
        training_history['episode_avg_queue'].append(avg_queue)
        training_history['epsilon_history'].append(agent.epsilon)
        if episode_losses:
            training_history['loss_history'].append(np.mean(episode_losses))
        
        # Log progress
        if (episode + 1) % 50 == 0:
            recent_rewards = training_history['episode_rewards'][-50:]
            avg_reward = np.mean(recent_rewards)
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (last 50): {avg_reward:.2f}")
            print(f"  Episode Reward: {episode_reward:.2f}")
            print(f"  Avg Waiting Time: {avg_waiting_time:.2f}s")
            print(f"  Avg Queue Length: {avg_queue:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            if episode_losses:
                print(f"  Avg Loss: {np.mean(episode_losses):.4f}")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = f'checkpoints/single_agent/ddqn_episode_{episode + 1}.pth'
            agent.save(checkpoint_path)
    
    # Close environment
    env.close()
    
    # Save training history to CSV
    save_training_history(training_history)
    
    print("\nTraining Complete!")
    return training_history


def save_training_history(history):
    """Save training metrics to CSV"""
    csv_path = 'results/single_agent/training_history.csv'
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'Avg_Waiting_Time', 'Avg_Queue', 'Epsilon', 'Loss'])
        
        for i in range(len(history['episode_rewards'])):
            row = [
                i + 1,
                history['episode_rewards'][i],
                history['episode_avg_waiting_time'][i],
                history['episode_avg_queue'][i],
                history['epsilon_history'][i],
                history['loss_history'][i] if i < len(history['loss_history']) else ''
            ]
            writer.writerow(row)
    
    print(f"Training history saved to {csv_path}")
