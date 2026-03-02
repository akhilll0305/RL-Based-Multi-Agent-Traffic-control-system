"""
Main Entry Point for DDQN Traffic Light Control System
"""

import os
import numpy as np
import torch
import argparse
from sumo_environment import SumoEnvironment
from agent import DDQNAgent
from train import train_ddqn
from evaluate import (evaluate_agent, evaluate_fixed_time, evaluate_random,
                      plot_training_curves, plot_comparison)
from generate_sumo_files import generate_all_sumo_files


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories():
    """Create necessary project directories"""
    directories = ['checkpoints/single_agent', 'models', 'results/single_agent', 'sumo_config/single_intersection', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Project directories created.")


def main(args):
    """Main execution function"""
    
    print("="*60)
    print("DDQN Traffic Light Control System with SUMO")
    print("="*60)
    
    # Setup
    set_seed(args.seed)
    create_directories()
    
    # Generate SUMO files if they don't exist
    if not os.path.exists('sumo_config/single_intersection/intersection.net.xml'):
        print("\nGenerating SUMO configuration files...")
        generate_all_sumo_files()
    
    # Hyperparameters
    STATE_DIM = 6
    ACTION_DIM = 2
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.001
    GAMMA = 0.95
    EPSILON_START = 1.0
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.01
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 10000
    TARGET_UPDATE_FREQ = 10
    
    if args.mode == 'train' or args.mode == 'all':
        print("\n" + "="*60)
        print("TRAINING MODE")
        print("="*60)
        
        # Create environment (no GUI during training)
        env = SumoEnvironment(
            net_file='sumo_config/single_intersection/intersection.net.xml',
            route_file='sumo_config/single_intersection/routes.rou.xml',
            use_gui=False,
            num_seconds=3600,
            delta_time=5
        )
        
        # Create agent
        agent = DDQNAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_decay=EPSILON_DECAY,
            epsilon_min=EPSILON_MIN,
            batch_size=BATCH_SIZE,
            buffer_capacity=BUFFER_CAPACITY
        )
        
        # Train
        training_history = train_ddqn(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            target_update_freq=TARGET_UPDATE_FREQ,
            save_freq=100
        )
        
        # Save final model
        agent.save('models/ddqn_traffic_final.pth')
        print("\nFinal model saved to models/ddqn_traffic_final.pth")
        
        # Plot training curves
        plot_training_curves('results/single_agent/training_history.csv')
    
    if args.mode == 'evaluate' or args.mode == 'all':
        print("\n" + "="*60)
        print("EVALUATION MODE")
        print("="*60)
        
        # Create environment
        env = SumoEnvironment(
            net_file='sumo_config/single_intersection/intersection.net.xml',
            route_file='sumo_config/single_intersection/routes.rou.xml',
            use_gui=args.gui,
            num_seconds=3600,
            delta_time=5
        )
        
        # Load trained agent
        agent = DDQNAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM
        )
        
        if os.path.exists(args.model_path):
            agent.load(args.model_path)
        else:
            print(f"Model not found at {args.model_path}. Using untrained agent.")
        
        # Evaluate DDQN agent
        ddqn_results = evaluate_agent(env, agent, num_episodes=args.eval_episodes, render=args.gui)
        
        # Evaluate baselines
        print("\n" + "-"*60)
        fixed_results = evaluate_fixed_time(env, num_episodes=args.eval_episodes, green_duration=30)
        
        print("\n" + "-"*60)
        random_results = evaluate_random(env, num_episodes=args.eval_episodes)
        
        # Plot comparison
        plot_comparison(ddqn_results, fixed_results, random_results)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"Results saved in results/ directory")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDQN Traffic Light Control with SUMO')
    
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['train', 'evaluate', 'all'],
                       help='Mode: train, evaluate, or all')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes (default: 500)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--model-path', type=str, default='models/ddqn_traffic_final.pth',
                       help='Path to load/save model')
    parser.add_argument('--gui', action='store_true',
                       help='Use SUMO GUI during evaluation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    main(args)
