"""
Experiment Manager for DDQN Traffic Control
Helps organize and compare multiple training runs
"""

import os
import shutil
import json
from datetime import datetime
import pandas as pd


class ExperimentManager:
    """Manage multiple DDQN training experiments"""
    
    def __init__(self, experiments_dir='experiments'):
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
        self.log_file = os.path.join(experiments_dir, 'experiment_log.json')
        self.experiments = self._load_experiments()
    
    def _load_experiments(self):
        """Load experiment log"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_experiments(self):
        """Save experiment log"""
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def save_experiment(self, name, description, config, results=None):
        """
        Save a training experiment
        
        Args:
            name: Experiment name (e.g., 'baseline_500ep', 'reward_tuned_v1')
            description: What was changed
            config: Dict of hyperparameters
            results: Optional evaluation results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id = f"{name}_{timestamp}"
        exp_dir = os.path.join(self.experiments_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Copy model
        if os.path.exists('models/ddqn_traffic_final.pth'):
            shutil.copy('models/ddqn_traffic_final.pth', 
                       os.path.join(exp_dir, 'model.pth'))
        
        # Copy checkpoint 400 (often best performing)
        if os.path.exists('checkpoints/single_agent/ddqn_episode_400.pth'):
            shutil.copy('checkpoints/single_agent/ddqn_episode_400.pth',
                       os.path.join(exp_dir, 'checkpoint_400.pth'))
        
        # Copy training history
        if os.path.exists('results/single_agent/training_history.csv'):
            shutil.copy('results/single_agent/training_history.csv',
                       os.path.join(exp_dir, 'training_history.csv'))
        
        # Copy all checkpoints
        checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        for checkpoint in os.listdir('checkpoints/single_agent'):
            if checkpoint.endswith('.pth'):
                shutil.copy(os.path.join('checkpoints/single_agent', checkpoint),
                           os.path.join(checkpoint_dir, checkpoint))
        
        # Save metadata
        self.experiments[exp_id] = {
            'name': name,
            'timestamp': timestamp,
            'description': description,
            'config': config,
            'results': results,
            'directory': exp_dir
        }
        self._save_experiments()
        
        print(f"\n✓ Experiment '{name}' saved to: {exp_dir}")
        print(f"  - Model: {exp_id}/model.pth")
        print(f"  - All checkpoints: {exp_id}/checkpoints/")
        print(f"  - Training history: {exp_id}/training_history.csv")
        
        return exp_id
    
    def list_experiments(self):
        """List all saved experiments"""
        if not self.experiments:
            print("No experiments saved yet.")
            return
        
        print("\n" + "="*80)
        print("SAVED EXPERIMENTS")
        print("="*80)
        
        for exp_id, exp_data in self.experiments.items():
            print(f"\n{exp_id}")
            print(f"  Description: {exp_data['description']}")
            print(f"  Timestamp: {exp_data['timestamp']}")
            if exp_data.get('results'):
                results = exp_data['results']
                print(f"  Avg Reward: {results.get('avg_reward', 'N/A')}")
                print(f"  Avg Waiting Time: {results.get('avg_waiting_time', 'N/A')}")
                print(f"  Avg Queue: {results.get('avg_queue', 'N/A')}")
        
        print("\n" + "="*80)
    
    def compare_experiments(self, exp_ids):
        """
        Compare multiple experiments
        
        Args:
            exp_ids: List of experiment IDs to compare
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Experiment Comparison', fontsize=16)
        
        for exp_id in exp_ids:
            if exp_id not in self.experiments:
                print(f"Warning: {exp_id} not found")
                continue
            
            exp_dir = self.experiments[exp_id]['directory']
            history_file = os.path.join(exp_dir, 'training_history.csv')
            
            if not os.path.exists(history_file):
                print(f"Warning: No training history for {exp_id}")
                continue
            
            df = pd.read_csv(history_file)
            label = self.experiments[exp_id]['name']
            
            # Plot rewards
            axes[0, 0].plot(df['episode'], df['avg_reward_last_50'], label=label, alpha=0.7)
            axes[0, 0].set_title('Average Reward (Last 50 Episodes)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot waiting time
            axes[0, 1].plot(df['episode'], df['avg_waiting_time'], label=label, alpha=0.7)
            axes[0, 1].set_title('Average Waiting Time')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Waiting Time (s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot queue length
            axes[1, 0].plot(df['episode'], df['avg_queue'], label=label, alpha=0.7)
            axes[1, 0].set_title('Average Queue Length')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Queue Length')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot loss
            axes[1, 1].plot(df['episode'], df['avg_loss'], label=label, alpha=0.7)
            axes[1, 1].set_title('Average Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_file = os.path.join(self.experiments_dir, 'comparison.png')
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved to: {comparison_file}")
        plt.show()
    
    def get_best_experiment(self, metric='avg_reward'):
        """
        Find best experiment based on metric
        
        Args:
            metric: 'avg_reward', 'avg_waiting_time', or 'avg_queue'
        
        Returns:
            exp_id of best experiment
        """
        best_exp = None
        best_value = None
        
        for exp_id, exp_data in self.experiments.items():
            if not exp_data.get('results'):
                continue
            
            value = exp_data['results'].get(metric)
            if value is None:
                continue
            
            # For reward, higher is better; for waiting time and queue, lower is better
            if metric == 'avg_reward':
                if best_value is None or value > best_value:
                    best_value = value
                    best_exp = exp_id
            else:
                if best_value is None or value < best_value:
                    best_value = value
                    best_exp = exp_id
        
        if best_exp:
            print(f"\nBest experiment (by {metric}): {best_exp}")
            print(f"  Value: {best_value}")
            print(f"  Model: {self.experiments[best_exp]['directory']}/model.pth")
        
        return best_exp
    
    def load_experiment(self, exp_id):
        """
        Load a saved experiment for testing
        
        Args:
            exp_id: Experiment ID
        
        Returns:
            Path to model file
        """
        if exp_id not in self.experiments:
            print(f"Error: Experiment {exp_id} not found")
            return None
        
        model_path = os.path.join(self.experiments[exp_id]['directory'], 'model.pth')
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
        
        print(f"\n✓ Loaded experiment: {exp_id}")
        print(f"  Description: {self.experiments[exp_id]['description']}")
        print(f"  Model path: {model_path}")
        
        return model_path


# Example usage functions
def save_current_training(name, description, config):
    """Save the current training run as an experiment"""
    manager = ExperimentManager()
    exp_id = manager.save_experiment(name, description, config)
    return exp_id


def compare_all_experiments():
    """Compare all saved experiments"""
    manager = ExperimentManager()
    manager.list_experiments()
    
    exp_ids = list(manager.experiments.keys())
    if len(exp_ids) > 1:
        print("\nGenerating comparison plots...")
        manager.compare_experiments(exp_ids)


if __name__ == '__main__':
    # Example: Save current training as baseline
    config = {
        'episodes': 500,
        'learning_rate': 0.001,
        'gamma': 0.95,
        'hidden_dim': 128,
        'epsilon_decay': 0.995,
        'state_dim': 6,
        'reward_function': 'default'
    }
    
    save_current_training(
        name='baseline_500ep',
        description='Initial training run with default hyperparameters',
        config=config
    )
