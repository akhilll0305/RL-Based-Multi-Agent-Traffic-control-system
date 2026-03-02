"""
DDQN Agent for Traffic Light Control
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from network import DDQNNetwork
from replay_buffer import ReplayBuffer


class DDQNAgent:
    def __init__(self, 
                 state_dim=6, 
                 action_dim=2,
                 hidden_dim=128,
                 learning_rate=0.001,
                 gamma=0.95,
                 epsilon_start=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 batch_size=64,
                 buffer_capacity=10000):
        """
        DDQN Agent for Traffic Light Control
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            hidden_dim: Hidden layer size
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum exploration rate
            batch_size: Batch size for training
            buffer_capacity: Replay buffer capacity
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Device configuration - Auto-detect GPU or use CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"🚀 GPU ENABLED - Using: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.device = torch.device("cpu")
            print("⚠️  GPU NOT AVAILABLE - Using CPU (training will be slower)")
            print("   To enable GPU: Install PyTorch with CUDA support")
        
        # Initialize networks
        self.online_network = DDQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DDQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy weights from online to target network
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()  # Target network in eval mode
        
        # Verify device
        dummy = torch.zeros(1).to(self.device)
        print(f"✓ Model loaded on: {dummy.device}")
        del dummy
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        
        # Training metrics
        self.training_step = 0
        self.loss_history = []
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state (numpy array)
            training: Whether in training mode (epsilon-greedy) or evaluation (greedy)
        
        Returns:
            action: Selected action (0 or 1)
        """
        # Epsilon-greedy exploration during training
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def train(self):
        """
        Train the DDQN agent using experience replay
        
        Returns:
            loss: Training loss value (None if buffer not large enough)
        """
        # Check if buffer has enough samples
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.online_network(states).gather(1, actions)
        
        # DDQN Target Calculation
        with torch.no_grad():
            # Online network selects best next action
            next_actions = self.online_network(next_states).argmax(dim=1, keepdim=True)
            
            # Target network evaluates the selected action
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            
            # Calculate TD target
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Track metrics
        self.training_step += 1
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def update_target_network(self):
        """Copy weights from online network to target network"""
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation tradeoff"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        print(f"Model loaded from {filepath}")
