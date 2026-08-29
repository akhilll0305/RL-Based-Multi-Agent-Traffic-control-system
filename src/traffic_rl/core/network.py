"""
Neural Network Architecture for DDQN Traffic Light Control
"""

import torch
import torch.nn as nn


class DDQNNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=128):
        """
        DDQN Network Architecture
        
        Args:
            state_dim: Dimension of state space (6)
            action_dim: Dimension of action space (2)
            hidden_dim: Number of neurons in hidden layers (128)
        """
        super(DDQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
        
        Returns:
            Q-values: Tensor of shape (batch_size, action_dim)
        """
        return self.network(state)
