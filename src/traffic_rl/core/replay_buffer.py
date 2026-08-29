"""
Experience Replay Buffer for DDQN
"""

import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=10000):
        """
        Experience Replay Buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def store(self, state, action, reward, next_state, done):
        """
        Store experience in buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample random batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones)
        """
        # Sample random batch
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)
