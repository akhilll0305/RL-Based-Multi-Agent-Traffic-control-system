"""
Supervisor Agent for Federated Hierarchical Traffic Control

Each supervisor manages a zone of 4 intersections.
It takes zone-level aggregated state and outputs coordination signals
that modify the behavior of local agents in its zone.

Supervisor Action Space:
  0 = NS_PRIORITY  (prioritize North-South green phases)
  1 = EW_PRIORITY  (prioritize East-West green phases)
  2 = BALANCED     (no directional bias)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from network import DDQNNetwork
from replay_buffer import ReplayBuffer


class SupervisorNetwork(nn.Module):
    """
    Deeper network for supervisor agent (zone-level decisions).
    Input: 12-dim zone state + 12-dim neighbor zone state = 24 features
    Output: 3 actions (NS_priority, EW_priority, balanced)
    """
    def __init__(self, state_dim=24, action_dim=3, hidden_dim=256):
        super(SupervisorNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        return self.network(state)


class SupervisorAgent:
    """
    Supervisor DDQN Agent for zone-level coordination.

    Takes aggregated zone state (12 features from own zone + 12 from neighbor zone = 24)
    and outputs a coordination action that modifies local agent rewards.
    """

    # Action meanings
    NS_PRIORITY = 0
    EW_PRIORITY = 1
    BALANCED = 2

    def __init__(self,
                 zone_name,
                 state_dim=24,
                 action_dim=3,
                 hidden_dim=256,
                 learning_rate=0.0005,
                 gamma=0.95,
                 epsilon_start=1.0,
                 epsilon_decay=0.997,
                 epsilon_min=0.05,
                 batch_size=32,
                 buffer_capacity=5000,
                 decision_interval=3):
        """
        Args:
            zone_name: 'zone_a' or 'zone_b'
            state_dim: Combined state dimension (own zone + neighbor zone)
            action_dim: 3 (NS_priority, EW_priority, balanced)
            decision_interval: Supervisor decides every N local agent steps
        """
        self.zone_name = zone_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.decision_interval = decision_interval
        self.step_count = 0

        # Current coordination action
        self.current_action = self.BALANCED

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Networks
        self.online_network = SupervisorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = SupervisorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)

        # Metrics
        self.training_step = 0
        self.loss_history = []

    def should_decide(self):
        """Check if supervisor should make a new decision this step"""
        self.step_count += 1
        return self.step_count % self.decision_interval == 0

    def select_action(self, own_zone_state, neighbor_zone_state, training=True):
        """
        Select coordination action based on combined zone states.

        Args:
            own_zone_state: np.array of shape (12,) from get_zone_state()
            neighbor_zone_state: np.array of shape (12,) from get_zone_state()
            training: Whether to use epsilon-greedy

        Returns:
            action: 0 (NS_priority), 1 (EW_priority), or 2 (balanced)
        """
        # Combine own + neighbor zone states
        combined = np.concatenate([own_zone_state, neighbor_zone_state])

        if training and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(combined).unsqueeze(0).to(self.device)
                q_values = self.online_network(state_tensor)
                action = torch.argmax(q_values, dim=1).item()

        self.current_action = action
        return action

    def get_reward_modifier(self, local_tls_id, local_action, current_phase):
        """
        Get reward modifier for a local agent based on supervisor's coordination action.

        Args:
            local_tls_id: TLS ID of the local agent
            local_action: Action taken by local agent (0=keep, 1=switch)
            current_phase: Current phase (0=NS green, 1=EW green)

        Returns:
            modifier: Float bonus/penalty to add to local agent's reward
        """
        modifier = 0.0

        if self.current_action == self.NS_PRIORITY:
            # Reward keeping NS green, penalize switching away from NS
            if current_phase == 0 and local_action == 0:
                modifier = 2.0    # Good: keeping NS green during NS priority
            elif current_phase == 0 and local_action == 1:
                modifier = -1.0   # Bad: switching away from NS during NS priority
            elif current_phase == 1 and local_action == 1:
                modifier = 1.0    # Good: switching back to NS

        elif self.current_action == self.EW_PRIORITY:
            # Reward keeping EW green
            if current_phase == 1 and local_action == 0:
                modifier = 2.0
            elif current_phase == 1 and local_action == 1:
                modifier = -1.0
            elif current_phase == 0 and local_action == 1:
                modifier = 1.0

        # BALANCED: no modifier (modifier stays 0.0)
        return modifier

    def store_experience(self, own_state, neighbor_state, action, reward, 
                         next_own_state, next_neighbor_state, done):
        """Store supervisor experience in replay buffer"""
        combined_state = np.concatenate([own_state, neighbor_state])
        combined_next = np.concatenate([next_own_state, next_neighbor_state])
        self.memory.store(combined_state, action, reward, combined_next, done)

    def train(self):
        """Train supervisor DDQN"""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        current_q = self.online_network(states).gather(1, actions)

        # DDQN target
        with torch.no_grad():
            next_actions = self.online_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_step += 1
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        checkpoint = {
            'online_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'zone_name': self.zone_name
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_network.load_state_dict(checkpoint['online_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
