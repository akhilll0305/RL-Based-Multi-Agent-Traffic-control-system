"""
Supervisor Agent â€” Local + Global Group Coordination
=====================================================
Step 1 (local):  24-dim input  (4 agents Ã— 6-dim each)
Step 2 (global): 28-dim input  (24 own + 4 from other supervisor)

The 4-dim cross-group summary:
  [avg_queue, max_queue, avg_waiting_time, boundary_queue]

New methods for Step 2:
  get_signals_with_global()  â€” forward pass with other group's summary
  store_with_global()        â€” store 28-dim experience in buffer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import os


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Neural Network
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class SupervisorNetwork(nn.Module):
    """
    3-layer MLP for the supervisor.

    Step 1 (local):  input_dim = 24  (4 agents Ã— 6-dim)
    Step 2 (global): input_dim = 28  (24 own + 4 from other supervisor)

    Output: 4 coordination signals via tanh â†’ range [-1, +1]
    """
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=4):
        super(SupervisorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))   # Signals in [-1, +1]
        return x


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Replay Buffer
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class SupervisorReplayBuffer:
    """
    Experience replay for the supervisor.
    Stores group-level transitions: (24-dim state, reward, 24-dim next_state, done)
    """
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def store(self, group_state, group_reward, next_group_state, done):
        self.buffer.append((
            np.array(group_state,      dtype=np.float32),
            float(group_reward),
            np.array(next_group_state, dtype=np.float32),
            float(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Supervisor Agent
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class SupervisorAgent:
    """
    Supervisor for one group of 4 intersections.

    Responsibilities:
      - Collect all local states â†’ build 24-dim group state
      - Output 4 coordination signals (one per agent)
      - Train on group average reward via TD learning
      - Update target network periodically

    Args:
        group_tls_ids      : list of 4 TLS IDs in this group, e.g. ['tls_1','tls_2','tls_3','tls_4']
        state_dim_per_agent: local state dimension per agent (default 6)
        hidden_dim         : hidden layer size (default 64)
        learning_rate      : Adam LR (default 0.001)
        gamma              : discount factor (default 0.95)
        buffer_capacity    : replay buffer size (default 10 000)
        batch_size         : training batch size (default 64)
    """

    def __init__(self,
                 group_tls_ids,
                 state_dim_per_agent=6,
                 hidden_dim=64,
                 learning_rate=0.001,
                 gamma=0.95,
                 buffer_capacity=10_000,
                 batch_size=64,
                 global_summary_dim=0):
        """
        Args:
            global_summary_dim: 0 for Step 1 (local only),
                                4 for Step 2 (+ cross-group summary)
        """
        self.group_tls_ids      = group_tls_ids
        self.group_size         = len(group_tls_ids)                           # 4
        self.local_input_dim    = self.group_size * state_dim_per_agent        # 24
        self.global_summary_dim = global_summary_dim                           # 0 or 4
        self.input_dim          = self.local_input_dim + global_summary_dim    # 24 or 28
        self.output_dim         = self.group_size                              # 4
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.train_step         = 0

        # â”€â”€ Device â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # â”€â”€ Networks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.online_net = SupervisorNetwork(self.input_dim, hidden_dim, self.output_dim).to(self.device)
        self.target_net = SupervisorNetwork(self.input_dim, hidden_dim, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # â”€â”€ Optimizer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # â”€â”€ Replay buffer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.memory = SupervisorReplayBuffer(capacity=buffer_capacity)

    # â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _build_group_state(self, local_states_dict):
        """
        Concatenate each agent's 6-dim local state â†’ 24-dim group state.
        Order follows self.group_tls_ids.
        """
        return np.concatenate([local_states_dict[tls] for tls in self.group_tls_ids],
                              dtype=np.float32)

    # â”€â”€ Core API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def get_signals(self, local_states_dict):
        """
        Step 1 forward pass â€” local only (24-dim).

        Args:
            local_states_dict: {tls_id: np.array shape (6,)}

        Returns:
            signals_dict: {tls_id: float in [-1, +1]}
        """
        group_state  = self._build_group_state(local_states_dict)
        state_tensor = torch.FloatTensor(group_state).unsqueeze(0).to(self.device)

        self.online_net.eval()
        with torch.no_grad():
            signals = self.online_net(state_tensor).squeeze(0).cpu().numpy()
        self.online_net.train()

        return {tls: float(signals[i]) for i, tls in enumerate(self.group_tls_ids)}

    def get_signals_with_global(self, local_states_dict, other_summary):
        """
        Step 2 forward pass â€” local (24-dim) + cross-group summary (4-dim) = 28-dim.

        Args:
            local_states_dict: {tls_id: np.array shape (6,)}
            other_summary    : np.array shape (4,)
                               [avg_queue, max_queue, avg_waiting_time, boundary_queue]
                               from the OTHER supervisor group

        Returns:
            signals_dict: {tls_id: float in [-1, +1]}
        """
        group_state   = self._build_group_state(local_states_dict)            # 24-dim
        global_state  = np.concatenate([group_state, other_summary.astype(np.float32)])  # 28-dim
        state_tensor  = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)

        self.online_net.eval()
        with torch.no_grad():
            signals = self.online_net(state_tensor).squeeze(0).cpu().numpy()
        self.online_net.train()

        return {tls: float(signals[i]) for i, tls in enumerate(self.group_tls_ids)}

    def store(self, local_states_dict, group_reward, next_local_states_dict, done):
        """
        Step 1: Store 24-dim group transition.

        Args:
            local_states_dict      : current  {tls_id: 6-dim state}
            group_reward           : scalar average reward of the group
            next_local_states_dict : next     {tls_id: 6-dim state}
            done                   : episode end flag
        """
        group_state      = self._build_group_state(local_states_dict)
        next_group_state = self._build_group_state(next_local_states_dict)
        self.memory.store(group_state, group_reward, next_group_state, done)

    def store_with_global(self, local_states_dict, other_summary,
                          group_reward,
                          next_local_states_dict, next_other_summary, done):
        """
        Step 2: Store 28-dim group transition (24 own + 4 cross-group summary).

        Args:
            local_states_dict      : current  {tls_id: 6-dim state}
            other_summary          : current  np.array shape (4,) from other supervisor
            group_reward           : scalar average reward of this group
            next_local_states_dict : next     {tls_id: 6-dim state}
            next_other_summary     : next     np.array shape (4,) from other supervisor
            done                   : episode end flag
        """
        group_state      = self._build_group_state(local_states_dict)
        next_group_state = self._build_group_state(next_local_states_dict)

        # Concatenate with cross-group summary â†’ 28-dim
        full_state      = np.concatenate([group_state,      other_summary.astype(np.float32)])
        next_full_state = np.concatenate([next_group_state, next_other_summary.astype(np.float32)])

        self.memory.store(full_state, group_reward, next_full_state, done)

    def train(self):
        """
        One gradient update for the supervisor.

        Training signal:
          current_signals = online_net(state)           # (batch, 4)
          td_target       = r + Î³ Ã— mean(target_net(s')) broadcast to (batch, 4)
          loss            = MSE(current_signals, td_target)

        Returns: loss value (float) or None if buffer too small.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states      = states.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Current predicted signals
        current_signals = self.online_net(states)                            # (B, 4)

        # TD target
        with torch.no_grad():
            next_signals = self.target_net(next_states)                      # (B, 4)
            # Scalar target per sample â†’ broadcast across 4 signal outputs
            td_target = rewards + (1 - dones) * self.gamma * next_signals.mean(dim=1, keepdim=True)
            td_target = td_target.expand_as(current_signals)                 # (B, 4)

        loss = F.mse_loss(current_signals, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step += 1
        return loss.item()

    def update_target_network(self):
        """Hard copy: online â†’ target."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # â”€â”€ Persistence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def save(self, path):
        """Save supervisor checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'train_step': self.train_step,
        }, path)
        print(f"  âœ“ Supervisor saved â†’ {path}")

    def load(self, path):
        """Load supervisor checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.train_step = ckpt.get('train_step', 0)
        print(f"  âœ“ Supervisor loaded â† {path}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Quick smoke-test (run this file directly to verify)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == '__main__':
    print("=== SupervisorAgent smoke test (local + global) ===\n")

    group_ids   = ['tls_1', 'tls_2', 'tls_3', 'tls_4']
    group_ids_b = ['tls_5', 'tls_6', 'tls_7', 'tls_8']

    # â”€â”€ Step 1 (local only, 24-dim) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("--- Step 1: Local supervisor (24-dim) ---")
    sup_local = SupervisorAgent(group_tls_ids=group_ids, global_summary_dim=0)
    fake_states = {tls: np.random.rand(6).astype(np.float32) for tls in group_ids}
    signals = sup_local.get_signals(fake_states)
    print("Local signals:")
    for tls, sig in signals.items():
        print(f"  {tls}: {sig:+.4f}")

    # â”€â”€ Step 2 (global, 28-dim) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- Step 2: Global supervisor (28-dim) ---")
    sup_global = SupervisorAgent(group_tls_ids=group_ids, global_summary_dim=4)

    # Fake cross-group summary from Group B
    fake_summary_b = np.array([8.0, 15.0, 48.3, 26.0], dtype=np.float32)
    # [avg_queue, max_queue, avg_waiting_time, boundary_queue]

    signals_global = sup_global.get_signals_with_global(fake_states, other_summary=fake_summary_b)
    print("Global signals (A sees B's summary):")
    for tls, sig in signals_global.items():
        print(f"  {tls}: {sig:+.4f}")

    # â”€â”€ Test store_with_global + train â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fake_next        = {tls: np.random.rand(6).astype(np.float32) for tls in group_ids}
    fake_next_summary = np.array([7.0, 12.0, 40.0, 20.0], dtype=np.float32)
    for _ in range(100):
        sup_global.store_with_global(
            fake_states, fake_summary_b,
            group_reward=-250.0,
            next_local_states_dict=fake_next,
            next_other_summary=fake_next_summary,
            done=False
        )
    loss = sup_global.train()
    print(f"\nTraining loss (28-dim): {loss:.6f}")

    # â”€â”€ Save / load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # sup_global.save('outputs/checkpoints/supervisor/test_global_supervisor.pth')
    # sup_global.load('outputs/checkpoints/supervisor/test_global_supervisor.pth')

    print("\nâœ… All checks passed! (local + global)")
