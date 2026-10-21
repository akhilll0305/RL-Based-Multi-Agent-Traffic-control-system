"""
Supervisor Agent: discrete-option group coordination
====================================================
Each supervisor watches one group of 4 intersections and, every decision step,
picks one *option* per agent from

    0 balance        1 prioritise-NS        2 prioritise-EW        3 hold

The option is passed to that agent as the same single float the local agents
always received (``OPTION_SIGNALS``), so their 7-dim state is unchanged and
partial weight transfer from 6-dim checkpoints still works.

Input:  local    24-dim (4 agents x 6-dim)
        global   28-dim (24 own + 4-dim summary from the other supervisor:
                 [avg_queue, max_queue, avg_waiting_time, boundary_queue])

Learning: a factored Q-network with one linear (unbounded) head of 4 option
values per agent, i.e. a branching DQN (Tavakoli et al., AAAI 2018). Each head
is trained with the standard Double DQN backup on the shared group reward:

    a*_i  = argmax_o Q_online(s', i, o)
    y_i   = r + gamma * (1 - done) * Q_target(s', i, a*_i)
    loss  = mean_i Huber(Q_online(s, i, o_i), y_i)

This replaces the earlier tanh regression, whose outputs could not reach the
TD target and whose single broadcast target pushed all 4 signals together.
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OPTIONS = ('balance', 'prioritise_ns', 'prioritise_ew', 'hold')
# Categorical code fed to the local agent (7th state feature).
OPTION_SIGNALS = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)


class SupervisorNetwork(nn.Module):
    """MLP trunk + linear head: input -> (group_size, n_options) Q-values."""

    def __init__(self, input_dim=24, hidden_dim=64, group_size=4, n_options=len(OPTIONS)):
        super().__init__()
        self.group_size, self.n_options = group_size, n_options
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, group_size * n_options)
        for layer in (self.fc1, self.fc2, self.head):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x).view(-1, self.group_size, self.n_options)


class SupervisorReplayBuffer:
    """(group state, options per agent, group reward, next group state, done)."""

    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, options, reward, next_state, done):
        self.buffer.append((np.asarray(state, dtype=np.float32),
                            np.asarray(options, dtype=np.int64),
                            float(reward),
                            np.asarray(next_state, dtype=np.float32),
                            float(done)))

    def sample(self, batch_size):
        s, o, r, s2, d = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(np.array(s)), torch.LongTensor(np.array(o)),
                torch.FloatTensor(r).unsqueeze(1), torch.FloatTensor(np.array(s2)),
                torch.FloatTensor(d).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)


class SupervisorAgent:
    """
    Supervisor for one group of intersections.

    Args:
        group_tls_ids       : TLS IDs in this group, e.g. ['tls_1', ..., 'tls_4']
        state_dim_per_agent : local state dimension per agent (6)
        global_summary_dim  : 0 for the local supervisor, 4 for the global one
        reward_scale        : multiplies the group reward before the TD backup
                              (rewards are O(-100) per step at calibrated demand)
        epsilon_*           : per-agent epsilon-greedy over options, decayed
                              linearly over ``epsilon_decay_steps`` train steps
    """

    def __init__(self,
                 group_tls_ids,
                 state_dim_per_agent=6,
                 hidden_dim=64,
                 learning_rate=0.001,
                 gamma=0.95,
                 buffer_capacity=10_000,
                 batch_size=64,
                 global_summary_dim=0,
                 reward_scale=0.01,
                 epsilon_start=1.0,
                 epsilon_min=0.05,
                 epsilon_decay_steps=50_000):
        self.group_tls_ids = list(group_tls_ids)
        self.group_size = len(self.group_tls_ids)
        self.local_input_dim = self.group_size * state_dim_per_agent
        self.global_summary_dim = global_summary_dim
        self.input_dim = self.local_input_dim + global_summary_dim
        self.n_options = len(OPTIONS)
        self.gamma = gamma
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.epsilon_start, self.epsilon_min = epsilon_start, epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.train_step = 0
        self.last_options = None          # {tls: option} from the latest get_signals* call

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.online_net = SupervisorNetwork(self.input_dim, hidden_dim, self.group_size).to(self.device)
        self.target_net = SupervisorNetwork(self.input_dim, hidden_dim, self.group_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.memory = SupervisorReplayBuffer(capacity=buffer_capacity)

    # ── Inputs ────────────────────────────────────────────────────
    def _build_group_state(self, local_states_dict):
        return np.concatenate([local_states_dict[tls] for tls in self.group_tls_ids],
                              dtype=np.float32)

    def _full_state(self, local_states_dict, other_summary=None):
        s = self._build_group_state(local_states_dict)
        if self.global_summary_dim:
            s = np.concatenate([s, np.asarray(other_summary, dtype=np.float32)])
        return s

    # ── Acting ────────────────────────────────────────────────────
    def q_values(self, local_states_dict, other_summary=None):
        """(group_size, n_options) numpy array of option values."""
        x = torch.FloatTensor(self._full_state(local_states_dict, other_summary)).unsqueeze(0)
        with torch.no_grad():
            return self.online_net(x.to(self.device)).squeeze(0).cpu().numpy()

    def select_options(self, local_states_dict, other_summary=None, training=False):
        q = self.q_values(local_states_dict, other_summary)
        opts = q.argmax(axis=1)
        if training:
            explore = np.random.rand(self.group_size) < self.epsilon
            opts = np.where(explore, np.random.randint(self.n_options, size=self.group_size), opts)
        self.last_options = {tls: int(opts[i]) for i, tls in enumerate(self.group_tls_ids)}
        return self.last_options

    @staticmethod
    def options_to_signals(options):
        return {tls: float(OPTION_SIGNALS[o]) for tls, o in options.items()}

    def get_signals(self, local_states_dict, training=False):
        """Local supervisor: {tls: signal}. The chosen options are in ``last_options``."""
        return self.options_to_signals(self.select_options(local_states_dict, None, training))

    def get_signals_with_global(self, local_states_dict, other_summary, training=False):
        """Global supervisor: same as get_signals with the other group's 4-dim summary."""
        return self.options_to_signals(self.select_options(local_states_dict, other_summary, training))

    # ── Experience ────────────────────────────────────────────────
    def _options_array(self, options):
        return [options[tls] for tls in self.group_tls_ids]

    def store(self, local_states_dict, group_reward, next_local_states_dict, done, options):
        """Local supervisor transition. ``options``: {tls: option} taken in this state."""
        self.memory.store(self._full_state(local_states_dict), self._options_array(options),
                          group_reward, self._full_state(next_local_states_dict), done)

    def store_with_global(self, local_states_dict, other_summary, group_reward,
                          next_local_states_dict, next_other_summary, done, options):
        """Global supervisor transition (28-dim states)."""
        self.memory.store(self._full_state(local_states_dict, other_summary),
                          self._options_array(options), group_reward,
                          self._full_state(next_local_states_dict, next_other_summary), done)

    # ── Learning ──────────────────────────────────────────────────
    def train(self):
        """One Double DQN update over all heads. Returns the loss, or None while warming up."""
        if len(self.memory) < self.batch_size:
            return None
        s, o, r, s2, d = (t.to(self.device) for t in self.memory.sample(self.batch_size))
        q = self.online_net(s).gather(2, o.unsqueeze(2)).squeeze(2)              # (B, G)
        with torch.no_grad():
            best = self.online_net(s2).argmax(dim=2, keepdim=True)              # (B, G, 1)
            q_next = self.target_net(s2).gather(2, best).squeeze(2)             # (B, G)
            target = r * self.reward_scale + (1.0 - d) * self.gamma * q_next    # (B, G)
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.train_step += 1
        frac = min(self.train_step / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_min - self.epsilon_start)
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Persistence ───────────────────────────────────────────────
    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'format': 'discrete_option_v1',
                    'online_net': self.online_net.state_dict(),
                    'target_net': self.target_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'train_step': self.train_step,
                    'epsilon': self.epsilon}, path)
        print(f"  Supervisor saved -> {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if ckpt.get('format') != 'discrete_option_v1':
            raise ValueError(f"{path} is a checkpoint of the old tanh supervisor; retrain it.")
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.train_step = ckpt.get('train_step', 0)
        self.epsilon = ckpt.get('epsilon', self.epsilon_min)
        print(f"  Supervisor loaded <- {path}")


if __name__ == '__main__':
    # Smoke test: heads learn *different* options when the reward depends on each agent's option.
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    ids = ['tls_1', 'tls_2', 'tls_3', 'tls_4']
    sup = SupervisorAgent(ids, global_summary_dim=4, reward_scale=1.0, epsilon_decay_steps=2000)
    best = {'tls_1': 1, 'tls_2': 2, 'tls_3': 0, 'tls_4': 3}
    summ = np.zeros(4, dtype=np.float32)
    for step in range(3000):
        st = {t: np.random.rand(6).astype(np.float32) for t in ids}
        sup.get_signals_with_global(st, summ, training=True)
        opts = sup.last_options
        reward = sum(1.0 if opts[t] == best[t] else 0.0 for t in ids) / 4.0
        nxt = {t: np.random.rand(6).astype(np.float32) for t in ids}
        sup.store_with_global(st, summ, reward, nxt, summ, done=True, options=opts)
        loss = sup.train()
        if step % 500 == 0 and loss is not None:
            print(f"step {step:4d}  loss {loss:.4f}  eps {sup.epsilon:.2f}")
    sup.get_signals_with_global(st, summ)
    print("learned options:", sup.last_options, " target:", best)
