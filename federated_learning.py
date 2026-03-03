"""
Federated Averaging (FedAvg) for Hierarchical Multi-Agent Traffic Control

Implements two levels of weight aggregation:
  1. Intra-zone FedAvg: Average weights of 4 local agents within each zone
  2. Inter-zone FedAvg: Average weights across zones via supervisor exchange

This creates a global shared model while allowing local adaptation.

Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
           from Decentralized Data" (2017)
"""

import copy
import torch
import numpy as np


def federated_average(agents, alpha=1.0):
    """
    Perform Federated Averaging (FedAvg) on a list of DDQN agents.
    Averages the online network weights and updates all agents.

    Args:
        agents: List of DDQNAgent instances (same architecture)
        alpha: Mixing factor (1.0 = full average, <1.0 = partial mix with local weights)

    Returns:
        avg_state_dict: The averaged model state dict
    """
    if not agents:
        return None

    # Get all state dicts
    state_dicts = [agent.online_network.state_dict() for agent in agents]

    # Average each parameter
    avg_state = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        avg_state[key] = stacked.mean(dim=0)

    # Apply averaged weights to all agents
    for agent in agents:
        if alpha >= 1.0:
            # Full replacement with global average
            agent.online_network.load_state_dict(avg_state)
        else:
            # Partial mix: (1-alpha)*local + alpha*global
            local_state = agent.online_network.state_dict()
            mixed = {}
            for key in avg_state:
                mixed[key] = (1 - alpha) * local_state[key].float() + alpha * avg_state[key]
            agent.online_network.load_state_dict(mixed)

        # Also update target network
        agent.update_target_network()

    return avg_state


def federated_average_supervisors(supervisor_a, supervisor_b, alpha=0.5):
    """
    Inter-zone FedAvg: Average weights between two supervisor agents.
    Uses a softer alpha (0.5) to preserve zone-specific knowledge.

    Args:
        supervisor_a: SupervisorAgent for Zone A
        supervisor_b: SupervisorAgent for Zone B
        alpha: Mixing factor (0.5 = equal blend)

    Returns:
        avg_state_dict: The averaged supervisor state dict
    """
    state_a = supervisor_a.online_network.state_dict()
    state_b = supervisor_b.online_network.state_dict()

    avg_state = {}
    for key in state_a:
        avg_state[key] = alpha * state_a[key].float() + (1 - alpha) * state_b[key].float()

    # Apply to both supervisors (each gets the blended model)
    supervisor_a.online_network.load_state_dict(avg_state)
    supervisor_b.online_network.load_state_dict(avg_state)

    supervisor_a.update_target_network()
    supervisor_b.update_target_network()

    return avg_state


class FederatedCoordinator:
    """
    Orchestrates the full federated learning process.

    Manages:
      - 8 local agents (4 per zone)
      - 2 supervisor agents
      - Intra-zone and inter-zone FedAvg schedules
    """

    def __init__(self,
                 local_agents,
                 supervisor_a,
                 supervisor_b,
                 zone_a_ids=None,
                 zone_b_ids=None,
                 intra_zone_interval=10,
                 inter_zone_interval=25,
                 intra_zone_alpha=0.8,
                 inter_zone_alpha=0.5):
        """
        Args:
            local_agents: Dict {tls_id: DDQNAgent} for all 8 intersections
            supervisor_a: SupervisorAgent for Zone A
            supervisor_b: SupervisorAgent for Zone B
            zone_a_ids: List of TLS IDs in Zone A
            zone_b_ids: List of TLS IDs in Zone B
            intra_zone_interval: FedAvg within zone every N episodes
            inter_zone_interval: FedAvg across zones every N episodes
            intra_zone_alpha: Mixing factor for intra-zone averaging
            inter_zone_alpha: Mixing factor for inter-zone averaging
        """
        self.local_agents = local_agents
        self.supervisor_a = supervisor_a
        self.supervisor_b = supervisor_b

        self.zone_a_ids = zone_a_ids or ['tls_1', 'tls_2', 'tls_3', 'tls_4']
        self.zone_b_ids = zone_b_ids or ['tls_5', 'tls_6', 'tls_7', 'tls_8']

        self.intra_zone_interval = intra_zone_interval
        self.inter_zone_interval = inter_zone_interval
        self.intra_zone_alpha = intra_zone_alpha
        self.inter_zone_alpha = inter_zone_alpha

        # Tracking
        self.episode_count = 0
        self.intra_zone_count = 0
        self.inter_zone_count = 0
        self.fedavg_history = []

    def maybe_aggregate(self, episode):
        """
        Check if FedAvg should be performed at this episode.
        Called at the end of each training episode.

        Args:
            episode: Current episode number

        Returns:
            str: Description of what was done, or None
        """
        self.episode_count = episode
        actions_taken = []

        # Intra-zone FedAvg (more frequent)
        if episode % self.intra_zone_interval == 0 and episode > 0:
            self._intra_zone_fedavg()
            self.intra_zone_count += 1
            actions_taken.append(f"Intra-zone FedAvg #{self.intra_zone_count}")

        # Inter-zone FedAvg (less frequent)
        if episode % self.inter_zone_interval == 0 and episode > 0:
            self._inter_zone_fedavg()
            self.inter_zone_count += 1
            actions_taken.append(f"Inter-zone FedAvg #{self.inter_zone_count}")

        if actions_taken:
            msg = " | ".join(actions_taken)
            self.fedavg_history.append({
                'episode': episode,
                'actions': actions_taken
            })
            return msg

        return None

    def _intra_zone_fedavg(self):
        """Average weights of local agents within each zone"""
        # Zone A
        zone_a_agents = [self.local_agents[tls] for tls in self.zone_a_ids]
        federated_average(zone_a_agents, alpha=self.intra_zone_alpha)

        # Zone B
        zone_b_agents = [self.local_agents[tls] for tls in self.zone_b_ids]
        federated_average(zone_b_agents, alpha=self.intra_zone_alpha)

    def _inter_zone_fedavg(self):
        """
        Average weights across zones:
          1. Average all 8 local agents together (soft mix)
          2. Average the two supervisor agents
        """
        # Cross-zone local agent averaging (softer alpha)
        all_agents = [self.local_agents[tls] for tls in self.zone_a_ids + self.zone_b_ids]
        federated_average(all_agents, alpha=self.inter_zone_alpha)

        # Supervisor averaging
        federated_average_supervisors(
            self.supervisor_a,
            self.supervisor_b,
            alpha=self.inter_zone_alpha
        )

    def get_stats(self):
        """Get FedAvg statistics"""
        return {
            'total_episodes': self.episode_count,
            'intra_zone_aggregations': self.intra_zone_count,
            'inter_zone_aggregations': self.inter_zone_count,
            'history': self.fedavg_history
        }
