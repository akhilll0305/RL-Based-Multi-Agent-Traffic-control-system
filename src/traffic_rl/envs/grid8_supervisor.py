"""
8-Intersection SUMO Environment with Local Supervisor Support
=============================================================
Modified from sumo_environment_8intersection.py:

Key changes:
  1. _get_state()          → 6-dim only (removes hardcoded neighbor queues)
  2. _get_all_local_states() → NEW: returns all agents' raw 6-dim states
  3. _build_enhanced_states() → NEW: appends supervisor signal → 7-dim per agent
  4. _calculate_all_rewards() → individual reward per agent (not group average)
  5. get_state_dim()          → returns 7

Layout (same physical network):
  Group A: tls_1, tls_2, tls_3, tls_4  (left 2×2)
  Group B: tls_5, tls_6, tls_7, tls_8  (right 2×2)
"""

import traci
import numpy as np
import os


class SupervisorSumoEnvironment:
    def __init__(self,
                 net_file='sumo_files/grid8/network.net.xml',
                 route_file='sumo_files/grid8/network.rou.xml',
                 use_gui=False,
                 num_seconds=3600,
                 delta_time=5):
        """
        8-Intersection environment designed to work with SupervisorAgent.

        Args:
            net_file   : SUMO network file
            route_file : SUMO route file
            use_gui    : show SUMO GUI
            num_seconds: episode duration in seconds
            delta_time : seconds between agent decisions
        """
        self.net_file    = net_file
        self.route_file  = route_file
        self.use_gui     = use_gui
        self.num_seconds = num_seconds
        self.delta_time  = delta_time
        self.sumo_running = False

        # ── TLS IDs ────────────────────────────────────────────────
        self.tls_ids = [
            'tls_1', 'tls_2', 'tls_3', 'tls_4',
            'tls_5', 'tls_6', 'tls_7', 'tls_8'
        ]

        # ── Groups ─────────────────────────────────────────────────
        self.group_a = ['tls_1', 'tls_2', 'tls_3', 'tls_4']
        self.group_b = ['tls_5', 'tls_6', 'tls_7', 'tls_8']
        self.groups  = {'group_a': self.group_a, 'group_b': self.group_b}

        self.tls_group = {}
        for tls in self.group_a:
            self.tls_group[tls] = 'group_a'
        for tls in self.group_b:
            self.tls_group[tls] = 'group_b'

        # ── Incoming edges per intersection ────────────────────────
        self.edges = {
            'tls_1': {'north': 'north_to_i1', 'south': 'i3_to_i1',
                      'east':  'i2_to_i1',    'west':  'west_to_i1'},
            'tls_2': {'north': 'north_to_i2', 'south': 'i4_to_i2',
                      'east':  'i5_to_i2',    'west':  'i1_to_i2'},
            'tls_3': {'north': 'i1_to_i3',    'south': 'south_to_i3',
                      'east':  'i4_to_i3',    'west':  'west_to_i3'},
            'tls_4': {'north': 'i2_to_i4',    'south': 'south_to_i4',
                      'east':  'i7_to_i4',    'west':  'i3_to_i4'},
            'tls_5': {'north': 'north_to_i5', 'south': 'i7_to_i5',
                      'east':  'i6_to_i5',    'west':  'i2_to_i5'},
            'tls_6': {'north': 'north_to_i6', 'south': 'i8_to_i6',
                      'east':  'east_to_i6',  'west':  'i5_to_i6'},
            'tls_7': {'north': 'i5_to_i7',    'south': 'south_to_i7',
                      'east':  'i8_to_i7',    'west':  'i4_to_i7'},
            'tls_8': {'north': 'i6_to_i8',    'south': 'south_to_i8',
                      'east':  'east_to_i8',  'west':  'i7_to_i8'},
        }

        # ── Phase definitions ──────────────────────────────────────
        self.phases = {0: 0, 1: 1}

        # ── Internal state tracking ────────────────────────────────
        self.current_phases         = {tls: 0   for tls in self.tls_ids}
        self.time_since_last_change = {tls: 0   for tls in self.tls_ids}
        self.simulation_step        = 0
    # ──────────────────────────────────────────────────────────────
    # SUMO lifecycle
    # ──────────────────────────────────────────────────────────────
    def start_sumo(self):
        """Start SUMO simulation with randomised traffic."""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", "sumo_files/grid8/network.sumocfg",
            "--no-step-log",         "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport",    "-1",
            "--no-warnings",         "true",
            "--random",             # randomise vehicle spawns each episode
        ]
        traci.start(sumo_cmd)
        self.sumo_running = True

    def reset(self):
        """Reset environment for a new episode. Returns local states (6-dim each)."""
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

        self.start_sumo()
        self.current_phases         = {tls: 0   for tls in self.tls_ids}
        self.time_since_last_change = {tls: 0   for tls in self.tls_ids}
        self.simulation_step        = 0
        for tls in self.tls_ids:
            traci.trafficlight.setPhase(tls, 0)
        for _ in range(self.delta_time):
            traci.simulationStep()

        return self._get_all_local_states()   # 6-dim per agent

    def close(self):
        """Close SUMO simulation."""
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

    # ──────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────
    def step(self, actions):
        """
        Execute actions for all 8 agents and advance simulation.

        Args:
            actions: {tls_id: 0 or 1}

        Returns:
            local_states : {tls_id: np.array shape (6,)}  ← raw 6-dim, no supervisor
            rewards      : {tls_id: float}                ← INDIVIDUAL reward
            done         : bool
            info         : dict with metrics
        """
        # Apply phase changes
        for tls, action in actions.items():
            if action == 1 and self.time_since_last_change[tls] >= 5:
                self.current_phases[tls] = 1 - self.current_phases[tls]
                traci.trafficlight.setPhase(tls, self.phases[self.current_phases[tls]])
                self.time_since_last_change[tls] = 0

        # Advance simulation
        for _ in range(self.delta_time):
            traci.simulationStep()
            self.simulation_step += 1

        for tls in self.tls_ids:
            self.time_since_last_change[tls] += self.delta_time

        local_states = self._get_all_local_states()
        rewards      = self._calculate_all_rewards(actions)
        done         = self.simulation_step >= self.num_seconds
        info         = self._get_metrics()

        return local_states, rewards, done, info

    # ──────────────────────────────────────────────────────────────
    # State methods
    # ──────────────────────────────────────────────────────────────
    def _get_local_state(self, tls_id):
        """
        Get LOCAL state for one intersection — 6 features only.
        No neighbor information (supervisor handles that).

        Returns: np.array shape (6,)
          [queue_N, queue_S, queue_E, queue_W, phase, time_since_change]
        """
        queues = []
        for direction in ['north', 'south', 'east', 'west']:
            edge  = self.edges[tls_id][direction]
            queue = traci.edge.getLastStepHaltingNumber(edge)
            queues.append(min(queue, 20))

        state = queues + [
            self.current_phases[tls_id],
            min(self.time_since_last_change[tls_id], 60)
        ]
        return np.array(state, dtype=np.float32)

    def _get_all_local_states(self):
        """
        Get 6-dim local states for ALL 8 agents.

        Returns: {tls_id: np.array shape (6,)}
        """
        return {tls: self._get_local_state(tls) for tls in self.tls_ids}

    def get_group_summary(self, local_states_dict, group_tls_ids, boundary_tls_ids):
        """
        Compute 4-value summary of one group for cross-supervisor communication.
        Args:
            local_states_dict : {tls_id: np.array shape (6,)}
            group_tls_ids     : list of tls_ids in the group
            boundary_tls_ids  : list of tls_ids in the group that face the OTHER group
                                (e.g. tls_2, tls_4 for Group A; tls_5, tls_7 for Group B)
        Returns:
            np.array shape (4,) containing:
               [avg_queue, max_queue, avg_waiting_time, boundary_queue]
        """
        # Queue info from local states (index 0-3 are the 4 direction queues)
        group_queues = [local_states_dict[tls][:4].sum() for tls in group_tls_ids]
        avg_queue    = np.mean(group_queues)
        max_queue    = np.max(group_queues)

        # Waiting time from SUMO (need to query directly)
        total_wait = 0.0
        vehicle_count = 0
        for tls in group_tls_ids:
            for direction in ['north', 'south', 'east', 'west']:
                edge = self.edges[tls][direction]
                for vid in traci.edge.getLastStepVehicleIDs(edge):
                    try:
                        total_wait += traci.vehicle.getWaitingTime(vid)
                        vehicle_count += 1
                    except Exception:
                        pass
        avg_wait_time = total_wait / max(vehicle_count, 1)

        # Boundary queue (only the agents facing the other group)
        boundary_queue = sum(local_states_dict[tls][:4].sum() for tls in boundary_tls_ids)

        return np.array([avg_queue, max_queue, avg_wait_time, boundary_queue], dtype=np.float32)

    def build_enhanced_states(self, local_states, signals_a, signals_b):
        """
        Append supervisor coordination signal to each agent's local state.
        Local state (6-dim) + supervisor signal (1-dim) = enhanced state (7-dim).

        Args:
            local_states : {tls_id: np.array shape (6,)}
            signals_a    : {tls_id: float} — Group A supervisor signals
            signals_b    : {tls_id: float} — Group B supervisor signals

        Returns:
            enhanced_states: {tls_id: np.array shape (7,)}
        """
        all_signals = {**signals_a, **signals_b}
        enhanced = {}
        for tls in self.tls_ids:
            signal  = np.array([all_signals[tls]], dtype=np.float32)
            enhanced[tls] = np.concatenate([local_states[tls], signal])
        return enhanced

    # ──────────────────────────────────────────────────────────────
    # Reward  (INDIVIDUAL — not group average)
    # ──────────────────────────────────────────────────────────────
    def _calculate_reward(self, tls_id, action):
        """
        Individual reward for one intersection.
        reward = -(queue) - 0.5 × (waiting_time) - 10 × quick_switch_penalty
        """
        total_queue = sum(
            traci.edge.getLastStepHaltingNumber(self.edges[tls_id][d])
            for d in ['north', 'south', 'east', 'west']
        )

        total_waiting = 0.0
        for direction in ['north', 'south', 'east', 'west']:
            edge = self.edges[tls_id][direction]
            for vid in traci.edge.getLastStepVehicleIDs(edge):
                try:
                    total_waiting += traci.vehicle.getWaitingTime(vid)
                except Exception:
                    pass

        reward = -total_queue - 0.5 * total_waiting

        # Penalise switching too quickly
        if action == 1 and self.time_since_last_change[tls_id] < 5:
            reward -= 10

        return reward

    def _calculate_all_rewards(self, actions):
        """
        Calculate INDIVIDUAL reward for each of the 8 agents.
        Each agent is rewarded only for its own intersection performance.

        Returns: {tls_id: float}
        """
        return {tls: self._calculate_reward(tls, actions[tls]) for tls in self.tls_ids}

    def get_group_avg_reward(self, rewards, group):
        """
        Utility: compute average reward of a group.
        Used by SupervisorAgent for its own training signal.

        Args:
            rewards: {tls_id: float}
            group  : list of tls_ids in the group

        Returns: float
        """
        return sum(rewards[tls] for tls in group) / len(group)

    # ──────────────────────────────────────────────────────────────
    # Metrics
    # ──────────────────────────────────────────────────────────────
    def _get_metrics(self):
        """Network-level and group-level metrics."""
        vehicle_ids = traci.vehicle.getIDList()
        total_wait  = sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids)

        metrics = {
            'total_vehicles'     : len(vehicle_ids),
            'network_waiting_time': total_wait,
            'avg_waiting_time'   : total_wait / max(len(vehicle_ids), 1),
            'per_intersection'   : {},
            'per_group'          : {},
        }

        for tls in self.tls_ids:
            tq = sum(
                traci.edge.getLastStepHaltingNumber(self.edges[tls][d])
                for d in ['north', 'south', 'east', 'west']
            )
            metrics['per_intersection'][tls] = {
                'queue'            : tq,
                'phase'            : self.current_phases[tls],
                'time_since_change': self.time_since_last_change[tls],
            }

        for group_name, members in self.groups.items():
            group_queue = sum(metrics['per_intersection'][m]['queue'] for m in members)
            metrics['per_group'][group_name] = {
                'total_queue': group_queue,
                'avg_queue'  : group_queue / len(members),
            }

        return metrics

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    def get_state_dim(self):
        """7-dim: 6 local features + 1 supervisor signal."""
        return 7

    def get_local_state_dim(self):
        """6-dim: local features only (input to supervisor)."""
        return 6

    def get_action_dim(self):
        """2 actions: keep (0) or switch (1)."""
        return 2
