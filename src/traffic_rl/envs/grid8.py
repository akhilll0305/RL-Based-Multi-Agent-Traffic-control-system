"""
8-Intersection Multi-Agent SUMO Environment with Grouped Cooperation
Two cooperative groups of 4 intersections each (no cross-group info sharing)

Layout:
  Group A: tls_1, tls_2, tls_3, tls_4
  Group B: tls_5, tls_6, tls_7, tls_8
  Border links: tls_2<->tls_5, tls_4<->tls_7 (traffic flows, no cooperative info)
"""

import traci
import numpy as np
import os


class EightIntersectionEnv:
    def __init__(self,
                 net_file='sumo_files/grid8/network.net.xml',
                 route_file='sumo_files/grid8/network.rou.xml',
                 use_gui=False,
                 num_seconds=3600,
                 delta_time=5):
        """
        8-Intersection environment with grouped cooperative control.

        Args:
            net_file: Path to SUMO network file
            route_file: Path to SUMO route file
            use_gui: Whether to use SUMO GUI
            num_seconds: Episode duration in seconds
            delta_time: Seconds between RL agent decisions
        """
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.sumo_running = False

        # All 8 traffic light IDs
        self.tls_ids = [
            'tls_1', 'tls_2', 'tls_3', 'tls_4',
            'tls_5', 'tls_6', 'tls_7', 'tls_8'
        ]

        # Group definitions
        self.group_a = ['tls_1', 'tls_2', 'tls_3', 'tls_4']
        self.group_b = ['tls_5', 'tls_6', 'tls_7', 'tls_8']

        self.groups = {
            'group_a': self.group_a,
            'group_b': self.group_b
        }

        # Which group each TLS belongs to
        self.tls_group = {}
        for tls in self.group_a:
            self.tls_group[tls] = 'group_a'
        for tls in self.group_b:
            self.tls_group[tls] = 'group_b'

        # Incoming edge IDs for each intersection(N, S, E, W)
        self.edges = {
            'tls_1': {
                'north': 'north_to_i1', 'south': 'i3_to_i1',
                'east': 'i2_to_i1',     'west': 'west_to_i1'
            },
            'tls_2': {
                'north': 'north_to_i2', 'south': 'i4_to_i2',
                'east': 'i5_to_i2',     'west': 'i1_to_i2'
            },
            'tls_3': {
                'north': 'i1_to_i3',    'south': 'south_to_i3',
                'east': 'i4_to_i3',     'west': 'west_to_i3'
            },
            'tls_4': {
                'north': 'i2_to_i4',    'south': 'south_to_i4',
                'east': 'i7_to_i4',     'west': 'i3_to_i4'
            },
            'tls_5': {
                'north': 'north_to_i5', 'south': 'i7_to_i5',
                'east': 'i6_to_i5',     'west': 'i2_to_i5'
            },
            'tls_6': {
                'north': 'north_to_i6', 'south': 'i8_to_i6',
                'east': 'east_to_i6',   'west': 'i5_to_i6'
            },
            'tls_7': {
                'north': 'i5_to_i7',    'south': 'south_to_i7',
                'east': 'i8_to_i7',     'west': 'i4_to_i7'
            },
            'tls_8': {
                'north': 'i6_to_i8',    'south': 'south_to_i8',
                'east': 'east_to_i8',   'west': 'i7_to_i8'
            }
        }

        # Neighbor mapping WITHIN groups only (cooperative info sharing)
        # Mirrors the original 4-agent cooperative setup
        self.neighbors = {
            # Group A (same topology as original 4-agent)
            'tls_1': ['tls_2', 'tls_3'],
            'tls_2': ['tls_1', 'tls_4'],
            'tls_3': ['tls_1', 'tls_4'],
            'tls_4': ['tls_2', 'tls_3'],
            # Group B (mirror topology)
            'tls_5': ['tls_6', 'tls_7'],
            'tls_6': ['tls_5', 'tls_8'],
            'tls_7': ['tls_5', 'tls_8'],
            'tls_8': ['tls_6', 'tls_7']
        }

        # Phase definitions
        self.phases = {0: 0, 1: 1}

        # Per-intersection state tracking
        self.current_phases = {tls: 0 for tls in self.tls_ids}
        self.time_since_last_change = {tls: 0 for tls in self.tls_ids}
        self.simulation_step = 0

        # Metrics
        self.total_waiting_time = {tls: 0 for tls in self.tls_ids}

    # ------------------------------------------------------------------
    # SUMO lifecycle
    # ------------------------------------------------------------------
    def start_sumo(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", "sumo_files/grid8/network.sumocfg",
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1",
            "--no-warnings", "true",
            "--random",  # Randomize vehicle spawns each episode for robust training
        ]
        traci.start(sumo_cmd)
        self.sumo_running = True

    def reset(self):
        """Reset environment for a new episode"""
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

        self.start_sumo()

        self.current_phases = {tls: 0 for tls in self.tls_ids}
        self.time_since_last_change = {tls: 0 for tls in self.tls_ids}
        self.simulation_step = 0
        self.total_waiting_time = {tls: 0 for tls in self.tls_ids}

        for tls in self.tls_ids:
            traci.trafficlight.setPhase(tls, 0)

        for _ in range(self.delta_time):
            traci.simulationStep()

        return self._get_all_states()

    def close(self):
        """Close SUMO simulation"""
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, actions):
        """
        Execute actions for all 8 agents and advance simulation.

        Args:
            actions: Dict {tls_id: 0 or 1}

        Returns:
            next_states, rewards, done, info
        """
        # Apply actions
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

        next_states = self._get_all_states()
        rewards = self._calculate_all_rewards(actions)
        done = self.simulation_step >= self.num_seconds
        info = self._get_metrics()

        return next_states, rewards, done, info

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _get_state(self, tls_id):
        """
        Get cooperative state for one intersection.
        8 features: 4 queues + phase + time_since_change + 2 neighbor total queues
        (neighbors are WITHIN same group only)
        """
        queues = []
        for direction in ['north', 'south', 'east', 'west']:
            edge = self.edges[tls_id][direction]
            queue = traci.edge.getLastStepHaltingNumber(edge)
            queues.append(min(queue, 20))

        state = queues + [
            self.current_phases[tls_id],
            min(self.time_since_last_change[tls_id], 60)
        ]

        # Add neighbor queue info (within group only)
        for neighbor_tls in self.neighbors[tls_id]:
            neighbor_total = sum(
                traci.edge.getLastStepHaltingNumber(self.edges[neighbor_tls][d])
                for d in ['north', 'south', 'east', 'west']
            )
            state.append(min(neighbor_total, 40))

        return np.array(state, dtype=np.float32)

    def _get_all_states(self):
        """Get states for all 8 intersections"""
        return {tls: self._get_state(tls) for tls in self.tls_ids}

    # ------------------------------------------------------------------
    # Reward (cooperative within groups)
    # ------------------------------------------------------------------
    def _calculate_reward(self, tls_id, action):
        """Calculate individual reward for one intersection"""
        total_queue = sum(
            traci.edge.getLastStepHaltingNumber(self.edges[tls_id][d])
            for d in ['north', 'south', 'east', 'west']
        )

        total_waiting = 0
        for direction in ['north', 'south', 'east', 'west']:
            edge = self.edges[tls_id][direction]
            for vid in traci.edge.getLastStepVehicleIDs(edge):
                try:
                    total_waiting += traci.vehicle.getWaitingTime(vid)
                except Exception:
                    pass

        self.total_waiting_time[tls_id] += total_waiting

        reward = -total_queue - 0.5 * total_waiting

        if action == 1 and self.time_since_last_change[tls_id] < 5:
            reward -= 10

        return reward

    def _calculate_all_rewards(self, actions):
        """
        Cooperative reward: each agent gets the average reward of its group.
        """
        individual = {
            tls: self._calculate_reward(tls, actions[tls])
            for tls in self.tls_ids
        }

        # Average within each group
        group_avg = {}
        for group_name, members in self.groups.items():
            avg = sum(individual[m] for m in members) / len(members)
            group_avg[group_name] = avg

        # Assign group-average reward to each agent
        rewards = {}
        for tls in self.tls_ids:
            rewards[tls] = group_avg[self.tls_group[tls]]

        return rewards

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _get_metrics(self):
        """Get network-level and group-level metrics"""
        vehicle_ids = traci.vehicle.getIDList()
        total_wait = sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids)

        metrics = {
            'total_vehicles': len(vehicle_ids),
            'network_waiting_time': total_wait,
            'avg_waiting_time': total_wait / max(len(vehicle_ids), 1),
            'per_intersection': {},
            'per_group': {}
        }

        for tls in self.tls_ids:
            tq = sum(
                traci.edge.getLastStepHaltingNumber(self.edges[tls][d])
                for d in ['north', 'south', 'east', 'west']
            )
            metrics['per_intersection'][tls] = {
                'queue': tq,
                'phase': self.current_phases[tls],
                'time_since_change': self.time_since_last_change[tls]
            }

        for group_name, members in self.groups.items():
            group_queue = sum(
                metrics['per_intersection'][m]['queue'] for m in members
            )
            metrics['per_group'][group_name] = {
                'total_queue': group_queue,
                'avg_queue': group_queue / len(members)
            }

        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_state_dim(self):
        """8 features: 6 local + 2 neighbor queues (cooperative within group)"""
        return 8

    def get_action_dim(self):
        """2 actions: keep or switch phase"""
        return 2
