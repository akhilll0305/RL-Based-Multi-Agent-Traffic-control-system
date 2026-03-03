"""
Federated Hierarchical Multi-Agent SUMO Environment
8 Intersections in 4x2 Grid, split into 2 Zones with Supervisor Agents

Zone A (Supervisor 1): TLS 1, 2, 3, 4 (left 2x2 grid)
Zone B (Supervisor 2): TLS 5, 6, 7, 8 (right 2x2 grid)

Architecture:
  - 8 Local DDQN Agents (one per intersection)
  - 2 Supervisor Agents (one per zone)
  - FedAvg weight aggregation within zones and across supervisors
"""

import traci
import numpy as np
import os


class FederatedSumoEnvironment:
    def __init__(self,
                 net_file='sumo_config/federated/federated.net.xml',
                 route_file='sumo_config/federated/federated.rou.xml',
                 use_gui=False,
                 num_seconds=3600,
                 delta_time=5):
        """
        Federated Hierarchical Multi-Agent SUMO Environment

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
        self.sumo_cmd = None
        self.sumo_running = False

        # All 8 traffic light IDs
        self.tls_ids = [f'tls_{i}' for i in range(1, 9)]

        # Zone definitions
        self.zone_a = ['tls_1', 'tls_2', 'tls_3', 'tls_4']
        self.zone_b = ['tls_5', 'tls_6', 'tls_7', 'tls_8']
        self.zones = {'zone_a': self.zone_a, 'zone_b': self.zone_b}

        # Map each TLS to its zone
        self.tls_to_zone = {}
        for tls in self.zone_a:
            self.tls_to_zone[tls] = 'zone_a'
        for tls in self.zone_b:
            self.tls_to_zone[tls] = 'zone_b'

        # Incoming edge IDs for each intersection (based on the 4x2 grid)
        self.edges = {
            # === Zone A ===
            'tls_1': {
                'north': 'north_to_i1',
                'south': 'i3_to_i1',
                'east': 'i2_to_i1',
                'west': 'west_to_i1'
            },
            'tls_2': {
                'north': 'north_to_i2',
                'south': 'i4_to_i2',
                'east': 'i5_to_i2',    # Inter-zone bridge
                'west': 'i1_to_i2'
            },
            'tls_3': {
                'north': 'i1_to_i3',
                'south': 'south_to_i3',
                'east': 'i4_to_i3',
                'west': 'west_to_i3'
            },
            'tls_4': {
                'north': 'i2_to_i4',
                'south': 'south_to_i4',
                'east': 'i7_to_i4',    # Inter-zone bridge
                'west': 'i3_to_i4'
            },
            # === Zone B ===
            'tls_5': {
                'north': 'north_to_i5',
                'south': 'i7_to_i5',
                'east': 'i6_to_i5',
                'west': 'i2_to_i5'     # Inter-zone bridge
            },
            'tls_6': {
                'north': 'north_to_i6',
                'south': 'i8_to_i6',
                'east': 'east_to_i6',
                'west': 'i5_to_i6'
            },
            'tls_7': {
                'north': 'i5_to_i7',
                'south': 'south_to_i7',
                'east': 'i8_to_i7',
                'west': 'i4_to_i7'     # Inter-zone bridge
            },
            'tls_8': {
                'north': 'i6_to_i8',
                'south': 'south_to_i8',
                'east': 'east_to_i8',
                'west': 'i7_to_i8'
            }
        }

        # Neighbor mapping (including cross-zone neighbors)
        self.neighbors = {
            'tls_1': ['tls_2', 'tls_3'],
            'tls_2': ['tls_1', 'tls_4', 'tls_5'],   # tls_5 is cross-zone
            'tls_3': ['tls_1', 'tls_4'],
            'tls_4': ['tls_2', 'tls_3', 'tls_7'],   # tls_7 is cross-zone
            'tls_5': ['tls_2', 'tls_6', 'tls_7'],   # tls_2 is cross-zone
            'tls_6': ['tls_5', 'tls_8'],
            'tls_7': ['tls_4', 'tls_5', 'tls_8'],   # tls_4 is cross-zone
            'tls_8': ['tls_6', 'tls_7']
        }

        # Phase definitions (0: NS green, 1: EW green)
        self.phases = {0: 0, 1: 1}

        # State tracking
        self.current_phases = {tls: 0 for tls in self.tls_ids}
        self.time_since_last_change = {tls: 0 for tls in self.tls_ids}
        self.simulation_step = 0

        # Metrics
        self.total_waiting_time = {tls: 0 for tls in self.tls_ids}
        self.vehicles_passed = {tls: 0 for tls in self.tls_ids}

    # ==================== SUMO Control ====================

    def start_sumo(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        self.sumo_cmd = [
            sumo_binary,
            "-c", "sumo_config/federated/federated.sumocfg",
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1",
            "--no-warnings", "true"
        ]
        traci.start(self.sumo_cmd)
        self.sumo_running = True

    def reset(self):
        """Reset environment for new episode"""
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

        self.start_sumo()

        # Reset state
        self.current_phases = {tls: 0 for tls in self.tls_ids}
        self.time_since_last_change = {tls: 0 for tls in self.tls_ids}
        self.simulation_step = 0
        self.total_waiting_time = {tls: 0 for tls in self.tls_ids}
        self.vehicles_passed = {tls: 0 for tls in self.tls_ids}

        # Set initial phases
        for tls in self.tls_ids:
            traci.trafficlight.setPhase(tls, 0)

        # Warm-up steps
        for _ in range(self.delta_time):
            traci.simulationStep()

        return self._get_all_states()

    def step(self, actions):
        """
        Execute actions for all 8 agents and advance simulation

        Args:
            actions: Dict {tls_id: action} (0=keep, 1=switch)

        Returns:
            next_states: Dict {tls_id: state_array}
            rewards: Dict {tls_id: reward_value}
            done: Boolean
            info: Dict with metrics
        """
        # Apply actions
        for tls, action in actions.items():
            if action == 1 and self.time_since_last_change[tls] >= 5:
                self.current_phases[tls] = 1 - self.current_phases[tls]
                traci.trafficlight.setPhase(tls, self.phases[self.current_phases[tls]])
                self.time_since_last_change[tls] = 0

        # Simulate
        for _ in range(self.delta_time):
            traci.simulationStep()
            self.simulation_step += 1

        # Update timers
        for tls in self.tls_ids:
            self.time_since_last_change[tls] += self.delta_time

        next_states = self._get_all_states()
        rewards = self._calculate_all_rewards(actions)
        done = self.simulation_step >= self.num_seconds
        info = self._get_metrics()

        return next_states, rewards, done, info

    def close(self):
        """Close SUMO simulation"""
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

    # ==================== State Observation ====================

    def _get_queue(self, tls_id, direction):
        """Get queue length on a specific incoming edge"""
        edge = self.edges[tls_id][direction]
        return min(traci.edge.getLastStepHaltingNumber(edge), 20)

    def _get_state(self, tls_id):
        """
        Get local state for one intersection (8 features):
          [queue_N, queue_S, queue_E, queue_W, phase, time_since_change,
           zone_avg_queue, cross_zone_pressure]
        """
        queues = [self._get_queue(tls_id, d) for d in ['north', 'south', 'east', 'west']]
        local_total = sum(queues)

        # Zone average queue (avg total queue of all agents in same zone)
        zone = self.tls_to_zone[tls_id]
        zone_queues = []
        for z_tls in self.zones[zone]:
            z_total = sum(self._get_queue(z_tls, d) for d in ['north', 'south', 'east', 'west'])
            zone_queues.append(z_total)
        zone_avg = np.mean(zone_queues)

        # Cross-zone pressure: queue on inter-zone bridge edges
        # Positive = traffic flowing into our zone, Negative = flowing out
        cross_zone_pressure = self._get_cross_zone_pressure(tls_id)

        state = queues + [
            self.current_phases[tls_id],
            min(self.time_since_last_change[tls_id], 60),
            min(zone_avg, 40),
            cross_zone_pressure
        ]

        return np.array(state, dtype=np.float32)

    def _get_cross_zone_pressure(self, tls_id):
        """
        Calculate cross-zone traffic pressure for a specific intersection.
        Only meaningful for border intersections (tls_2, tls_4, tls_5, tls_7).
        For interior intersections, returns 0.
        """
        # Border intersections and their inter-zone bridge edges
        bridge_edges = {
            'tls_2': ('i5_to_i2', 'i2_to_i5'),  # inflow from zone B, outflow to zone B
            'tls_4': ('i7_to_i4', 'i4_to_i7'),
            'tls_5': ('i2_to_i5', 'i5_to_i2'),
            'tls_7': ('i4_to_i7', 'i7_to_i4'),
        }

        if tls_id not in bridge_edges:
            return 0.0

        inflow_edge, outflow_edge = bridge_edges[tls_id]
        inflow = traci.edge.getLastStepHaltingNumber(inflow_edge)
        outflow = traci.edge.getLastStepHaltingNumber(outflow_edge)

        # Positive = more traffic coming in from other zone
        return min(max(inflow - outflow, -10), 10)

    def _get_all_states(self):
        """Get states for all 8 intersections"""
        return {tls: self._get_state(tls) for tls in self.tls_ids}

    # ==================== Zone-Level Observations (for Supervisors) ====================

    def get_zone_state(self, zone_name):
        """
        Get aggregated zone-level state for supervisor agent.

        Returns 12 features:
          [avg_queue_N, avg_queue_S, avg_queue_E, avg_queue_W,
           total_vehicles_in_zone, avg_waiting_time_zone,
           phase_distribution (fraction in NS), avg_time_since_change,
           cross_zone_inflow, cross_zone_outflow,
           neighbor_zone_avg_queue, zone_throughput]
        """
        zone_tls = self.zones[zone_name]

        # Average queue per direction across zone intersections
        avg_queues = []
        for d in ['north', 'south', 'east', 'west']:
            q = np.mean([self._get_queue(tls, d) for tls in zone_tls])
            avg_queues.append(q)

        # Total vehicles on incoming edges in zone
        total_vehicles = 0
        for tls in zone_tls:
            for d in ['north', 'south', 'east', 'west']:
                edge = self.edges[tls][d]
                total_vehicles += traci.edge.getLastStepVehicleNumber(edge)

        # Average waiting time in zone
        zone_waiting = 0
        zone_veh_count = 0
        for tls in zone_tls:
            for d in ['north', 'south', 'east', 'west']:
                edge = self.edges[tls][d]
                for vid in traci.edge.getLastStepVehicleIDs(edge):
                    try:
                        zone_waiting += traci.vehicle.getWaitingTime(vid)
                        zone_veh_count += 1
                    except:
                        pass
        avg_waiting = zone_waiting / max(zone_veh_count, 1)

        # Phase distribution (fraction of intersections with NS green)
        ns_count = sum(1 for tls in zone_tls if self.current_phases[tls] == 0)
        phase_dist = ns_count / len(zone_tls)

        # Average time since last phase change
        avg_time_change = np.mean([self.time_since_last_change[tls] for tls in zone_tls])

        # Cross-zone flows
        if zone_name == 'zone_a':
            inflow_edges = ['i5_to_i2', 'i7_to_i4']
            outflow_edges = ['i2_to_i5', 'i4_to_i7']
        else:
            inflow_edges = ['i2_to_i5', 'i4_to_i7']
            outflow_edges = ['i5_to_i2', 'i7_to_i4']

        cross_inflow = sum(traci.edge.getLastStepVehicleNumber(e) for e in inflow_edges)
        cross_outflow = sum(traci.edge.getLastStepVehicleNumber(e) for e in outflow_edges)

        # Neighbor zone average queue
        other_zone = 'zone_b' if zone_name == 'zone_a' else 'zone_a'
        other_tls = self.zones[other_zone]
        neighbor_avg = np.mean([
            sum(self._get_queue(tls, d) for d in ['north', 'south', 'east', 'west'])
            for tls in other_tls
        ])

        # Zone throughput (vehicles that have passed through)
        throughput = 0
        for tls in zone_tls:
            for d in ['north', 'south', 'east', 'west']:
                edge = self.edges[tls][d]
                throughput += traci.edge.getLastStepVehicleNumber(edge)

        state = avg_queues + [
            min(total_vehicles, 100),
            min(avg_waiting, 100),
            phase_dist,
            min(avg_time_change, 60),
            min(cross_inflow, 20),
            min(cross_outflow, 20),
            min(neighbor_avg, 40),
            min(throughput, 100)
        ]

        return np.array(state, dtype=np.float32)

    # ==================== Reward Calculation ====================

    def _calculate_reward(self, tls_id, action):
        """Calculate reward for a single intersection"""
        total_queue = sum(
            self._get_queue(tls_id, d) for d in ['north', 'south', 'east', 'west']
        )

        # Waiting time penalty
        total_waiting = 0
        for d in ['north', 'south', 'east', 'west']:
            edge = self.edges[tls_id][d]
            for vid in traci.edge.getLastStepVehicleIDs(edge):
                try:
                    total_waiting += traci.vehicle.getWaitingTime(vid)
                except:
                    pass

        self.total_waiting_time[tls_id] += total_waiting

        # Base reward: minimize queue + waiting time
        reward = -total_queue - 0.5 * total_waiting

        # Penalty for switching too fast
        if action == 1 and self.time_since_last_change[tls_id] < 5:
            reward -= 10

        return reward

    def _calculate_all_rewards(self, actions):
        """
        Calculate hierarchical rewards:
          - 70% local reward (intersection-level)
          - 30% zone-level shared reward (zone average)
        """
        # Individual rewards
        individual = {
            tls: self._calculate_reward(tls, actions[tls])
            for tls in self.tls_ids
        }

        # Zone-level shared rewards
        zone_avgs = {}
        for zone_name, zone_tls in self.zones.items():
            zone_avgs[zone_name] = np.mean([individual[tls] for tls in zone_tls])

        # Blended reward: 70% local + 30% zone
        blended = {}
        for tls in self.tls_ids:
            zone = self.tls_to_zone[tls]
            blended[tls] = 0.7 * individual[tls] + 0.3 * zone_avgs[zone]

        return blended

    def get_zone_reward(self, zone_name):
        """
        Get zone-level reward for supervisor agent.
        Sum of all waiting time + queue across zone intersections.
        """
        zone_tls = self.zones[zone_name]

        total_queue = sum(
            sum(self._get_queue(tls, d) for d in ['north', 'south', 'east', 'west'])
            for tls in zone_tls
        )

        total_waiting = 0
        for tls in zone_tls:
            for d in ['north', 'south', 'east', 'west']:
                edge = self.edges[tls][d]
                for vid in traci.edge.getLastStepVehicleIDs(edge):
                    try:
                        total_waiting += traci.vehicle.getWaitingTime(vid)
                    except:
                        pass

        # Cross-zone balance bonus: reward if traffic is balanced
        if zone_name == 'zone_a':
            bridge_edges = ['i2_to_i5', 'i4_to_i7']
        else:
            bridge_edges = ['i5_to_i2', 'i7_to_i4']
        bridge_queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in bridge_edges)

        return -total_queue - 0.5 * total_waiting - 0.3 * bridge_queue

    # ==================== Metrics ====================

    def _get_metrics(self):
        """Get comprehensive metrics for evaluation"""
        vehicle_ids = traci.vehicle.getIDList()

        metrics = {
            'total_vehicles': len(vehicle_ids),
            'network_waiting_time': sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids),
            'avg_waiting_time': sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
                                / max(len(vehicle_ids), 1),
            'per_intersection': {},
            'per_zone': {}
        }

        # Per-intersection
        for tls in self.tls_ids:
            total_queue = sum(
                self._get_queue(tls, d) for d in ['north', 'south', 'east', 'west']
            )
            metrics['per_intersection'][tls] = {
                'queue': total_queue,
                'phase': self.current_phases[tls],
                'time_since_change': self.time_since_last_change[tls],
                'zone': self.tls_to_zone[tls]
            }

        # Per-zone
        for zone_name, zone_tls in self.zones.items():
            zone_queue = sum(
                metrics['per_intersection'][tls]['queue'] for tls in zone_tls
            )
            metrics['per_zone'][zone_name] = {
                'total_queue': zone_queue,
                'avg_queue': zone_queue / len(zone_tls),
                'intersections': zone_tls
            }

        return metrics

    # ==================== Dimension Getters ====================

    def get_local_state_dim(self):
        """Local agent state dimension (8 features)"""
        return 8  # 4 queues + phase + time + zone_avg + cross_zone_pressure

    def get_zone_state_dim(self):
        """Supervisor state dimension (12 features)"""
        return 12

    def get_action_dim(self):
        """Action dimension (keep or switch)"""
        return 2
