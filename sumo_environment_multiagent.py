"""
Multi-Agent SUMO Environment for 4 Intersections
Manages 4 independent traffic lights with optional cooperation
"""

import traci
import numpy as np
import os


class MultiAgentSumoEnvironment:
    def __init__(self, 
                 net_file='sumo_config/multi_intersection/multiagent.net.xml',
                 route_file='sumo_config/multi_intersection/multiagent.rou.xml',
                 use_gui=False,
                 num_seconds=3600,
                 delta_time=5,
                 cooperative=False):
        """
        Multi-Agent SUMO Environment for 4 Intersections
        
        Args:
            net_file: Path to SUMO network file
            route_file: Path to SUMO route file
            use_gui: Whether to use SUMO GUI
            num_seconds: Episode duration in seconds
            delta_time: Seconds between RL agent decisions
            cooperative: If True, agents share observations (neighbor queues)
        """
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.cooperative = cooperative
        self.sumo_cmd = None
        self.sumo_running = False
        
        #Traffic light IDs for 4 intersections
        self.tls_ids = ['tls_1', 'tls_2', 'tls_3', 'tls_4']
        
        # Edge IDs for each intersection
        # Intersection 1 (Top-Left)
        self.edges = {
            'tls_1': {
                'north': 'north_to_i1',
                'south': 'i3_to_i1',
                'east': 'i2_to_i1',
                'west': 'west_to_i1'
            },
            # Intersection 2 (Top-Right)
            'tls_2': {
                'north': 'north_to_i2',
                'south': 'i4_to_i2',
                'east': 'east_to_i2',
                'west': 'i1_to_i2'
            },
            # Intersection 3 (Bottom-Left)
            'tls_3': {
                'north': 'i1_to_i3',
                'south': 'south_to_i3',
                'east': 'i4_to_i3',
                'west': 'west_to_i3'
            },
            # Intersection 4 (Bottom-Right)
            'tls_4': {
                'north': 'i2_to_i4',
                'south': 'south_to_i4',
                'east': 'east_to_i4',
                'west': 'i3_to_i4'
            }
        }
        
        # Neighbor mapping for cooperation
        self.neighbors = {
            'tls_1': ['tls_2', 'tls_3'],  # Int 1 neighbors: Int 2 (east), Int 3 (south)
            'tls_2': ['tls_1', 'tls_4'],  # Int 2 neighbors: Int 1 (west), Int 4 (south)
            'tls_3': ['tls_1', 'tls_4'],  # Int 3 neighbors: Int 1 (north), Int 4 (east)
            'tls_4': ['tls_2', 'tls_3']   # Int 4 neighbors: Int 2 (north), Int 3 (west)
        }
        
        # Phase definitions (0: NS green, 1: EW green)
        self.phases = {0: 0, 1: 1}
        
        # State tracking for each intersection
        self.current_phases = {tls: 0 for tls in self.tls_ids}
        self.time_since_last_change = {tls: 0 for tls in self.tls_ids}
        self.simulation_step = 0
        
        # Metrics tracking
        self.total_waiting_time = {tls: 0 for tls in self.tls_ids}
        self.vehicles_passed = {tls: 0 for tls in self.tls_ids}
    
    def start_sumo(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        self.sumo_cmd = [
            sumo_binary,
            "-c", "sumo_config/multi_intersection/multiagent.sumocfg",
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1",
            "--no-warnings", "true"
        ]
        traci.start(self.sumo_cmd)
        self.sumo_running = True
    
    def reset(self):
        """Reset environment for new episode"""
        # Close existing SUMO instance if running
        if self.sumo_running:
            traci.close()
            self.sumo_running = False
        
        # Start new SUMO simulation
        self.start_sumo()
        
        # Reset internal state for all intersections
        self.current_phases = {tls: 0 for tls in self.tls_ids}
        self.time_since_last_change = {tls: 0 for tls in self.tls_ids}
        self.simulation_step = 0
        self.total_waiting_time = {tls: 0 for tls in self.tls_ids}
        self.vehicles_passed = {tls: 0 for tls in self.tls_ids}
        
        # Set initial traffic light phases for all intersections
        for tls in self.tls_ids:
            traci.trafficlight.setPhase(tls, 0)
        
        # Run a few steps to initialize
        for _ in range(self.delta_time):
            traci.simulationStep()
        
        # Return states for all 4 agents
        states = self._get_all_states()
        return states
    
    def step(self, actions):
        """
        Execute actions for all agents and advance simulation
        
        Args:
            actions: Dict {tls_id: action} where action is 0 (keep) or 1 (switch)
        
        Returns:
            next_states: Dict {tls_id: state}
            rewards: Dict {tls_id: reward}
            done: Boolean indicating episode end
            info: Dict with additional metrics
        """
        # Apply actions for each intersection
        for tls, action in actions.items():
            if action == 1 and self.time_since_last_change[tls] >= 5:
                # Switch phase
                self.current_phases[tls] = 1 - self.current_phases[tls]
                traci.trafficlight.setPhase(tls, self.phases[self.current_phases[tls]])
                self.time_since_last_change[tls] = 0
        
        # Simulate for delta_time seconds
        for _ in range(self.delta_time):
            traci.simulationStep()
            self.simulation_step += 1
        
        # Update time since last change for all intersections
        for tls in self.tls_ids:
            self.time_since_last_change[tls] += self.delta_time
        
        # Get new states for all agents
        next_states = self._get_all_states()
        
        # Calculate rewards for each agent
        rewards = self._calculate_all_rewards(actions)
        
        # Check if episode is done
        done = self.simulation_step >= self.num_seconds
        
        # Additional info
        info = self._get_metrics()
        
        return next_states, rewards, done, info
    
    def _get_state(self, tls_id):
        """Get state for a single intersection (6 features for independent, 6+ for cooperative)"""
        # Get queue lengths for this intersection
        queues = []
        for direction in ['north', 'south', 'east', 'west']:
            edge = self.edges[tls_id][direction]
            queue = traci.edge.getLastStepHaltingNumber(edge)
            queues.append(min(queue, 20))  # Cap at 20
        
        # Base state: [queue_n, queue_s, queue_e, queue_w, phase, time_since_change]
        state = queues + [
            self.current_phases[tls_id],
            min(self.time_since_last_change[tls_id], 60)
        ]
        
        # If cooperative, add neighbor queue information
        if self.cooperative:
            neighbor_queues = []
            for neighbor_tls in self.neighbors[tls_id]:
                # Get total queue at neighbor intersection
                neighbor_total_queue = sum([
                    traci.edge.getLastStepHaltingNumber(self.edges[neighbor_tls][direction])
                    for direction in ['north', 'south', 'east', 'west']
                ])
                neighbor_queues.append(min(neighbor_total_queue, 40))  # Cap at 40
            
            state.extend(neighbor_queues)
        
        return np.array(state, dtype=np.float32)
    
    def _get_all_states(self):
        """Get states for all 4 intersections"""
        return {tls: self._get_state(tls) for tls in self.tls_ids}
    
    def _calculate_reward(self, tls_id, action):
        """Calculate reward for a single intersection"""
        # Get queue lengths for this intersection
        total_queue = sum([
            traci.edge.getLastStepHaltingNumber(self.edges[tls_id][direction])
            for direction in ['north', 'south', 'east', 'west']
        ])
        
        # Get waiting times for vehicles at this intersection
        # (Approximate by summing waiting times of vehicles on incoming edges)
        total_waiting = 0
        for direction in ['north', 'south', 'east', 'west']:
            edge = self.edges[tls_id][direction]
            vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge)
            for vid in vehicles_on_edge:
                try:
                    total_waiting += traci.vehicle.getWaitingTime(vid)
                except:
                    pass
        
        # Update cumulative metrics
        self.total_waiting_time[tls_id] += total_waiting
        
        # Reward calculation
        reward = -total_queue - 0.5 * total_waiting
        
        # Penalty for switching too quickly
        if action == 1 and self.time_since_last_change[tls_id] < 5:
            reward -= 10
        
        return reward
    
    def _calculate_all_rewards(self, actions):
        """Calculate rewards for all intersections"""
        # Individual rewards
        individual_rewards = {
            tls: self._calculate_reward(tls, actions[tls]) 
            for tls in self.tls_ids
        }
        
        # If cooperative, use shared reward (average of all)
        if self.cooperative:
            avg_reward = sum(individual_rewards.values()) / len(individual_rewards)
            return {tls: avg_reward for tls in self.tls_ids}
        else:
            return individual_rewards
    
    def _get_metrics(self):
        """Get detailed metrics for evaluation"""
        vehicle_ids = traci.vehicle.getIDList()
        
        metrics = {
            'total_vehicles': len(vehicle_ids),
            'network_waiting_time': sum([traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids]),
            'avg_waiting_time': sum([traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids]) / max(len(vehicle_ids), 1),
            'per_intersection': {}
        }
        
        # Per-intersection metrics
        for tls in self.tls_ids:
            total_queue = sum([
                traci.edge.getLastStepHaltingNumber(self.edges[tls][direction])
                for direction in ['north', 'south', 'east', 'west']
            ])
            
            metrics['per_intersection'][tls] = {
                'queue': total_queue,
                'phase': self.current_phases[tls],
                'time_since_change': self.time_since_last_change[tls]
            }
        
        return metrics
    
    def close(self):
        """Close SUMO simulation"""
        if self.sumo_running:
            traci.close()
            self.sumo_running = False
    
    def get_state_dim(self):
        """Get state dimension (6 for independent, 8 for cooperative with 2 neighbors)"""
        if self.cooperative:
            return 6 + 2  # Base 6 + 2 neighbor queues
        return 6
    
    def get_action_dim(self):
        """Get action dimension (always 2: keep or switch)"""
        return 2
