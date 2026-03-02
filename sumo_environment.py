"""
SUMO Environment Wrapper for Traffic Light Control with DDQN
"""

import traci
import sumolib
import numpy as np
import os


class SumoEnvironment:
    def __init__(self, 
                 net_file='sumo_config/single_intersection/intersection.net.xml',
                 route_file='sumo_config/single_intersection/routes.rou.xml',
                 use_gui=False,
                 num_seconds=3600,
                 delta_time=5):
        """
        SUMO Environment for Traffic Light Control
        
        Args:
            net_file: Path to SUMO network file
            route_file: Path to SUMO route file
            use_gui: Whether to use SUMO GUI (True) or sumo-cmd (False)
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
        
        # Traffic light ID
        self.tls_id = "center"
        
        # Edge IDs for each direction
        self.edges = {
            'north': 'north_in',
            'south': 'south_in',
            'east': 'east_in',
            'west': 'west_in'
        }
        
        # Phase definitions
        self.phases = {
            0: 0,  # North-South green
            1: 1   # East-West green
        }
        
        self.current_phase = 0
        self.time_since_last_change = 0
        self.simulation_step = 0
        
        # Metrics tracking
        self.total_waiting_time = 0
        self.vehicles_passed = 0
    
    def start_sumo(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        self.sumo_cmd = [
            sumo_binary,
            "-c", "sumo_config/single_intersection/simulation.sumocfg",
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
        
        # Reset internal state
        self.current_phase = 0
        self.time_since_last_change = 0
        self.simulation_step = 0
        self.total_waiting_time = 0
        self.vehicles_passed = 0
        
        # Set initial traffic light phase
        traci.trafficlight.setPhase(self.tls_id, self.current_phase)
        
        # Run a few steps to initialize
        for _ in range(self.delta_time):
            traci.simulationStep()
        
        return self._get_state()
    
    def step(self, action):
        """
        Execute action and advance simulation
        
        Args:
            action: 0 (keep current phase) or 1 (switch phase)
        
        Returns:
            next_state: numpy array of state
            reward: float reward value
            done: boolean indicating episode end
            info: dict with additional metrics
        """
        # Apply action
        if action == 1 and self.time_since_last_change >= 5:
            # Switch phase
            self.current_phase = 1 - self.current_phase
            traci.trafficlight.setPhase(self.tls_id, self.phases[self.current_phase])
            self.time_since_last_change = 0
        
        # Simulate for delta_time seconds
        for _ in range(self.delta_time):
            traci.simulationStep()
            self.simulation_step += 1
        
        self.time_since_last_change += self.delta_time
        
        # Get new state
        next_state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self.simulation_step >= self.num_seconds
        
        # Additional info
        info = {
            'total_waiting_time': self.total_waiting_time,
            'vehicles_passed': self.vehicles_passed,
            'current_phase': self.current_phase
        }
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """
        Extract state from SUMO simulation
        
        Returns:
            numpy array: [queue_N, queue_S, queue_E, queue_W, phase, time_since_change]
        """
        # Get queue lengths (halting vehicles) for each edge
        queue_north = traci.edge.getLastStepHaltingNumber(self.edges['north'])
        queue_south = traci.edge.getLastStepHaltingNumber(self.edges['south'])
        queue_east = traci.edge.getLastStepHaltingNumber(self.edges['east'])
        queue_west = traci.edge.getLastStepHaltingNumber(self.edges['west'])
        
        # Cap queues at 20 for normalization
        queue_north = min(queue_north, 20)
        queue_south = min(queue_south, 20)
        queue_east = min(queue_east, 20)
        queue_west = min(queue_west, 20)
        
        state = np.array([
            queue_north,
            queue_south,
            queue_east,
            queue_west,
            self.current_phase,
            min(self.time_since_last_change, 60)
        ], dtype=np.float32)
        
        return state
    
    def _calculate_reward(self, action):
        """
        Calculate reward based on current traffic state
        
        Returns:
            float: reward value
        """
        # Get queue lengths
        total_queue = sum([
            traci.edge.getLastStepHaltingNumber(self.edges['north']),
            traci.edge.getLastStepHaltingNumber(self.edges['south']),
            traci.edge.getLastStepHaltingNumber(self.edges['east']),
            traci.edge.getLastStepHaltingNumber(self.edges['west'])
        ])
        
        # Get waiting times for all vehicles
        vehicle_ids = traci.vehicle.getIDList()
        total_waiting = sum([traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids])
        
        # Update cumulative metrics
        self.total_waiting_time += total_waiting
        
        # Reward calculation
        reward = -total_queue - 0.5 * total_waiting
        
        # Penalty for switching too quickly
        if action == 1 and self.time_since_last_change < 5:
            reward -= 10
        
        return reward
    
    def close(self):
        """Close SUMO simulation"""
        if self.sumo_running:
            traci.close()
            self.sumo_running = False
    
    def get_traffic_metrics(self):
        """Get detailed traffic metrics for evaluation"""
        vehicle_ids = traci.vehicle.getIDList()
        
        metrics = {
            'total_vehicles': len(vehicle_ids),
            'total_waiting_time': sum([traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids]),
            'average_speed': np.mean([traci.vehicle.getSpeed(vid) for vid in vehicle_ids]) if vehicle_ids else 0,
            'total_queue': sum([
                traci.edge.getLastStepHaltingNumber(edge) 
                for edge in self.edges.values()
            ])
        }
        
        return metrics
