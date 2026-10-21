"""
SUMO Environment Wrapper for Traffic Light Control with DDQN
"""

import traci
import sumolib
import numpy as np
import os

from traffic_rl.envs.common import (SumoEpisodeMixin, DIRECTIONS, MIN_SWITCH_S, switch_due,
                                    queue_feature, time_feature)


class SumoEnvironment(SumoEpisodeMixin):
    def __init__(self, 
                 net_file='sumo_files/single/intersection.net.xml',
                 route_file='sumo_files/single/calibrated.rou.xml',
                 use_gui=False,
                 num_seconds=3600,
                 delta_time=5,
                 seed=None,
                 reward_mode='queue'):
        """
        SUMO Environment for Traffic Light Control
        
        Args:
            net_file: Path to SUMO network file
            route_file: Path to SUMO route file
            use_gui: Whether to use SUMO GUI (True) or sumo-cmd (False)
            num_seconds: Episode duration in seconds
            delta_time: Seconds between RL agent decisions
            seed: None = fresh random traffic each episode; int = reproducible
            reward_mode: 'queue' = -queue only (default; revision);
                         'original' = -queue - 0.5 * consecutive waiting time (NCA submission)
        """
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.sumo_cmd = None
        self.sumo_running = False
        self.sumocfg = "sumo_files/single/simulation.sumocfg"
        self.reward_mode = reward_mode

        # Traffic light ID
        self.tls_id = "center"
        self.tls_ids = [self.tls_id]
        
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
        self._init_episode_plumbing(
            seed, {self.tls_id: [self.edges[d] for d in DIRECTIONS]})

    def reset(self):
        """Reset environment for new episode (closes and summarises the previous one)"""
        self._start_episode()

        # Reset internal state
        self.current_phase = 0
        self.time_since_last_change = 0
        self.simulation_step = 0
        self.total_waiting_time = 0
        self.vehicles_passed = 0

        # Run a few steps to initialize (lights already held on NS green)
        self._advance(self.delta_time)

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
        if switch_due(action, self.time_since_last_change):
            # Switch phase (forced after the maximum green)
            self.current_phase = 1 - self.current_phase
            self.signals.switch(self.tls_id, self.current_phase)
            self.time_since_last_change = 0

        # Simulate for delta_time seconds
        self._advance(self.delta_time)
        self.simulation_step += self.delta_time

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
        queue_north = self._queue(self.edges['north'])
        queue_south = self._queue(self.edges['south'])
        queue_east = self._queue(self.edges['east'])
        queue_west = self._queue(self.edges['west'])
        
        # log1p, not a cap: a capped input stops changing once every queue is long
        state = np.array([
            queue_feature(queue_north),
            queue_feature(queue_south),
            queue_feature(queue_east),
            queue_feature(queue_west),
            self.current_phase,
            time_feature(self.time_since_last_change)
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
            self._queue(self.edges['north']),
            self._queue(self.edges['south']),
            self._queue(self.edges['east']),
            self._queue(self.edges['west'])
        ])
        
        # Reward calculation
        if self.reward_mode == 'queue':
            reward = -total_queue
        else:
            # Get waiting times for all vehicles (only needed by the original reward)
            vehicle_ids = traci.vehicle.getIDList()
            total_waiting = sum([traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids])
            self.total_waiting_time += total_waiting
            reward = -total_queue - 0.5 * total_waiting
        
        # Penalty for switching too quickly
        if action == 1 and self.time_since_last_change < MIN_SWITCH_S:
            reward -= 10
        
        return reward
    
    def get_traffic_metrics(self):
        """Get detailed traffic metrics for evaluation"""
        vehicle_ids = traci.vehicle.getIDList()
        
        metrics = {
            'total_vehicles': len(vehicle_ids),
            'total_waiting_time': sum([traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids]),
            'average_speed': np.mean([traci.vehicle.getSpeed(vid) for vid in vehicle_ids]) if vehicle_ids else 0,
            'total_queue': sum([
                self._queue(edge) 
                for edge in self.edges.values()
            ])
        }
        
        return metrics
