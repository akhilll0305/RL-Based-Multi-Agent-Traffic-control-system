"""
Shared SUMO plumbing for every environment
==========================================
1. ``SignalController``: the net files carry a static 42/3/42/3 s program per
   light. ``traci.trafficlight.setPhase`` only moves that program's pointer, so
   on its own SUMO keeps cycling, a "keep" action does not hold the green, and a
   switch from EW back to NS skips the yellow. The controller holds whichever
   green the agent chose until the agent switches, and inserts a yellow on every
   switch.
2. ``EpisodeMetrics``: trip-based metrics from ``--tripinfo-output`` (travel
   time, time loss, accumulated waiting time over completed trips), throughput
   from real arrivals, and queue length averaged over every simulated second.
3. ``SumoEpisodeMixin``: launch / advance / finish / close shared by the envs.
"""

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import traci

# Phase indices of every tlLogic in sumo_files/*/*.net.xml
GREEN_NS, YELLOW_NS, GREEN_EW, YELLOW_EW = 0, 1, 2, 3
HOLD_S = 1e6          # "forever": SUMO never advances the program by itself
YELLOW_S = 3
MIN_GREEN_S = 10     # green must last this long before a switch request is honoured
MIN_SWITCH_S = YELLOW_S + MIN_GREEN_S   # seconds since the last switch (yellow included)
MAX_GREEN_S = 60     # a green that has lasted this long is ended even if the agent keeps it
MAX_SWITCH_S = YELLOW_S + MAX_GREEN_S
DIRECTIONS = ('north', 'south', 'east', 'west')


def switch_due(action, time_since_change):
    """Honour a switch request after the minimum green; force one after the maximum green.

    Without a maximum green an agent whose inputs stop changing can keep one green for the
    whole episode (seen on grid4: ~3300 s on one green, 500+ cars starved)."""
    return ((action == 1 and time_since_change >= MIN_SWITCH_S)
            or time_since_change >= MAX_SWITCH_S)


def queue_feature(q):
    """State encoding of a queue: log1p instead of a hard cap, so the input never saturates."""
    return float(np.log1p(q))


def time_feature(time_since_change):
    """Seconds since the last switch, scaled to [0, ~1]; it can't exceed the maximum green."""
    return min(time_since_change, MAX_SWITCH_S) / MAX_GREEN_S


class SignalController:
    """Phase bit 0 = NS green, 1 = EW green. Call ``tick`` after every simulationStep."""

    def __init__(self, tls_ids, yellow_s=YELLOW_S):
        self.tls_ids = list(tls_ids)
        self.yellow_s = yellow_s
        self.pending = {}            # tls -> [seconds of yellow left, target phase bit]

    def reset(self):
        self.pending = {}
        for tls in self.tls_ids:
            traci.trafficlight.setPhase(tls, GREEN_NS)
            traci.trafficlight.setPhaseDuration(tls, HOLD_S)

    def switch(self, tls, target):
        """Start the yellow that ends the current green; the target green follows it."""
        yellow = YELLOW_EW if target == 0 else YELLOW_NS
        traci.trafficlight.setPhase(tls, yellow)
        traci.trafficlight.setPhaseDuration(tls, HOLD_S)
        self.pending[tls] = [self.yellow_s, target]

    def tick(self):
        for tls in list(self.pending):
            self.pending[tls][0] -= 1
            if self.pending[tls][0] <= 0:
                target = self.pending.pop(tls)[1]
                traci.trafficlight.setPhase(tls, GREEN_NS if target == 0 else GREEN_EW)
                traci.trafficlight.setPhaseDuration(tls, HOLD_S)


class EpisodeMetrics:
    """Accumulates per-second statistics and parses tripinfo at the end of an episode."""

    def __init__(self, approach_edges):
        # approach_edges: {tls_id: [incoming edge ids]}
        self.approach_edges = approach_edges
        self.tls_ids = list(approach_edges)

    def start(self):
        self.arrived = 0
        self.seconds = 0
        self.queue_sum = {tls: 0.0 for tls in self.tls_ids}
        for edges in self.approach_edges.values():
            for e in edges:
                traci.edge.subscribe(e, [traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER])

    def tick(self):
        self.arrived += traci.simulation.getArrivedNumber()
        self.seconds += 1
        res = traci.edge.getAllSubscriptionResults()
        halt = traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER
        for tls, edges in self.approach_edges.items():
            self.queue_sum[tls] += sum(res[e][halt] for e in edges)

    def summary(self, tripinfo_path, pending_insertion):
        completed, unfinished = [], 0
        in_system = []               # every vehicle due before the end: completed, in network, or never inserted
        end = float(self.seconds)
        if os.path.exists(tripinfo_path):
            for t in ET.parse(tripinfo_path).getroot().iter('tripinfo'):
                depart, delay = float(t.get('depart', -1)), float(t.get('departDelay', 0))
                if depart < 0:                                   # never inserted
                    in_system.append(delay)
                elif t.get('vaporized') or float(t.get('arrival', -1)) < 0:
                    unfinished += 1
                    in_system.append(end - depart + delay)
                else:
                    completed.append((float(t.get('duration')), float(t.get('timeLoss')),
                                      float(t.get('waitingTime')), delay))
                    in_system.append(float(t.get('duration')) + delay)
        c = np.array(completed) if completed else np.full((1, 4), np.nan)
        secs = max(self.seconds, 1)
        per_tls_queue = {tls: self.queue_sum[tls] / secs for tls in self.tls_ids}
        return {
            'sim_seconds'         : self.seconds,
            'trips_completed'     : len(completed),
            'arrived'             : self.arrived,
            'throughput_veh_h'    : self.arrived * 3600.0 / secs,
            'mean_travel_time'    : float(np.mean(c[:, 0])),
            'mean_time_loss'      : float(np.mean(c[:, 1])),
            'mean_waiting_time'   : float(np.mean(c[:, 2])),
            'mean_depart_delay'   : float(np.mean(c[:, 3])),
            # fair across controllers: also counts vehicles held outside or still in the network
            'mean_time_in_system_all': float(np.mean(in_system)) if in_system else float('nan'),
            'vehicles_due'        : len(in_system),
            'unfinished'          : unfinished,
            'pending_insertion'   : pending_insertion,
            'mean_queue'          : float(np.mean(list(per_tls_queue.values()))),
            'per_intersection_queue': per_tls_queue,
        }


class SumoEpisodeMixin:
    """Expects the env to define ``sumocfg``, ``use_gui`` and ``tls_ids``.

    seed=None keeps the old behaviour (``--random``: a fresh random seed per episode).
    seed=s gives episode k the SUMO seed s + k, so a run is reproducible.
    """

    def _init_episode_plumbing(self, seed, approach_edges):
        """approach_edges: {tls_id: [incoming edge ids]} used for the queue metric."""
        self.seed = seed
        self.episode_index = 0
        self.episode_seed = None
        self.signals = SignalController(self.tls_ids)
        self.metrics = EpisodeMetrics(approach_edges)
        fd, self.tripinfo_path = tempfile.mkstemp(prefix=f'tripinfo_{os.getpid()}_', suffix='.xml')
        os.close(fd)
        self.last_episode = None
        self.episode_log = []            # one summary per finished episode
        self._episode_open = False

    def _sumo_command(self):
        cmd = [
            "sumo-gui" if self.use_gui else "sumo",
            "-c", self.sumocfg,
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1",
            "--no-warnings", "true",
            "--tripinfo-output", self.tripinfo_path,
            "--tripinfo-output.write-unfinished", "true",
            "--tripinfo-output.write-undeparted", "true",
        ]
        if self.use_gui:     # demo: start playing at once and close the window when the episode ends
            cmd += ["--start", "--quit-on-end"]
        if getattr(self, 'route_file', None):     # overrides the sumocfg's route file
            cmd += ["--route-files", self.route_file]
        if self.seed is None:
            self.episode_seed = None
            cmd.append("--random")
        else:
            self.episode_seed = self.seed + self.episode_index
            cmd += ["--seed", str(self.episode_seed)]
        return cmd

    def _start_episode(self):
        """Close any previous episode, launch SUMO, put every light on a held NS green."""
        self.finish_episode()
        traci.start(self._sumo_command())
        self.sumo_running = True
        self._episode_open = True
        self.episode_index += 1
        self.signals.reset()
        self.metrics.start()
        self._refresh_entry_backlog()

    def _advance(self, seconds):
        for _ in range(seconds):
            traci.simulationStep()
            self.signals.tick()
            self.metrics.tick()
        self._refresh_entry_backlog()

    def _refresh_entry_backlog(self):
        """Vehicles SUMO could not insert yet (entry edge full), counted per first edge."""
        self._entry_backlog = {}
        for vid in traci.simulation.getPendingVehicles():
            e = traci.vehicle.getRoute(vid)[0]
            self._entry_backlog[e] = self._entry_backlog.get(e, 0) + 1

    def _queue(self, edge):
        """Halting vehicles on an approach plus those still waiting to enter through it.

        Without the second term an agent can starve an entry approach: the cars it blocks are
        never inserted, so they appear in neither the queue nor the travel times."""
        return traci.edge.getLastStepHaltingNumber(edge) + self._entry_backlog.get(edge, 0)

    def finish_episode(self):
        """Close SUMO and return the episode's trip metrics (cached; safe to call twice)."""
        if self._episode_open:
            pending = len(traci.simulation.getPendingVehicles())
            traci.close()
            self.sumo_running = False
            self._episode_open = False
            self.last_episode = self.metrics.summary(self.tripinfo_path, pending)
            self.last_episode['episode_seed'] = self.episode_seed
            self.episode_log.append({k: v for k, v in self.last_episode.items()
                                     if k != 'per_intersection_queue'})
        elif getattr(self, 'sumo_running', False):
            traci.close()
            self.sumo_running = False
        return self.last_episode

    def close(self):
        self.finish_episode()

    @staticmethod
    def print_trip_summary(path):
        """Print the trip metrics of an episode log written by save_episode_log (the paper's metrics)."""
        import csv
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        print(f"\nTrip metrics over {len(rows)} episode(s) (per vehicle; the numbers in the paper tables):")
        for key, label in [('mean_time_in_system_all', 'Time in system, all vehicles (s)'),
                           ('mean_travel_time', 'Travel time, completed trips (s)'),
                           ('mean_waiting_time', 'Waiting time, completed trips (s)'),
                           ('throughput_veh_h', 'Throughput (veh/h)'),
                           ('mean_queue', 'Mean queue (veh per approach)'),
                           ('pending_insertion', 'Never entered the network (veh)')]:
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, '', 'nan')]
            if vals:
                print(f"  {label:34s}: {np.mean(vals):.2f}")
        seeds = [r.get('episode_seed') for r in rows]
        print(f"  SUMO seeds: {seeds[0]}..{seeds[-1]}")

    def save_episode_log(self, path):
        """Write one row of trip metrics per episode since the last save, then clear the log."""
        import csv
        self.finish_episode()
        if not self.episode_log:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['episode'] + list(self.episode_log[0]))
            w.writeheader()
            for i, row in enumerate(self.episode_log, 1):
                w.writerow({'episode': i, **row})
        self.episode_log = []
