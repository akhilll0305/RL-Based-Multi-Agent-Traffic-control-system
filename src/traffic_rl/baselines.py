"""
Non-learning signal controllers used as baselines
=================================================
Both run on the same envs and metrics as the RL agents. Run the env with
``delta_time=1`` so the green times are honoured to the second (the RL agents
decide every 5 s). The env inserts the 3 s yellow on every switch, so a
controller asks for a switch ``green + yellow`` seconds after the previous one.

``webster_plan`` implements Webster (1958): C0 = (1.5 L + 5) / (1 - Y), and
effective green split in proportion to each phase's critical flow ratio y_i.
Each intersection gets its own isolated plan with zero offset (no coordination).
"""

import xml.etree.ElementTree as ET
from collections import defaultdict

import sumolib

from traffic_rl.envs.common import YELLOW_S, MIN_GREEN_S


class FixedTimeController:
    """Cycles NS green -> yellow -> EW green -> yellow with fixed green times per light.

    plans: {tls_id: (green_ns_s, green_ew_s)}. The env starts every light on NS green.
    """

    def __init__(self, plans, yellow_s=YELLOW_S):
        self.plans = plans
        self.yellow_s = yellow_s
        self.reset()

    def reset(self):
        self.phase = {tls: 0 for tls in self.plans}
        # Start at 0 so the first NS green lasts green + yellow: the env counts that first green
        # from 0 and refuses a switch before MIN_SWITCH_S, which would desynchronise the plan.
        self.clock = {tls: 0 for tls in self.plans}

    def act(self, delta_time=1):
        """Actions for the next decision, then advance the internal clocks by delta_time."""
        actions = {}
        for tls, greens in self.plans.items():
            if self.clock[tls] >= self.yellow_s + greens[self.phase[tls]]:
                actions[tls] = 1
                self.phase[tls] = 1 - self.phase[tls]
                self.clock[tls] = 0
            else:
                actions[tls] = 0
            self.clock[tls] += delta_time
        return actions


class MaxPressureController:
    """Max-pressure (Varaiya 2013) under the RL agents' rules: decide every ``delta_time`` s,
    switch only after the minimum green (the env forces one after the maximum green).

    Pressure of a phase = sum over its two approaches of (queue on the approach, entry backlog
    included) - (halting vehicles on the exit the through movement feeds). Switch when the red
    phase has the higher pressure. Call ``bind(env)`` once before ``reset``.
    """

    def bind(self, env, net_file):
        net = sumolib.net.readNet(net_file)
        self.env = env
        self.single = hasattr(env, 'tls_id')
        self.edges = {env.tls_id: env.edges} if self.single else env.edges
        self.exit_of = {}
        for dirs in self.edges.values():
            for e in dirs.values():
                straight = [to.getID() for to, conns in net.getEdge(e).getOutgoing().items()
                            if any(c.getDirection() == 's' for c in conns)]
                self.exit_of[e] = straight[0] if straight else None
        self.reset()
        return self

    def reset(self):
        self.phase = {tls: 0 for tls in self.edges}

    def _pressure(self, tls, dirs):
        import traci
        p = 0.0
        for d in dirs:
            e = self.edges[tls][d]
            out = self.exit_of[e]
            p += self.env._queue(e) - (traci.edge.getLastStepHaltingNumber(out) if out else 0)
        return p

    def act(self, delta_time=5):
        from traffic_rl.envs.common import MIN_SWITCH_S, switch_due
        actions = {}
        for tls in self.edges:
            phase = self.env.current_phase if self.single else self.env.current_phases[tls]
            since = self.env.time_since_last_change if self.single else self.env.time_since_last_change[tls]
            ns, ew = self._pressure(tls, ('north', 'south')), self._pressure(tls, ('east', 'west'))
            green, red = (ns, ew) if phase == 0 else (ew, ns)
            actions[tls] = int(red > green and since >= MIN_SWITCH_S)
            if switch_due(actions[tls], since):      # what the env will do this step
                self.phase[tls] = 1 - phase
        return actions


def route_flows(route_file):
    """[(veh/h, [edge ids])] for every <flow> in a route file."""
    root = ET.parse(route_file).getroot()
    routes = {r.get('id'): r.get('edges').split() for r in root.iter('route') if r.get('id')}
    out = []
    for f in root.iter('flow'):
        if f.get('probability') is not None:
            q = float(f.get('probability')) * 3600.0
        elif f.get('vehsPerHour') is not None:
            q = float(f.get('vehsPerHour'))
        elif f.get('period') is not None:
            q = 3600.0 / float(f.get('period'))
        else:
            q = float(f.get('number')) * 3600.0 / (float(f.get('end')) - float(f.get('begin')))
        edges = routes[f.get('route')] if f.get('route') else [f.get('from'), f.get('to')]
        out.append((q, edges))
    return out


def approach_flows(route_file, approach_edges, scale=1.0):
    """{tls: {direction: veh/h}} entering each approach, through-traffic included."""
    per_edge = defaultdict(float)
    for q, edges in route_flows(route_file):
        for e in edges:
            per_edge[e] += q * scale
    return {tls: {d: per_edge[e] for d, e in dirs.items()} for tls, dirs in approach_edges.items()}


def lane_counts(net_file, approach_edges):
    net = sumolib.net.readNet(net_file)
    return {tls: {d: net.getEdge(e).getLaneNumber() for d, e in dirs.items()}
            for tls, dirs in approach_edges.items()}


def critical_ratios(flows, lanes, sat_flow_per_lane):
    """(y_NS, y_EW): the higher flow ratio of the two approaches served by each phase."""
    y = {d: flows[d] / (sat_flow_per_lane * lanes[d]) for d in flows}
    return max(y['north'], y['south']), max(y['east'], y['west'])


def webster_plan(flows, lanes, sat_flow_per_lane, lost_per_phase_s=4.0,
                 yellow_s=YELLOW_S, min_green_s=MIN_GREEN_S, cycle_bounds=(30.0, 120.0)):
    """Returns (green_ns_s, green_ew_s, cycle_s, Y) for one isolated intersection."""
    y_ns, y_ew = critical_ratios(flows, lanes, sat_flow_per_lane)
    Y = y_ns + y_ew
    L = 2 * lost_per_phase_s
    if Y >= 0.95:            # oversaturated: Webster's formula breaks down, use the max cycle
        cycle = cycle_bounds[1]
    else:
        cycle = min(max((1.5 * L + 5.0) / (1.0 - Y), cycle_bounds[0]), cycle_bounds[1])
    effective = cycle - L
    g_ns, g_ew = (effective * y_ns / Y, effective * y_ew / Y) if Y > 0 else (effective / 2,) * 2
    # displayed green = effective green + lost time - yellow
    to_display = lost_per_phase_s - yellow_s
    return (max(round(g_ns + to_display), min_green_s),
            max(round(g_ew + to_display), min_green_s), cycle, Y)


def degree_of_saturation(flows, lanes, sat_flow_per_lane, green_ns, green_ew,
                         lost_per_phase_s=4.0, yellow_s=YELLOW_S):
    """v/c of the critical approach of each phase under a given plan: (x_NS, x_EW)."""
    cycle = green_ns + green_ew + 2 * yellow_s
    to_eff = yellow_s - lost_per_phase_s
    y_ns, y_ew = critical_ratios(flows, lanes, sat_flow_per_lane)
    return (y_ns * cycle / (green_ns + to_eff), y_ew * cycle / (green_ew + to_eff))
