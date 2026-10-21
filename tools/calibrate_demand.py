"""Calibrate per-intersection demand to a target degree of saturation (v/c).

Every flow in a network's route file is multiplied by one factor, so the OD
pattern is unchanged and only the load rises. For each candidate factor the
script reports the critical-approach v/c under the fixed-time plan (and under a
Webster plan), and simulates the fixed-time controller to confirm the network
stays stable (bounded backlog, trips complete).

    python tools/calibrate_demand.py --sat-flow            # measure s (veh/h/lane) once
    python tools/calibrate_demand.py --net grid8 --scales 1,2,2.5,3 --seeds 3
    python tools/calibrate_demand.py --write grid8=2.5     # write the calibrated route file
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from traffic_rl.paths import bootstrap  # noqa: E402
bootstrap()

import numpy as np  # noqa: E402
from traffic_rl import baselines as bl  # noqa: E402

NETS = {
    'single': ('traffic_rl.envs.single', 'SumoEnvironment',
               'sumo_files/single/intersection.net.xml', 'sumo_files/single/routes.rou.xml'),
    'grid4': ('traffic_rl.envs.grid4', 'MultiAgentSumoEnvironment',
              'sumo_files/grid4/multiagent.net.xml', 'sumo_files/grid4/multiagent.rou.xml'),
    'grid8': ('traffic_rl.envs.grid8', 'EightIntersectionEnv',
              'sumo_files/grid8/network.net.xml', 'sumo_files/grid8/network.rou.xml'),
}
CALIBRATED_NAME = 'calibrated.rou.xml'
SCRATCH = ROOT / 'outputs' / 'calibration'


def make_env(net, route_file, seed, num_seconds=3600, delta_time=1):
    mod, cls, _, _ = NETS[net]
    env_cls = getattr(__import__(mod, fromlist=[cls]), cls)
    kw = {'route_file': route_file} if route_file else {}     # None = the env's default
    return env_cls(num_seconds=num_seconds, delta_time=delta_time, seed=seed, **kw)


def approach_edges(env):
    if hasattr(env, 'tls_id'):                       # single intersection
        return {env.tls_id: dict(env.edges)}
    return {tls: dict(env.edges[tls]) for tls in env.tls_ids}


def write_scaled_routes(net, scale, out_path):
    """Copy the original route file with every flow probability multiplied by scale."""
    src = NETS[net][3]
    tree = ET.parse(src)
    for f in tree.getroot().iter('flow'):
        p = float(f.get('probability')) * scale
        if p > 1.0:
            raise ValueError(f'{net}: flow {f.get("id")} would need probability {p:.2f} > 1')
        f.set('probability', f'{p:.5f}')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding='UTF-8', xml_declaration=True)
    return str(out_path)


def run_episode(env, controller):
    env.reset()
    controller.reset()
    done = False
    single = hasattr(env, 'tls_id')
    while not done:
        acts = controller.act(env.delta_time)
        done = env.step(acts[env.tls_id] if single else acts)[2]
        env_phase = {env.tls_id: env.current_phase} if single else env.current_phases
        assert env_phase == controller.phase, f'env refused a planned switch: {env_phase} vs {controller.phase}'
    return env.finish_episode()


def measure_sat_flow(_=None):
    """Saturation flow (veh/h/lane): queue the NS approaches on red, then count discharge."""
    import traci
    route = write_scaled_routes('single', 3.0, SCRATCH / 'satflow.rou.xml')
    env = make_env('single', route, seed=1, num_seconds=400)
    env.reset()
    for _ in range(150):                 # EW green: NS queues build
        env.step(0) if env.current_phase == 1 else env.step(1)
    env.step(1)                          # switch to NS (3 s yellow, then green)
    env.step(0); env.step(0)             # yellow + green start
    ns = [env.edges['north'], env.edges['south']]
    lanes = sum(traci.edge.getLaneNumber(e) for e in ns)
    on_edge = set(v for e in ns for v in traci.edge.getLastStepVehicleIDs(e))
    left, secs = 0, 0
    for _ in range(4):                   # skip start-up lost time
        env.step(0)
        on_edge = set(v for e in ns for v in traci.edge.getLastStepVehicleIDs(e))
    for _ in range(25):
        env.step(0); secs += 1
        now = set(v for e in ns for v in traci.edge.getLastStepVehicleIDs(e))
        left += len(on_edge - now)
        on_edge = now
    queue_left = sum(traci.edge.getLastStepHaltingNumber(e) for e in ns)
    env.close()
    return {'sat_flow_veh_h_lane': round(left * 3600.0 / (secs * lanes)),
            'vehicles_counted': left, 'lanes': lanes, 'queue_still_waiting': queue_left}


def plans_for(net, scale, sat_flow, kind):
    _, _, net_file, route_file = NETS[net]
    env = make_env(net, route_file, seed=0)
    edges = approach_edges(env)
    flows = bl.approach_flows(route_file, edges, scale)
    lanes = bl.lane_counts(net_file, edges)
    plans, xs = {}, {}
    for tls in edges:
        if kind == 'webster':
            g_ns, g_ew, _, _ = bl.webster_plan(flows[tls], lanes[tls], sat_flow)
        else:
            g_ns, g_ew = 30, 30
        plans[tls] = (g_ns, g_ew)
        xs[tls] = max(bl.degree_of_saturation(flows[tls], lanes[tls], sat_flow, g_ns, g_ew))
    return plans, xs, flows


def _sim(job):
    net, scale, seed, plans = job
    route = write_scaled_routes(net, scale, SCRATCH / f'{net}_x{scale}_s{seed}.rou.xml')
    env = make_env(net, route, seed=1000 + seed)
    m = run_episode(env, bl.FixedTimeController(plans))
    m.pop('per_intersection_queue')
    return net, scale, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sat-flow', action='store_true')
    p.add_argument('--sat-flow-value', type=float, default=None,
                   help='veh/h/lane; measure it first with --sat-flow')
    p.add_argument('--net', choices=list(NETS), action='append')
    p.add_argument('--scales', default='1')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--write', action='append', default=[], help='net=scale')
    args = p.parse_args()

    if args.sat_flow:
        print(json.dumps(measure_sat_flow()))
        return
    for spec in args.write:
        net, scale = spec.split('=')
        out = Path(NETS[net][3]).parent / CALIBRATED_NAME
        write_scaled_routes(net, float(scale), out)
        print(f'wrote {out} (scale {scale})')
    if not args.net:
        return

    s = args.sat_flow_value
    jobs, report = [], {}
    for net in args.net:
        for scale in [float(x) for x in args.scales.split(',')]:
            plans, xs_fixed, flows = plans_for(net, scale, s, 'fixed')
            _, xs_web, _ = plans_for(net, scale, s, 'webster')
            report[(net, scale)] = {
                'veh_h_entering_per_tls': round(np.mean([sum(f.values()) for f in flows.values()])),
                'x_fixed_mean': round(float(np.mean(list(xs_fixed.values()))), 3),
                'x_fixed_max': round(max(xs_fixed.values()), 3),
                'x_webster_mean': round(float(np.mean(list(xs_web.values()))), 3),
                'sims': []}
            jobs += [(net, scale, k, plans) for k in range(args.seeds)]
    with Pool(args.workers) as pool:
        for net, scale, m in pool.imap_unordered(_sim, jobs):
            report[(net, scale)]['sims'].append(m)
    for (net, scale), r in sorted(report.items()):
        sims = r.pop('sims')
        agg = {k: round(float(np.mean([m[k] for m in sims])), 2)
               for k in ('mean_travel_time', 'mean_time_loss', 'mean_waiting_time',
                         'throughput_veh_h', 'mean_queue', 'unfinished', 'pending_insertion')}
        print(json.dumps({'net': net, 'scale': scale, **r, **agg}), flush=True)


if __name__ == '__main__':
    main()
