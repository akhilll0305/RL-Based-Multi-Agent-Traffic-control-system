"""
Evaluate the non-learning baselines on every network
====================================================
Fixed-time (30 s greens) and Webster (per-intersection isolated plan) run on the
same evaluation seeds and trip metrics as the RL rows, one CSV row per episode:

    outputs/results/baselines/<net>_<controller>.csv

    python scripts/evaluate_baselines.py --episodes 20
    python scripts/evaluate_baselines.py --net grid8 --episodes 20 --workers 8

Time-varying demand (Webster planned on the hourly mean, i.e. the calibrated file):

    python scripts/evaluate_baselines.py --demand dynamic   # -> <net>_<controller>_dynamic.csv
"""

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------

import argparse
import os
import sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from calibrate_demand import NETS, make_env, approach_edges, run_episode  # noqa: E402
from traffic_rl import baselines as bl  # noqa: E402

EVAL_SEED_BASE = 100_000          # RL evaluation must use the same base
SAT_FLOW = 1764.0                 # veh/h/lane, measured: tools/calibrate_demand.py --sat-flow
OUT_DIR = 'outputs/results/baselines'


def plans(net, controller, route_file):
    _, _, net_file, _ = NETS[net]
    env = make_env(net, route_file, seed=0)
    edges = approach_edges(env)
    if controller == 'fixed_time':
        return {tls: (30, 30) for tls in edges}
    if controller == 'max_pressure':
        return None
    flows = bl.approach_flows(route_file, edges)
    lanes = bl.lane_counts(net_file, edges)
    return {tls: bl.webster_plan(flows[tls], lanes[tls], SAT_FLOW)[:2] for tls in edges}


def _job(args):
    net, controller, route_file, k, plan, seed_base = args
    if controller == 'max_pressure':        # decides every 5 s like the RL agents
        env = make_env(net, route_file, seed=seed_base + k, delta_time=5)
        m = run_episode(env, bl.MaxPressureController().bind(env, NETS[net][2]))
    else:
        env = make_env(net, route_file, seed=seed_base + k)
        m = run_episode(env, bl.FixedTimeController(plan))
    per_q = m.pop('per_intersection_queue')
    m.update({f'queue_{t}': q for t, q in per_q.items()})
    return {'net': net, 'controller': controller, 'episode': k + 1, **m}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--net', choices=list(NETS), action='append')
    p.add_argument('--controller', choices=['fixed_time', 'webster', 'max_pressure'], action='append')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--route-file', default=None, help='default: the env default route file')
    p.add_argument('--demand', choices=['stationary', 'dynamic'], default='stationary',
                   help='dynamic: run on sumo_files/<net>/dynamic.rou.xml, plan on the calibrated '
                        '(hourly mean) demand, write <net>_<controller>_dynamic.csv')
    p.add_argument('--seed-base', type=int, default=EVAL_SEED_BASE)
    p.add_argument('--run-route-file', default=None,
                   help='simulate on this file while planning on --route-file (e.g. time-varying demand '
                        'with plans from its hourly mean); give --tag too')
    p.add_argument('--tag', default='', help='suffix for the output file names')
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = []
    for net in args.net or list(NETS):
        route = args.route_file or make_env(net, None, seed=0).route_file
        plan_route = route
        if args.demand == 'dynamic':
            route = f'sumo_files/{net}/dynamic.rou.xml'
        if args.run_route_file:
            route = args.run_route_file
        for ctl in args.controller or ['fixed_time', 'webster']:
            plan = plans(net, ctl, plan_route)
            print(f'{net:6s} {ctl:10s} route={route} plan(first tls)={next(iter(plan.values())) if plan else None}')
            jobs += [(net, ctl, route, k, plan, args.seed_base) for k in range(args.episodes)]
    with Pool(args.workers) as pool:
        rows = pool.map(_job, jobs)
    df = pd.DataFrame(rows)
    for (net, ctl), g in df.groupby(['net', 'controller']):
        tag = '' if args.demand == 'stationary' else f'_{args.demand}'
        tag += '' if args.seed_base == EVAL_SEED_BASE else f'_seeds{args.seed_base}'
        tag += args.tag
        out = f'{OUT_DIR}/{net}_{ctl}{tag}.csv'
        g.sort_values('episode').to_csv(out, index=False)
        print(f'{out}: travel {g.mean_travel_time.mean():.1f} s, time loss {g.mean_time_loss.mean():.1f} s, '
              f'wait {g.mean_waiting_time.mean():.1f} s, throughput {g.throughput_veh_h.mean():.0f} veh/h')


if __name__ == '__main__':
    main()
