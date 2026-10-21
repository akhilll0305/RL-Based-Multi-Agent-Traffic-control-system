"""
Evaluate trained RL controllers on the baselines' evaluation seeds
==================================================================
Greedy policy, same SUMO seeds (100000 + k) and trip metrics as
scripts/evaluate_baselines.py, one CSV row per episode:

    outputs/results/rl_eval/<net>_<label>.csv

    python scripts/evaluate_rl.py --phase 0 --model outputs/models/ddqn_traffic_final.pth
    python tools/make_tables.py --name phase0 --ref "Webster=outputs/results/baselines/single_webster.csv" \
        "Fixed-time 30s=outputs/results/baselines/single_fixed_time.csv" \
        "DDQN=outputs/results/rl_eval/single_ddqn.csv"
"""

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------

import argparse
import contextlib
import io
import os
from multiprocessing import Pool

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')     # tiny nets: CPU is faster
import pandas as pd

EVAL_SEED_BASE = 100_000          # must match scripts/evaluate_baselines.py
OUT_DIR = 'outputs/results/rl_eval'


def _phase0_episode(job):
    model, k, route_file = job
    import torch
    torch.set_num_threads(1)
    from traffic_rl.envs.single import SumoEnvironment
    from traffic_rl.core.agent import DDQNAgent
    with contextlib.redirect_stdout(io.StringIO()):
        agent = DDQNAgent(state_dim=6, action_dim=2, hidden_dim=128)
        agent.load(model)
    env = SumoEnvironment(num_seconds=3600, delta_time=5, seed=EVAL_SEED_BASE + k,
                          **({'route_file': route_file} if route_file else {}))
    state, done, switches, reward = env.reset(), False, 0, 0.0
    while not done:
        a = agent.select_action(state, training=False)
        switches += int(a == 1)
        state, r, done, _ = env.step(a)
        reward += r
    m = env.finish_episode()
    per_q = m.pop('per_intersection_queue')
    m.update({f'queue_{t}': q for t, q in per_q.items()})
    return {'net': 'single', 'controller': 'ddqn', 'episode': k + 1,
            'episode_reward': reward, 'switch_actions': switches, **m}


PHASES = {0: ('single', 'ddqn', _phase0_episode)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', type=int, choices=list(PHASES), required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--label', default=None)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--route-file', default=None, help='default: the env default (calibrated demand)')
    args = p.parse_args()

    net, label, fn = PHASES[args.phase]
    label = args.label or label
    with Pool(args.workers) as pool:
        rows = pool.map(fn, [(args.model, k, args.route_file) for k in range(args.episodes)])
    df = pd.DataFrame(rows).sort_values('episode')
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f'{OUT_DIR}/{net}_{label}.csv'
    df.to_csv(out, index=False)
    print(f'{out}: travel {df.mean_travel_time.mean():.1f} s, time loss {df.mean_time_loss.mean():.1f} s, '
          f'wait {df.mean_waiting_time.mean():.1f} s, throughput {df.throughput_veh_h.mean():.0f} veh/h, '
          f'switches/episode {df.switch_actions.mean():.0f}')


if __name__ == '__main__':
    main()
