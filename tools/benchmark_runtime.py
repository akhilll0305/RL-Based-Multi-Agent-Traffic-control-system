"""Wall-clock benchmark for one training episode on the 8-intersection network.

Runs one episode with 8 local DDQN agents doing select/store/train every decision,
which is the dominant per-episode cost of every Phase 2 training run. Used to pick
the device and the number of parallel seeds.

    python tools/benchmark_runtime.py --device cpu --num-seconds 3600
    python tools/benchmark_runtime.py --parallel 1,4,8,12 --device cpu
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_one(device, num_seconds, threads):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(ROOT / "src"))
    from traffic_rl.paths import bootstrap
    bootstrap()
    import contextlib, io
    import torch
    torch.set_num_threads(threads)
    from traffic_rl.envs.grid8_supervisor import SupervisorSumoEnvironment
    from traffic_rl.core.agent import DDQNAgent

    with contextlib.redirect_stdout(io.StringIO()):
        env = SupervisorSumoEnvironment(num_seconds=num_seconds, delta_time=5)
        agents = {t: DDQNAgent(state_dim=7, action_dim=2) for t in env.tls_ids}

    t0 = time.perf_counter()
    states = env.reset()
    sim_time = rl_time = 0.0
    steps = 0
    done = False
    while not done:
        a0 = time.perf_counter()
        enh = {t: list(states[t]) + [0.0] for t in env.tls_ids}
        actions = {t: agents[t].select_action(enh[t], training=True) for t in env.tls_ids}
        a1 = time.perf_counter()
        nxt, rewards, done, _ = env.step(actions)
        a2 = time.perf_counter()
        nenh = {t: list(nxt[t]) + [0.0] for t in env.tls_ids}
        for t in env.tls_ids:
            agents[t].memory.store(enh[t], actions[t], rewards[t], nenh[t], done)
            agents[t].train()
        a3 = time.perf_counter()
        sim_time += a2 - a1
        rl_time += (a1 - a0) + (a3 - a2)
        states = nxt
        steps += 1
    env.close()
    total = time.perf_counter() - t0
    return {"device": device, "num_seconds": num_seconds, "steps": steps,
            "total_s": round(total, 2), "sumo_s": round(sim_time, 2),
            "rl_s": round(rl_time, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--num-seconds", type=int, default=3600)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--parallel", default="", help="comma list of N, e.g. 1,4,8")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if not args.parallel:
        print(json.dumps(run_one(args.device, args.num_seconds, args.threads)))
        return

    for n in [int(x) for x in args.parallel.split(",")]:
        cmd = [sys.executable, __file__, "--device", args.device,
               "--num-seconds", str(args.num_seconds), "--threads", str(args.threads)]
        t0 = time.perf_counter()
        procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) for _ in range(n)]
        outs = [json.loads(pr.communicate()[0].strip().splitlines()[-1]) for pr in procs]
        wall = time.perf_counter() - t0
        per = sum(o["total_s"] for o in outs) / n
        print(json.dumps({"parallel": n, "device": args.device, "wall_s": round(wall, 1),
                          "mean_episode_s": round(per, 1),
                          "episodes_per_hour": round(3600 * n / wall, 1)}), flush=True)


if __name__ == "__main__":
    main()
