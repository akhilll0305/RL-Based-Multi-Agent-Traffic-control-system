"""
Security-phase experiment runner.

Step 6.1 scope:
- load_trained_system(env): load 8 DDQN agents + 2 supervisors from final checkpoints
- run_single_scenario(...): evaluate one scenario loop with security layer inserted

This file does not modify existing training/evaluation files.
"""

from __future__ import annotations

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------


import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from traffic_rl.core.agent import DDQNAgent
from traffic_rl.security.layer import SecurityLayer
from traffic_rl.envs.grid8_supervisor import SupervisorSumoEnvironment
from traffic_rl.supervisor.agent import SupervisorAgent


CHECKPOINT_DIR_AGENTS = "outputs/checkpoints/supervisor/agents"
CHECKPOINT_DIR_SUPERVISORS = "outputs/checkpoints/supervisor/supervisors"
RESULTS_DIR = "outputs/results/security"
SCENARIOS = ["baseline", "guard", "attack", "defense", "unreliable", "secure"]
METRICS = ('mean_time_in_system_all', 'mean_travel_time', 'mean_time_loss', 'mean_waiting_time',
           'throughput_veh_h', 'pending_insertion', 'mean_queue', 'unfinished')


def load_trained_system(
    env: SupervisorSumoEnvironment,
) -> Tuple[Dict[str, DDQNAgent], SupervisorAgent, SupervisorAgent]:
    """
    Load frozen trained local agents and supervisors for security evaluation.

    Returns:
        agents, supervisor_a, supervisor_b
    """
    agents: Dict[str, DDQNAgent] = {}

    for tls in env.tls_ids:
        agent = DDQNAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            hidden_dim=128,
            learning_rate=1e-4,
            epsilon_start=0.0,
            epsilon_decay=1.0,
            epsilon_min=0.0,
        )
        ckpt_path = os.path.join(CHECKPOINT_DIR_AGENTS, f"{tls}_final.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing agent checkpoint: {ckpt_path}")
        agent.load(ckpt_path)
        agent.epsilon = 0.0
        agents[tls] = agent

    supervisor_a = SupervisorAgent(
        group_tls_ids=env.group_a,
        state_dim_per_agent=env.get_local_state_dim(),
        hidden_dim=64,
        learning_rate=0.001,
        gamma=0.95,
        buffer_capacity=10_000,
        batch_size=64,
    )
    supervisor_b = SupervisorAgent(
        group_tls_ids=env.group_b,
        state_dim_per_agent=env.get_local_state_dim(),
        hidden_dim=64,
        learning_rate=0.001,
        gamma=0.95,
        buffer_capacity=10_000,
        batch_size=64,
    )

    sup_a_ckpt = os.path.join(CHECKPOINT_DIR_SUPERVISORS, "supervisor_a_final.pth")
    sup_b_ckpt = os.path.join(CHECKPOINT_DIR_SUPERVISORS, "supervisor_b_final.pth")
    if not os.path.exists(sup_a_ckpt):
        raise FileNotFoundError(f"Missing supervisor checkpoint: {sup_a_ckpt}")
    if not os.path.exists(sup_b_ckpt):
        raise FileNotFoundError(f"Missing supervisor checkpoint: {sup_b_ckpt}")

    supervisor_a.load(sup_a_ckpt)
    supervisor_b.load(sup_b_ckpt)

    return agents, supervisor_a, supervisor_b


def run_single_scenario(
    env: SupervisorSumoEnvironment,
    agents: Dict[str, DDQNAgent],
    supervisor_a: SupervisorAgent,
    supervisor_b: SupervisorAgent,
    security: SecurityLayer,
    episodes: int,
) -> Dict[str, object]:
    """
    Run one security scenario and collect episode-level metrics.

    Security insertion points:
    - after env.reset() local states
    - after each env.step() next local states
    """
    per_intersection_rewards = {tls: [] for tls in env.tls_ids}
    network_rewards = []
    avg_intersection_rewards = []
    wait_times = []
    episode_rows = []

    for ep in range(episodes):
        local_states = env.reset()
        security.reset()
        det_before = {k: getattr(security, k) for k in ('tp', 'fp', 'fn', 'tn')}
        local_states = security.process(local_states, step=0)

        signals_a = supervisor_a.get_signals({tls: local_states[tls] for tls in env.group_a})
        signals_b = supervisor_b.get_signals({tls: local_states[tls] for tls in env.group_b})
        enhanced = env.build_enhanced_states(local_states, signals_a, signals_b)

        done = False
        step = 0
        ep_reward = {tls: 0.0 for tls in env.tls_ids}

        while not done:
            actions = {
                tls: agents[tls].select_action(enhanced[tls], training=False)
                for tls in env.tls_ids
            }

            next_local, rewards, done, info = env.step(actions)
            step += 1
            next_local = security.process(next_local, step=step)

            next_sig_a = supervisor_a.get_signals({tls: next_local[tls] for tls in env.group_a})
            next_sig_b = supervisor_b.get_signals({tls: next_local[tls] for tls in env.group_b})
            enhanced = env.build_enhanced_states(next_local, next_sig_a, next_sig_b)
            local_states = next_local

            for tls in env.tls_ids:
                ep_reward[tls] += float(rewards[tls])


        trip = env.finish_episode()
        # Trip-based: accumulated waiting time averaged over completed trips
        wait_times.append(float(trip["mean_waiting_time"]))

        for tls in env.tls_ids:
            per_intersection_rewards[tls].append(ep_reward[tls])

        net_reward = float(sum(ep_reward.values()))
        avg_intersection = float(net_reward / len(env.tls_ids))

        network_rewards.append(net_reward)
        avg_intersection_rewards.append(avg_intersection)

        row = {
            "episode": int(ep + 1),
            "network_reward": net_reward,
            "avg_intersection_reward": avg_intersection,
            "avg_wait_time": float(wait_times[-1]),
            "episode_seed": trip["episode_seed"],
        }
        for k in METRICS:
            row[k] = float(trip[k])
        for k in ('tp', 'fp', 'fn', 'tn'):
            row[k] = int(getattr(security, k) - det_before[k])
        for tls in env.tls_ids:
            row[tls] = float(ep_reward[tls])
        episode_rows.append(row)

    metrics = security.get_metrics()

    summary = {
        "episodes": int(episodes),
        "avg_network_reward": float(np.mean(network_rewards)) if network_rewards else 0.0,
        "std_network_reward": float(np.std(network_rewards)) if network_rewards else 0.0,
        "avg_intersection_reward": float(np.mean(avg_intersection_rewards)) if avg_intersection_rewards else 0.0,
        "std_intersection_reward": float(np.std(avg_intersection_rewards)) if avg_intersection_rewards else 0.0,
        "avg_wait_time": float(np.mean(wait_times)) if wait_times else 0.0,
        "std_wait_time": float(np.std(wait_times)) if wait_times else 0.0,
        "detection_rate": float(metrics.get("detection_rate", 0.0)),
        "false_positive_rate": float(metrics.get("false_positive_rate", 0.0)),
    }
    for k in METRICS:
        vals = [r[k] for r in episode_rows]
        summary[f"mean_{k}" if not k.startswith("mean_") else k] = float(np.mean(vals))
        summary[f"std_{k.replace('mean_', '')}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    for k in ("precision", "recall", "f1", "mcc", "roc_auc", "flags_per_attack",
              "tp", "fp", "fn", "tn", "total_attack_events"):
        summary[k] = float(metrics.get(k, float("nan")))

    return {
        "episodes": int(episodes),
        "episode_rows": episode_rows,
        "network_rewards": network_rewards,
        "avg_intersection_rewards": avg_intersection_rewards,
        "per_intersection_rewards": per_intersection_rewards,
        "wait_times": wait_times,
        "security_metrics": metrics,
        "summary": summary,
        "avg_network_reward": summary["avg_network_reward"],
        "std_network_reward": summary["std_network_reward"],
        "avg_intersection_reward": summary["avg_intersection_reward"],
        "std_intersection_reward": summary["std_intersection_reward"],
        "avg_wait_time": summary["avg_wait_time"],
        "detection_rate": summary["detection_rate"],
        "false_positive_rate": summary["false_positive_rate"],
    }


def _build_security_layer(mode: str, seed: int = 42, attack_type: str = "random",
                          z_threshold: float = 3.0, std_floor: float = 1e-6,
                          min_deviation: float = 0.0, persistence: int = 1,
                          detector: str = "zscore", residual_threshold: float = 5.0,
                          spike_rollback: bool = False) -> SecurityLayer:
    """Create scenario-specific security layer with reasonable defaults."""
    return SecurityLayer(
        mode=mode,
        attack_type=attack_type,
        std_floor=std_floor,
        min_deviation=min_deviation,
        persistence=persistence,
        detector=detector,
        residual_threshold=residual_threshold,
        spike_rollback=spike_rollback,
        fdi_prob=0.15,
        fdi_min=10.0,
        fdi_max=15.0,
        window_size=20,
        z_threshold=z_threshold,
        packet_loss_prob=0.05,
        max_delay_steps=3,
        lstm_checkpoint="outputs/checkpoints/security/lstm_predictor.pth",
        seed=seed,
    )


def run_all_scenarios(
    episodes: int = 20,
    use_gui: bool = False,
    num_seconds: int = 3600,
    delta_time: int = 5,
    seed: int = 42,
    attack_type: str = "random",
    z_threshold: float = 3.0,
    std_floor: float = 1e-6,
    eval_seed: int = 100_000,
    route_file: str = None,
    tag: str = "",
    min_deviation: float = 0.0,
    persistence: int = 1,
    scenarios=SCENARIOS,
    detector: str = "zscore",
    residual_threshold: float = 5.0,
    spike_rollback: bool = False,
) -> Dict[str, Dict[str, object]]:
    """
    Run all five security scenarios sequentially and save per-scenario episode CSVs.

    Returns:
        scenario name -> run_single_scenario output
    """
    out_dir = os.path.join(RESULTS_DIR, attack_type + tag)
    os.makedirs(out_dir, exist_ok=True)

    # Bootstrap environment for model initialization/loading dimensions.
    routes = {'route_file': route_file} if route_file else {}
    bootstrap_env = SupervisorSumoEnvironment(
        use_gui=False,
        num_seconds=num_seconds,
        delta_time=delta_time,
        **routes,
    )
    agents, supervisor_a, supervisor_b = load_trained_system(bootstrap_env)

    all_results: Dict[str, Dict[str, object]] = {}

    for scenario in tqdm(scenarios, desc="Running scenarios"):
        # Same SUMO seed per episode index in every scenario (common random numbers),
        # so scenarios can be compared episode-by-episode with a paired test.
        env = SupervisorSumoEnvironment(
            use_gui=use_gui,
            num_seconds=num_seconds,
            delta_time=delta_time,
            seed=eval_seed,       # the shared test seeds, so 'baseline' matches the Phase 2 evaluation
            **routes,
        )
        security = _build_security_layer(mode=scenario, seed=seed, attack_type=attack_type,
                                         z_threshold=z_threshold, std_floor=std_floor,
                                         min_deviation=min_deviation, persistence=persistence,
                                         detector=detector, residual_threshold=residual_threshold,
                                         spike_rollback=spike_rollback)

        try:
            result = run_single_scenario(
                env=env,
                agents=agents,
                supervisor_a=supervisor_a,
                supervisor_b=supervisor_b,
                security=security,
                episodes=episodes,
            )
        finally:
            env.close()

        df = pd.DataFrame(result["episode_rows"])
        out_path = os.path.join(out_dir, f"{scenario}_results.csv")
        df.to_csv(out_path, index=False)

        all_results[scenario] = result

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all security scenarios sequentially")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per scenario")
    parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
    parser.add_argument("--num-seconds", type=int, default=3600, help="Episode duration")
    parser.add_argument("--delta-time", type=int, default=5, help="Decision interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--attack-type", default="random",
                        choices=["random", "detector_aware", "ramp", "replay", "hide"])
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--std-floor", type=float, default=1e-6,
                        help="Minimum std in the z-score (1e-6 = original detector)")
    parser.add_argument("--eval-seed", type=int, default=100_000, help="SUMO seed of the first episode")
    parser.add_argument("--route-file", default=None, help="Route file (default: grid8 calibrated)")
    parser.add_argument("--tag", default="", help="Suffix for the output folder, e.g. _floor1")
    parser.add_argument("--min-deviation", type=float, default=0.0,
                        help="Detector also needs |value - window mean| >= this many vehicles (0 = original)")
    parser.add_argument("--persistence", type=int, default=1,
                        help="Correct a lane after this many flagged steps in a row (1 = original)")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    parser.add_argument("--detector", default="zscore", choices=["zscore", "residual"],
                        help="zscore = original window z-score; residual = |reading - LSTM forecast|")
    parser.add_argument("--residual-threshold", type=float, default=5.0,
                        help="Vehicles between reading and LSTM forecast that count as an alarm")
    parser.add_argument("--spike-rollback", action="store_true",
                        help="Residual detector: erase one-step spikes from the detector history")
    args = parser.parse_args()
    out_dir = os.path.join(RESULTS_DIR, args.attack_type + args.tag)

    results = run_all_scenarios(
        episodes=args.episodes,
        use_gui=args.gui,
        num_seconds=args.num_seconds,
        delta_time=args.delta_time,
        seed=args.seed,
        attack_type=args.attack_type,
        z_threshold=args.z_threshold,
        std_floor=args.std_floor,
        eval_seed=args.eval_seed,
        route_file=args.route_file,
        tag=args.tag,
        min_deviation=args.min_deviation,
        persistence=args.persistence,
        scenarios=args.scenarios,
        detector=args.detector,
        residual_threshold=args.residual_threshold,
        spike_rollback=args.spike_rollback,
    )

    summary_rows = []
    print("\nPer-scenario run complete. Saved files:")
    for scenario in args.scenarios:
        print(f"  {os.path.join(out_dir, f'{scenario}_results.csv')}")
        summary = results[scenario]["summary"]
        print(
            f"    avg_network_reward={summary['avg_network_reward']:.3f}, "
            f"avg_wait_time={summary['avg_wait_time']:.3f}, "
            f"detection_rate={summary['detection_rate']:.3f}, "
            f"false_positive_rate={summary['false_positive_rate']:.3f}"
        )
        # Store for combined table
        row = {"scenario": scenario}
        row.update(summary)
        summary_rows.append(row)

    # Save master summary table
    summary_df = pd.DataFrame(summary_rows)
    comparison_path = os.path.join(out_dir, "scenario_comparison.csv")
    summary_df.to_csv(comparison_path, index=False)
    print(f"\n=> Master comparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
