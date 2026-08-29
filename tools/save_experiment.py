"""Archive a finished training run as a named, reproducible experiment.

Replaces the old one-off ``save_baseline.py`` / ``save_extended.py`` /
``save_improved.py`` scripts. The three configurations they hardcoded are kept
here as presets; ``--preset custom`` takes the values from the command line.

    python tools/save_experiment.py --preset baseline
    python tools/save_experiment.py --preset custom --name my_run --episodes 800
"""

# --- repo bootstrap: make `traffic_rl` importable and anchor the CWD at the repo root ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from traffic_rl.paths import bootstrap as _bootstrap
_bootstrap()
# --------------------------------------------------------------------------

import argparse

from experiment_manager import save_current_training

BASE = {
    'learning_rate': 0.001,
    'gamma': 0.95,
    'hidden_dim': 128,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'batch_size': 64,
    'buffer_capacity': 10000,
    'target_update_freq': 10,
    'state_dim': 6,
    'action_dim': 2,
    'reward_function': 'default: -(queue + 0.5*waiting + 10*switching)',
}

PRESETS = {
    'baseline': {
        'name': 'baseline_500ep_GPU',
        'description': 'First successful GPU training - 500 episodes, '
                       '61% improvement from -17308 to -6670 avg reward',
        'config': {**BASE, 'episodes': 500},
    },
    'extended': {
        'name': 'extended_1000ep_GPU',
        'description': 'Extended baseline - same configuration trained for 1000 episodes',
        'config': {**BASE, 'episodes': 1000},
    },
    'improved': {
        'name': 'improved_500ep_GPU',
        'description': 'Improved variant - gamma 0.99, 256 hidden units, 14-dim state',
        'config': {**BASE, 'episodes': 500, 'gamma': 0.99,
                   'hidden_dim': 256, 'state_dim': 14},
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', choices=[*PRESETS, 'custom'], default='baseline')
    ap.add_argument('--name', help='experiment name (required for --preset custom)')
    ap.add_argument('--description', default='')
    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--learning-rate', type=float, default=BASE['learning_rate'])
    ap.add_argument('--gamma', type=float, default=BASE['gamma'])
    ap.add_argument('--hidden-dim', type=int, default=BASE['hidden_dim'])
    ap.add_argument('--state-dim', type=int, default=BASE['state_dim'])
    args = ap.parse_args()

    if args.preset == 'custom':
        if not args.name:
            ap.error('--name is required when using --preset custom')
        name, description = args.name, args.description
        config = {**BASE, 'episodes': args.episodes, 'learning_rate': args.learning_rate,
                  'gamma': args.gamma, 'hidden_dim': args.hidden_dim,
                  'state_dim': args.state_dim}
    else:
        preset = PRESETS[args.preset]
        name = args.name or preset['name']
        description = args.description or preset['description']
        config = preset['config']

    exp_id = save_current_training(name=name, description=description, config=config)

    print("\n" + "=" * 78)
    print(f"Experiment saved: {exp_id}")
    print(f"Location:         outputs/experiments/{exp_id}/")
    print("\nEvaluate this snapshot later with:")
    print(f"  python scripts/phase0_single/run.py --mode evaluate "
          f"--model-path outputs/experiments/{exp_id}/model.pth --eval-episodes 10")
    print("=" * 78)


if __name__ == '__main__':
    main()
