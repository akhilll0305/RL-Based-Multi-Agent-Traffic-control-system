"""Central path configuration.

Every directory the project reads from or writes to is defined here, resolved
absolutely from the repository root. Scripts call :func:`bootstrap` on startup
so that relative paths behave identically no matter where they are invoked from.
"""

import os
import sys
from pathlib import Path

# src/traffic_rl/paths.py -> parents[2] is the repository root
ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "src"
SUMO_FILES = ROOT / "sumo_files"
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
ASSETS = ROOT / "assets"

CHECKPOINTS = OUTPUTS / "checkpoints"
RESULTS = OUTPUTS / "results"
ANALYSIS = OUTPUTS / "analysis"
MODELS = OUTPUTS / "models"
EXPERIMENTS = OUTPUTS / "experiments"

# Per-phase SUMO network definitions
SUMO_SINGLE = SUMO_FILES / "single"
SUMO_GRID4 = SUMO_FILES / "grid4"
SUMO_GRID8 = SUMO_FILES / "grid8"


def bootstrap():
    """Put ``src/`` on ``sys.path`` and anchor the CWD at the repository root."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    os.chdir(ROOT)
    return ROOT
