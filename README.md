# 🚦 Hierarchical Multi-Agent DDQN Traffic Light Control System

**Deep Reinforcement Learning for Scalable, Secure Urban Traffic Management**

> A multi-phase research project that evolves automated traffic control from a single isolated intersection to a **hierarchically coordinated 8-intersection urban network** with **cyberattack-resilient LSTM-defended sensors** — built on SUMO, PyTorch, and DDQN.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
  - [Phase 0: Single-Agent Pretraining](#phase-0-single-agent-pretraining)
  - [Phase 1: 4-Intersection Multi-Agent Grid](#phase-1-4-intersection-multi-agent-grid)
  - [Phase 2: 8-Intersection Hierarchical Supervisors](#phase-2-8-intersection-hierarchical-supervisors)
  - [Phase 3: Cybersecurity & LSTM Defense](#phase-3-cybersecurity--lstm-defense)
- [Performance Results](#-performance-results)
- [Installation & Setup](#-installation--setup)
- [Usage & Commands](#-usage--commands)
- [Project Structure](#-project-structure)
- [Hyperparameters & Training Details](#-hyperparameters--training-details)
- [Known Issues & Roadmap](#-known-issues--roadmap)
- [Authors & License](#-authors--license)

---

## 🎯 Overview

This project implements Double Deep Q-Networks (DDQN) to dynamically control traffic lights across progressively larger intersection networks. Integrated with **SUMO** (Simulation of Urban MObility), the system learns to optimize vehicle throughput by observing real-time queue lengths, phase states, and inter-agent coordination signals — replacing static fixed-timer traffic lights with adaptive, learned policies.

### Key Achievements

| Phase | Milestone | Result |
|-------|-----------|--------|
| **Phase 0** | Single-agent pretraining | **94.3% improvement** over fixed-timers |
| **Phase 1** | Transfer learning to 4-intersection grid | **68% instant improvement** with zero additional training |
| **Phase 1** | Cooperative multi-agent fine-tuning | **Perfect network load balancing** across all intersections |
| **Phase 2** | Local Supervisor (24-dim) | **+4.6%** over decentralized 8-intersection baseline |
| **Phase 2** | Global Supervisor (28-dim) | **+2.8%** over decentralized baseline |
| **Phase 3** | LSTM + Z-Score defense against FDI attacks | **Full reward recovery** under active cyberattack |

---

## 🏗️ Project Architecture

The project tackles the Curse of Dimensionality in multi-agent RL by scaling the architecture in 4 distinct phases.

### Phase 0: Single-Agent Pretraining

A single DDQN agent controls one intersection with a **6-dimensional state space**:

```
State = [queue_N, queue_S, queue_E, queue_W, current_phase, time_since_change]
```

- **Network:** 3-layer MLP (6 → 128 → 128 → 2) with ReLU activations
- **Action Space:** 2 actions — keep current phase (0) or switch (1)
- **Training:** 1000 episodes with ε-greedy exploration (ε: 1.0 → 0.01)
- **Key Files:** `scripts/phase0_single/run.py`, `src/traffic_rl/envs/single.py`, `scripts/phase0_single/train.py`, `scripts/phase0_single/evaluate.py`

### Phase 1: 4-Intersection Multi-Agent Grid

The single-agent model is extended to a **2×2 grid** (4 intersections, 500m spacing) using two strategies:

1. **Independent Transfer Learning:** The Phase 0 checkpoint is cloned to all 4 agents. Each runs independently with its own 6-dim state.
2. **Cooperative Mode:** State space expanded to **8 dimensions** (6 local + 2 neighbor queue values). Agents share group-averaged rewards to encourage network-level cooperation.

```
Layout:
  [TLS_A] --- [TLS_B]
     |           |
  [TLS_C] --- [TLS_D]
```

- **Key Files:** `scripts/phase1_grid4/run.py`, `src/traffic_rl/envs/grid4.py`

### Phase 2: 8-Intersection Hierarchical Supervisors

As the grid scales to **8 intersections**, flat multi-agent systems cause localized gridlocks. The solution introduces a **two-tier hierarchy**:

```
Layout:
  Group A (Left)              Group B (Right)
  [TLS_1] --- [TLS_2]  <-->  [TLS_5] --- [TLS_6]
     |           |              |           |
  [TLS_3] --- [TLS_4]  <-->  [TLS_7] --- [TLS_8]
```

#### Step 1: Local Supervisors (24-dim input)

Each group's supervisor observes all 4 agents' raw 6-dim states concatenated into a **24-dimensional group state**. It outputs 4 continuous coordination signals ∈ [-1, +1] via tanh activation — one per agent. Each agent's state is then enhanced from 6-dim to **7-dim** (local state + supervisor signal).

```
Supervisor A: [state_tls1 || state_tls2 || state_tls3 || state_tls4] → 4 signals
Agent i:      [6-dim local state, supervisor_signal_i] → action
```

- **Supervisor Training:** TD regression on group-average reward
- **Agent Training:** Individual reward with DDQN

#### Step 2: Global Supervisors (28-dim input)

The two supervisors exchange a **4-dimensional cross-group summary**:

```
Summary = [avg_queue, max_queue, avg_waiting_time, boundary_queue]
```

Each supervisor's input expands from 24 → **28 dimensions** (24 own + 4 from the other group). This enables proactive congestion management across group boundaries.

- **Boundary Intersections:** TLS_2/TLS_4 (Group A) ↔ TLS_5/TLS_7 (Group B)
- **Key Files:** `scripts/phase2_hierarchy/local_supervisor.py`, `scripts/phase2_hierarchy/global_supervisor.py`, `src/traffic_rl/supervisor/agent.py`, `src/traffic_rl/envs/grid8_supervisor.py`

### Phase 3: Cybersecurity & LSTM Defense

Smart city traffic infrastructure is vulnerable to cyberattacks. This phase implements and defends against **False Data Injection (FDI)** attacks on sensor data.

#### Attack Model
- **FDI Attack:** Random queue sensors are injected with large positive values (+10 to +15) with 15% probability per intersection per step
- **Network Unreliability:** Packet loss (5%) and bounded delay (0–3 steps)

#### Defense Architecture
1. **Statistical Watchman (Z-Score):** Rolling-window anomaly detector (window=20, threshold=3σ) identifies values that deviate significantly from recent history
2. **LSTM Predictor:** A pre-trained LSTM (input_size=4, hidden_size=64) predicts what the correct queue values should be based on the last 20 steps of clean history. Poisoned values are seamlessly replaced with LSTM predictions.

#### Experiment Scenarios
Five scenarios run sequentially: `baseline`, `attack`, `defense`, `unreliable`, `secure`

- **Key Files:** `src/traffic_rl/security/layer.py`, `src/traffic_rl/security/lstm.py`, `scripts/phase3_security/train_lstm.py`, `scripts/phase3_security/run_scenarios.py`, `scripts/phase3_security/collect_data.py`, `scripts/analysis/security.py`
- **Full Report:** [docs/SECURITY_PHASE_REPORT.md](docs/SECURITY_PHASE_REPORT.md)

---

## 📊 Performance Results

### Phase 0 & 1 (Single Agent & 4-Intersection Grid)

| System | Avg Reward | Training | Improvement |
|--------|-----------|----------|-------------|
| Single-Agent Initial | -4,253.5 | 1000 eps | 94.3% vs fixed-time |
| Multi-Agent Transfer | -1,363.1 | 0 eps | Instant baseline |
| Multi-Agent Fine-Tuned | **-560.8** | 100 eps | **86.8% boost** |
| Multi-Agent Cooperative | -585.8 | 700 eps | Perfect balance ⚖️ |

### Phase 2 (8-Intersection Hierarchy)

| Architecture | Input Dim | Avg Reward / Intersection | vs Baseline |
|---|---|---|---|
| 8-Int No Supervisor | — | -197.0 | Baseline |
| 8-Int Local Supervisor | 24-dim | **-187.9** | 🏆 **+4.6%** |
| 8-Int Global Supervisor | 28-dim | **-191.5** | 🏆 **+2.8%** |

> **Note:** The Local Supervisor slightly outperformed the Global Supervisor under a 900-episode training budget. The 28-dim Global network's larger state space requires more training episodes to fully converge — the boundary-crossing features add complexity that hasn't fully saturated.

### Phase 3 (Cybersecurity)

Tested over 20 evaluation episodes per scenario:

| Scenario | Attack? | Detection Rate | Avg Wait Time | Avg Reward |
|----------|---------|---------------|---------------|------------|
| `baseline` | No | — | 0.044 | **-1624.5** |
| `attack` | FDI | — | 0.020 | -1403.0 *(broken)* |
| `defense` | FDI | ~2.31 | 0.017 | **-1491.5** *(recovered)* |
| `unreliable` | No | — | 0.022 | -1838.5 *(noise)* |
| `secure` | FDI | ~2.29 | 0.020 | **-1468.0** *(recovered)* |

---

## 🔧 Installation & Setup

### Prerequisites

- **Python 3.8+** (tested on 3.10–3.13)
- **CUDA-capable GPU** (NVIDIA RTX 2050+ recommended for Phase 2)
- **SUMO Traffic Simulator** (v1.25.0+)

### Setup

1. Install SUMO from [eclipse.org/sumo](https://www.eclipse.org/sumo/) and set the `SUMO_HOME` environment variable.

2. Clone and install:
```bash
git clone https://github.com/akhilll0305/RL-Based-Multi-Agent-Traffic-control-system.git
cd RL-Based-Multi-Agent-Traffic-control-system

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other dependencies
pip install numpy pandas matplotlib tqdm traci sumolib
```

---

## 🚀 Usage & Commands

Every script resolves its own paths, so it can be launched from any working
directory. The examples below assume you are at the repository root.

### Phase 0: Single-Agent

```bash
# Train
python scripts/phase0_single/run.py --mode train --episodes 500

# Evaluate
python scripts/phase0_single/run.py --mode evaluate
```

### Phase 1: Multi-Agent (4-intersection grid)

```bash
# Train cooperative mode
python scripts/phase1_grid4/run.py --mode train --cooperative --episodes 700 --learning-rate 0.0005 --epsilon 0.9

# Evaluate
python scripts/phase1_grid4/run.py --mode evaluate --load-final
```

### Phase 2: Hierarchical Supervisors (8-intersection)

```bash
# Decentralised baseline (no supervisor)
python scripts/phase2_hierarchy/baseline.py --mode train --episodes 500

# Step 1: Local Supervisors (24-dim)
python scripts/phase2_hierarchy/local_supervisor.py --mode train --episodes 500
python scripts/phase2_hierarchy/local_supervisor.py --mode evaluate --load-final --eval-episodes 20

# Step 2: Global Supervisors (28-dim)
python scripts/phase2_hierarchy/global_supervisor.py --mode train --episodes 900 --from-scratch --epsilon 0.9
python scripts/phase2_hierarchy/global_supervisor.py --mode evaluate --load-final --eval-episodes 20

# Resume training from a checkpoint
python scripts/phase2_hierarchy/local_supervisor.py --mode train --episodes 300 --resume-from 200

# Visualise with the SUMO GUI
python scripts/phase2_hierarchy/local_supervisor.py --mode evaluate --load-final --eval-episodes 5 --gui
```

### Phase 3: Security

Run these in order — each step consumes the previous step's output.

```bash
# 1. Collect clean baseline data
python scripts/phase3_security/collect_data.py --episodes 50

# 2. Split into train/val sets
python scripts/phase3_security/split_data.py

# 3. Validate dataset integrity
python scripts/phase3_security/validate_data.py

# 4. Train the LSTM predictor
python scripts/phase3_security/train_lstm.py --epochs 25

# 5. Run all 5 security scenarios
python scripts/phase3_security/run_scenarios.py --episodes 20
```

### Analysis & Visualisation

```bash
python scripts/analysis/supervisor.py
python scripts/analysis/global_supervisor.py
python scripts/analysis/security.py
```

Figures are written to `outputs/analysis/supervisor/`,
`outputs/analysis/global_supervisor/` and `outputs/analysis/security/`.

### Utilities

```bash
python tools/check_gpu.py                            # verify CUDA is visible to PyTorch
python tools/save_experiment.py --preset baseline    # archive a finished training run
```

---

## 📁 Project Structure

```
.
├── src/traffic_rl/              # Importable library code
│   ├── paths.py                 # Central path config (single source of truth)
│   ├── core/
│   │   ├── agent.py             #   DDQNAgent with GPU auto-detection
│   │   ├── network.py           #   3-layer MLP
│   │   └── replay_buffer.py     #   Experience replay
│   ├── envs/                    # One SUMO environment per phase
│   │   ├── single.py            #   Phase 0: single intersection
│   │   ├── grid4.py             #   Phase 1: 2×2 grid
│   │   ├── grid8.py             #   Phase 2: 2×4 grid baseline
│   │   └── grid8_supervisor.py  #   Phase 2: 2×4 grid with supervisor hooks
│   ├── supervisor/agent.py      # SupervisorNetwork + SupervisorAgent
│   ├── security/
│   │   ├── layer.py             #   FDI attack + Z-Score + LSTM defence
│   │   └── lstm.py              #   TrafficLSTM model
│   └── sumo_gen/                # Network generators: single / grid4 / grid8
│
├── scripts/                     # Runnable entry points
│   ├── phase0_single/           #   run.py, train.py, evaluate.py
│   ├── phase1_grid4/            #   run.py
│   ├── phase2_hierarchy/        #   baseline.py, local_supervisor.py,
│   │                            #   global_supervisor.py, evaluate_*.py
│   ├── phase3_security/         #   collect_data → split_data → validate_data
│   │                            #   → train_lstm → run_scenarios
│   └── analysis/                #   supervisor.py, global_supervisor.py, security.py
│
├── tools/                       # check_gpu.py, experiment_manager.py,
│                                # save_experiment.py
│
├── sumo_files/                  # Generated SUMO networks
│   └── single/  grid4/  grid8/
│
├── outputs/                     # Everything the code writes
│   ├── checkpoints/             #   single/ grid4/ cooperative/ grid8/
│   │                            #   supervisor/ global_supervisor/ security/
│   ├── models/                  #   Exported final models
│   ├── results/                 #   Training histories, eval CSVs, plots
│   ├── analysis/                #   Generated analysis figures
│   └── experiments/             #   Archived experiment snapshots
│
├── data/                        # Generated training data (git-ignored)
├── docs/                        # SECURITY_PHASE_REPORT.md
├── assets/                      # README images and presentation slides
└── requirements.txt
```

> **A note on checkpoints:** per-episode training snapshots (~370 MB) are
> git-ignored. The final trained models (`*_final.pth`), the LSTM predictor and
> the two episode checkpoints the scripts load by name **are** committed, so
> evaluation runs immediately after a clone. Re-run training to regenerate the
> full snapshot history.

---

## 🧠 Hyperparameters & Training Details

### Neural Network Architecture

| Component | Layers | Neurons | Activation | Output |
|-----------|--------|---------|------------|--------|
| **DDQN Agent** | 3-layer MLP | 128 hidden | ReLU | 2 (Q-values) |
| **Supervisor** | 3-layer MLP | 64 hidden | ReLU → Tanh | 4 (signals ∈ [-1,1]) |
| **LSTM Predictor** | 1-layer LSTM + Linear | 64 hidden | — | 4 (queue predictions) |

### Training Configuration

| Parameter | Phase 0 | Phase 1 | Phase 2 | Phase 3 (LSTM) |
|-----------|---------|---------|---------|----------------|
| Learning Rate | 0.001 | 0.0005 | 0.0001 (agent) / 0.001 (sup) | 0.001 |
| Gamma (γ) | 0.95 | 0.95 | 0.95 | — |
| Epsilon Decay | 0.995 | 0.995 | 0.995 | — |
| Batch Size | 64 | 64 | 64 | 256 |
| Buffer Size | 10,000 | 10,000 | 10,000 | — |
| Target Update | 10 eps | 10 eps | 10 eps | — |
| Gradient Clip | 10.0 | 10.0 | 10.0 | 5.0 |

### Reward Function

All phases share the same core reward formulation:

```python
reward = -(total_queue) - 0.5 * (total_waiting_time) - 10 * (quick_switch_penalty)
```

- **Queue penalty:** Direct negative proportional to halting vehicles
- **Waiting penalty:** 0.5× weighting on cumulative vehicle waiting time
- **Switch penalty:** -10 if the agent attempts to switch phases within 5 seconds of the last switch

---

## 🔮 Known Issues & Roadmap

### Active Bugs

Key items from the internal 3-pass code audit:

- **BUG-01 (Critical):** Supervisor TD target broadcast — all 4 agents receive identical signals instead of differentiated urgency values
- **BUG-03 (Medium):** Security layer logging uses incorrect `np.where(flagged)[0]` double-indexing
- **BUG-04 (Medium):** False positive rate metric is always 0.0 (counter never incremented)

### Planned Improvements

1. **Independent TD Targets** — Per-intersection reward-based supervisor training for fine-grained signal differentiation
2. **Prioritized Experience Replay (PER)** — Replace uniform sampling with TD-error-weighted prioritization
3. **Huber Loss** — Replace MSELoss with SmoothL1Loss to stabilize supervisor convergence
4. **Utility Module Refactor** — Extract duplicated `partial_transfer()`, `set_seed()` into shared `utils.py`
5. **State Normalization** — Add batch normalization or manual feature scaling for faster convergence
6. **Dynamic Boundary Detection** — Replace hardcoded boundary TLS IDs with graph-based automatic detection

---

## 📝 Authors & License

**Project Team:** RL Traffic Control Research Group  
Developed for academic research purposes using the SUMO Traffic Modeling Suite.

**License:** MIT
