# 🚦 Federated Hierarchical Multi-Agent Traffic Light Control System

**Advanced Deep Reinforcement Learning with Federated Learning for Intelligent Traffic Management**

> A scalable multi-agent reinforcement learning system that uses Federated Hierarchical MARL to manage 8 intersections across 2 zones with supervisor agents. Extends from single-agent DDQN → multi-agent → cooperative → **federated hierarchical** with FedAvg weight aggregation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Federated Hierarchical System](#-federated-hierarchical-system-new)
- [Performance Results](#performance-results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Technical Highlights](#technical-highlights)
- [Future Work](#future-work)

---

## 🎯 Overview

This project implements a sophisticated multi-agent deep reinforcement learning system for traffic light control using Double Deep Q-Networks (DDQN). Starting from a single-agent baseline trained for 1000 episodes, we demonstrate:

1. **Transfer Learning Success**: Single-agent knowledge transfers effectively to multi-agent scenarios
2. **Multi-Agent Scalability**: System scales from 1 to 4 intersections with improved per-intersection performance
3. **Cooperative Coordination**: Agents sharing neighbor information achieve perfect load balancing
4. **🆕 Federated Hierarchical Control**: 8 intersections in 2 zones with supervisor agents and FedAvg
5. **Real-World Integration**: Full SUMO (Simulation of Urban MObility) integration with GPU acceleration

### Key Achievements

- ✅ **86.8% improvement** in reward over single-agent baseline
- ✅ **58.9% improvement** from fine-tuning (100 episodes)
- ✅ **Perfect load balancing** with cooperative agents
- ✅ **Near-zero waiting times** achieved across all systems

---

## 🌟 Key Features

### Multi-Mode Training
- **Transfer Learning**: Leverage pre-trained single-agent models
- **Independent Multi-Agent**: Each intersection optimized locally
- **Cooperative Multi-Agent**: Network-level coordination with neighbor information
- **Resume Capability**: Continue training from any checkpoint

### Advanced RL Techniques
- Double Deep Q-Network (DDQN) architecture
- Experience replay with prioritized sampling
- Target network for stability
- Epsilon-greedy exploration with decay
- GPU-accelerated training (CUDA support)

### Comprehensive Evaluation
- Multiple checkpoint saving (every 20 episodes)
- Real-time training metrics
- Performance visualization tools
- SUMO-GUI integration for visual inspection
- Statistical analysis across multiple episodes

---

## 🏗️ System Architecture

### Network Topology

```
        ┌─────────────┐
        │   BOUNDARY  │
        │   (North)   │
        └──────┬──────┘
               │
    ┌──────────┼──────────┐
    │          │          │
BOUNDARY   TLS_1 ←→ TLS_2   BOUNDARY
(West)         │          │   (East)
               ↕          ↕
           TLS_3 ←→ TLS_4
               │          │
               └──────────┘
                   │
            ┌──────┴──────┐
            │   BOUNDARY  │
            │   (South)   │
            └─────────────┘

2×2 Grid Network: 4 Intersections, 500m spacing
```

### Agent Architecture

**State Space:**
- Independent Mode: 6 features
  - Queue lengths (N, S, E, W): 4 features
  - Current phase: 1 feature
  - Time since last change: 1 feature

- Cooperative Mode: 8 features
  - Base features: 6 (as above)
  - Neighbor queue info: 2 features (adjacent intersections)

**Action Space:**
- Action 0: Keep current phase
- Action 1: Switch to alternate phase

**Neural Network:**
- Input Layer: 6 or 8 neurons (state dimension)
- Hidden Layer 1: 128 neurons (ReLU)
- Hidden Layer 2: 128 neurons (ReLU)
- Output Layer: 2 neurons (Q-values for actions)

**Reward Function:**
```python
reward = -(queue_length + 0.5 * waiting_time + 10 * phase_switch_penalty)
```

![Architecture Diagram](readme_visuals/4_architecture_diagram.png)
*Figure 4: Multi-agent system architecture showing agent-environment interaction and cooperative connections*

---

## 🆕 Federated Hierarchical System (NEW)

### Overview

The latest extension scales the system to **8 intersections** in a **4×2 grid**, organized into **2 zones** managed by **supervisor agents** that communicate via **Federated Averaging (FedAvg)**.

### Network Topology (4×2 Grid)

```
       Zone A (Supervisor 1)           Zone B (Supervisor 2)
      ┌───────────────────────┐       ┌───────────────────────┐
      │  [TLS1] ──── [TLS2]──┼───────┼──[TLS5] ──── [TLS6]  │
      │    |            |     │       │    |            |     │
      │  [TLS3] ──── [TLS4]──┼───────┼──[TLS7] ──── [TLS8]  │
      └───────────────────────┘       └───────────────────────┘
                              ↑       ↑
                         Inter-Zone Bridges
                       (i2↔i5 and i4↔i7)
```

### Hierarchical Architecture

| Level | Component | Input | Output | Count |
|-------|-----------|-------|--------|-------|
| **Local** | DDQN Agent | 8-dim state (queues, phase, zone_avg, cross_zone_pressure) | Keep/Switch phase | 8 agents |
| **Zone** | Supervisor Agent | 24-dim state (own zone 12 + neighbor zone 12) | NS_priority / EW_priority / Balanced | 2 supervisors |
| **Global** | FedAvg Coordinator | Agent weights | Averaged global model | 1 coordinator |

### Federated Averaging (FedAvg)

Based on McMahan et al. (2017), the system uses **two levels of weight aggregation**:

1. **Intra-Zone FedAvg** (every 10 episodes, α=0.8):
   - Average weights of 4 local agents within each zone
   - Preserves zone-specific knowledge while sharing learned policies

2. **Inter-Zone FedAvg** (every 25 episodes, α=0.5):
   - Average weights across all 8 local agents (softer blend)
   - Average supervisor weights for global coordination
   - Creates a shared global model while allowing local adaptation

### Hierarchical Reward Structure

```
Local reward = -queue_length - 0.5 × waiting_time - phase_switch_penalty
Zone reward  = 0.7 × local_reward + 0.3 × zone_average_reward
Final reward = zone_reward + supervisor_modifier
```

The supervisor dynamically adjusts local agent rewards:
- **NS_priority**: Bonus for maintaining N-S green phases
- **EW_priority**: Bonus for maintaining E-W green phases
- **Balanced**: No directional bias

### Usage

```bash
# Generate 8-intersection network
python generate_sumo_federated.py

# Train federated system (700 episodes)
python main_federated.py --episodes 700

# Train with SUMO visualization
python main_federated.py --gui

# Evaluate trained model
python main_federated.py --evaluate --gui

# Generate training plots
python evaluate_federated.py
```

### New Files

| File | Description |
|------|-------------|
| `generate_sumo_federated.py` | 4×2 grid SUMO network generator |
| `sumo_environment_federated.py` | 8-intersection environment with zone-level observations |
| `supervisor_agent.py` | Supervisor DDQN with coordination signals |
| `federated_learning.py` | FedAvg implementation + FederatedCoordinator |
| `train_federated.py` | Hierarchical training loop for all 10 agents |
| `main_federated.py` | CLI entry point for training and evaluation |
| `evaluate_federated.py` | Visualization and comparison plots |

---

## 📊 Performance Results

### System Comparison

| System | Avg Reward | Training Time | Method | Performance |
|--------|-----------|---------------|--------|-------------|
| Single-Agent | -4,253.5 | 1000 episodes | Baseline | 94.3% vs fixed-time |
| Multi-Agent Transfer | -1,363.1 | 0 episodes | Episode 900×4 | 68% better per intersection |
| Multi-Agent Fine-Tuned | **-560.8** | 100 episodes | Transfer + Train | **86.8% improvement** ✅ |
| Multi-Agent Cooperative | -585.8 | 700 episodes | From scratch | Perfect balance ⚖️ |

### Per-Intersection Performance

**Independent Fine-Tuned:**
```
TLS_1 (Top-Left):     -807.5  (handles higher load)
TLS_2 (Top-Right):    -663.5  
TLS_3 (Bottom-Left):  -448.0  
TLS_4 (Bottom-Right): -324.0  (optimal location)

Average: -560.8
```

**Cooperative:**
```
All Intersections:    -585.8  (perfectly balanced)

Average: -585.8
```

### Training Progression

The system shows clear learning curves with three distinct phases:

1. **Phase 1 - Transfer (Episode 900)**: Instant deployment with strong baseline performance
2. **Phase 2 - Fine-Tuning (100 episodes)**: Rapid adaptation to multi-agent network dynamics
3. **Phase 3 - Convergence**: Stable optimal policy achieved

![System Performance](readme_visuals/1_system_performance.png)
*Figure 1: Performance comparison across all system architectures*

![Training Evolution](readme_visuals/2_training_evolution.png)
*Figure 2: Progressive improvement through transfer learning and fine-tuning*

![Per-Intersection Comparison](readme_visuals/3_intersection_comparison.png)
*Figure 3: Independent vs Cooperative performance breakdown*

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: NVIDIA RTX 2050 or better)
- SUMO Traffic Simulator
- 8GB+ RAM

### Install Dependencies

```bash
# Clone repository
git clone <your-repo-url>
cd RL-Project-main

# Create virtual environment
python -m venv myenv

# Activate environment
# Windows:
myenv\Scripts\activate
# Linux/Mac:
source myenv/bin/activate

# Install packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install traci numpy pandas matplotlib tqdm
```

### SUMO Installation

Download and install SUMO from: https://www.eclipse.org/sumo/

Set environment variable:
```bash
# Windows
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo

# Linux/Mac
export SUMO_HOME=/usr/share/sumo
```

---

## 🚀 Usage

### Generate Multi-Agent Network

```bash
python generate_sumo_multiagent.py
```

Creates a 2×2 grid network in `sumo_files_multiagent/`

### Evaluate Pre-Trained Single-Agent

```bash
python main.py --mode evaluate --model-path checkpoints/ddqn_episode_900.pth --eval-episodes 10
```

### Multi-Agent: Test Transfer Learning

```bash
python main_multiagent.py --mode test --test-episodes 10 --pretrained-model checkpoints/ddqn_episode_900.pth
```

### Multi-Agent: Fine-Tune (Independent)

```bash
python main_multiagent.py --mode train --episodes 100 --learning-rate 0.0001 --epsilon 0.1 --pretrained-model checkpoints/ddqn_episode_900.pth
```

### Multi-Agent: Train Cooperative

```bash
python main_multiagent.py --mode train --cooperative --episodes 700 --learning-rate 0.0005 --epsilon 0.9
```

### Evaluate Fine-Tuned Models

**Independent:**
```bash
python main_multiagent.py --mode evaluate --eval-episodes 50 --load-finetuned
```

**Cooperative:**
```bash
python main_multiagent.py --mode evaluate --cooperative --eval-episodes 50 --load-finetuned
```

### Visual Demonstration (SUMO-GUI)

```bash
python main_multiagent.py --mode evaluate --gui --eval-episodes 1 --load-finetuned
```

### Resume Training from Checkpoint

```bash
python main_multiagent.py --mode train --cooperative --episodes 320 --resume-from 380
```

---

## 📁 Project Structure

```
RL-Project-main/
│
├── agent.py                        # DDQN Agent implementation
├── network.py                      # Neural network architecture
├── replay_buffer.py                # Experience replay mechanism
├── sumo_environment.py             # Single-agent SUMO wrapper
├── sumo_environment_multiagent.py  # Multi-agent SUMO wrapper (4 intersections)
├── sumo_environment_federated.py   # Federated SUMO wrapper (8 intersections) 🆕
│
├── supervisor_agent.py             # Zone supervisor DDQN agent 🆕
├── federated_learning.py           # FedAvg implementation 🆕
│
├── main.py                         # Single-agent training/evaluation
├── main_multiagent.py              # Multi-agent training/evaluation
├── main_federated.py               # Federated system entry point 🆕
│
├── train.py                        # Single-agent training loop
├── train_federated.py              # Hierarchical training loop 🆕
│
├── evaluate.py                     # Single/multi-agent evaluation
├── evaluate_federated.py           # Federated evaluation plots 🆕
│
├── generate_sumo_files.py          # Single intersection generator
├── generate_sumo_multiagent.py     # 2×2 grid generator
├── generate_sumo_federated.py      # 4×2 grid generator 🆕
│
├── sumo_config/
│   ├── single_intersection/        # 1-intersection SUMO files
│   ├── multi_intersection/         # 2×2 grid SUMO files
│   └── federated/                  # 4×2 grid SUMO files 🆕
│
├── checkpoints/
│   ├── single_agent/               # Single-agent models
│   ├── multiagent/                 # Independent fine-tuned models
│   ├── cooperative/                # Cooperative models
│   └── federated/                  # Federated hierarchical models 🆕
│
├── results/
│   ├── single_agent/               # Single-agent results
│   ├── multiagent/                 # Multi-agent results
│   ├── cooperative/                # Cooperative results
│   └── federated/                  # Federated training results 🆕
│
├── docs/                           # Reports and documentation
├── scripts/                        # Utility scripts
└── README.md                       # This file
```

---

## 🎓 Training Details

### Single-Agent Pre-Training (Episode 900)

- **Duration**: 1000 episodes (Episode 900 selected as best)
- **Learning Rate**: 0.0005
- **Epsilon Decay**: 0.995 per episode
- **Replay Buffer**: 50,000 experiences
- **Batch Size**: 64
- **Target Update**: Every 10 episodes
- **Hardware**: NVIDIA RTX 2050 (4.29GB VRAM)

### Multi-Agent Fine-Tuning (Independent)

- **Initial Weights**: Episode 900 (pretrained)
- **Duration**: 100 episodes
- **Learning Rate**: 0.0001 (gentle fine-tuning)
- **Epsilon Start**: 0.1 (low exploration)
- **Training Time**: ~40 minutes

### Multi-Agent Training (Cooperative)

- **Initial Weights**: Random (dimension mismatch with pretrained)
- **Duration**: 700 episodes
- **Learning Rate**: 0.0005 (from scratch)
- **Epsilon Start**: 0.9 (high exploration)
- **Training Time**: ~5 hours
- **Checkpoints**: Saved every 20 episodes

![Training Efficiency](readme_visuals/5_training_efficiency.png)
*Figure 5: Training time comparison showing the advantage of transfer learning*

---

## 📈 Evaluation

### Metrics Tracked

- **Episode Reward**: Cumulative reward per episode
- **Average Waiting Time**: Mean vehicle waiting time (seconds)
- **Queue Length**: Number of halted vehicles per edge
- **Phase Switches**: Number of traffic light changes
- **Network Balance**: Load distribution across intersections

### Statistical Analysis

All evaluations performed over 50 episodes to ensure reliability:
- Mean performance
- Per-intersection breakdown
- Network-level metrics
- Comparison with fixed-time baseline

### Baseline Comparison

**Fixed-Time Controller:**
- 60-second cycles (30s NS, 30s EW)
- Average waiting time: 141.0s
- Average queue: 11.0 vehicles

**DDQN Performance:**
- 94.3% reduction in waiting time
- 81.8% reduction in queue length
- Adaptive phase switching based on traffic conditions

---

## 💡 Technical Highlights

### 1. Transfer Learning Success

The Episode 900 checkpoint demonstrates remarkable generalization:
- Trained on single intersection
- Deployed to 4 intersections without modification
- Achieved -1,363.1 avg reward immediately
- Proves state representation is universal

### 2. Fine-Tuning Efficiency

Only 100 episodes needed to achieve 58.9% improvement:
- Leverages pretrained knowledge
- Adapts to network-specific traffic patterns
- 7× faster than training from scratch
- Demonstrates value of transfer learning

### 3. Cooperative Coordination

Agents successfully learn network-level optimization:
- Perfect load balancing achieved
- All intersections perform identically
- Demonstrates true multi-agent learning
- Network equilibrium discovered

### 4. GPU Acceleration

Efficient training with CUDA:
- 4.29GB VRAM sufficient for 4 agents
- ~25 seconds per episode
- Parallel neural network updates
- Real-time decision making

---

## 🔬 Research Contributions

### 1. Multi-Agent Scalability

Demonstrated successful scaling from 1→4 intersections:
- 68% better per-intersection performance
- Linear scalability potential
- Foundation for larger networks

### 2. Transfer Learning Methodology

Proven approach for multi-agent deployment:
- Single-agent pretraining
- Multi-agent transfer
- Targeted fine-tuning
- Results-driven efficiency

### 3. Independent vs Cooperative Analysis

Comprehensive comparison providing insights:
- Independent: Better average performance
- Cooperative: Perfect load balancing
- Trade-offs quantified
- Application-specific recommendations

---

## 🎯 Future Work

### ✅ Completed

1. **Federated Hierarchical MARL** (8 intersections, 2 supervisors, FedAvg)
2. **Hierarchical Control** (local agents + zone supervisors)
3. **Communication Protocols** (supervisor coordination signals)

### Short-Term Enhancements

1. **Stochastic Traffic Patterns**
   - Random vehicle spawn times
   - Variable flow rates
   - More realistic scenarios

2. **Even Larger Networks**
   - 4×4 grid (16 intersections)
   - Real city topology
   - Scalability testing

### Long-Term Goals

1. **Real-World Deployment**
   - Integration with actual traffic systems
   - Real-time data feeds
   - Adaptive learning in production

2. **Multi-Objective Optimization**
   - Minimize emissions
   - Prioritize emergency vehicles
   - Pedestrian safety

3. **Advanced RL Algorithms**
   - Multi-Agent PPO
   - QMIX for value decomposition
   - Graph Neural Networks for topology

---

## 📚 References

- [SUMO Traffic Simulator](https://www.eclipse.org/sumo/)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 👥 Acknowledgments

- Original single-agent training (1000 episodes) by teammate
- SUMO development team for excellent traffic simulation tools
- PyTorch community for deep learning framework

---

## 📜 License

This project is for academic and research purposes.

---

## 📧 Contact

For questions or collaboration opportunities, please reach out through the repository.

---

**Last Updated**: March 2025  
**Status**: Federated Hierarchical System Implemented ✅
