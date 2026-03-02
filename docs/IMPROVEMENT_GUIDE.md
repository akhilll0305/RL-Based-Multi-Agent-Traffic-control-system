# DDQN Traffic Light Agent Improvement Guide

## Current Performance Baseline (500 Episodes)
- Average Reward: ~-6,850
- Waiting Time: 0.21s
- Queue Length: 0.50-0.75
- Training Time: ~67 minutes

---

## Improvement Strategies (Ranked by Impact)

### 🔥 HIGH IMPACT (Try These First)

#### 1. **Adjust Reward Function Weights**
**Location:** `sumo_environment.py`, `_calculate_reward()` method

**Current:**
```python
reward = -(total_queue + 0.5 * avg_waiting_time + 10 * switching_penalty)
```

**Try These Variations:**
```python
# Option A: Penalize waiting time more
reward = -(total_queue + 2.0 * avg_waiting_time + 10 * switching_penalty)

# Option B: Reduce switching penalty (allow more flexibility)
reward = -(total_queue + 0.5 * avg_waiting_time + 5 * switching_penalty)

# Option C: Add throughput bonus
throughput = info.get('vehicles_passed', 0)
reward = -(total_queue + 0.5 * avg_waiting_time + 10 * switching_penalty) + 0.1 * throughput

# Option D: Exponential penalty for queues (punish congestion harder)
reward = -(total_queue**1.5 + 0.5 * avg_waiting_time + 10 * switching_penalty)
```

**Impact:** Can improve by 20-40% with proper tuning

---

#### 2. **Increase Training Episodes**
**Location:** Command line argument

**Current:** 500 episodes

**Try:**
```powershell
# Longer training (2-3 hours)
python main.py --mode train --episodes 1000

# Extended training (4-6 hours)
python main.py --mode train --episodes 2000
```

**Impact:** Usually converges better, +10-20% improvement

---

#### 3. **Improve State Representation**
**Location:** `sumo_environment.py`, `_get_state()` method

**Current State (6 features):**
- queue_north, queue_south, queue_east, queue_west
- current_phase
- time_since_last_change

**Enhanced State (10-12 features):**
```python
def _get_state(self):
    # Existing queues
    queues = [
        self.traci.edge.getLastStepHaltingNumber('north_in'),
        self.traci.edge.getLastStepHaltingNumber('south_in'),
        self.traci.edge.getLastStepHaltingNumber('east_in'),
        self.traci.edge.getLastStepHaltingNumber('west_in')
    ]
    
    # NEW: Average speeds (traffic flow indicator)
    speeds = [
        self.traci.edge.getLastStepMeanSpeed('north_in') / 13.89,  # Normalize by max speed
        self.traci.edge.getLastStepMeanSpeed('south_in') / 13.89,
        self.traci.edge.getLastStepMeanSpeed('east_in') / 13.89,
        self.traci.edge.getLastStepMeanSpeed('west_in') / 13.89
    ]
    
    # NEW: Vehicle counts (approaching traffic)
    vehicle_counts = [
        self.traci.edge.getLastStepVehicleNumber('north_in'),
        self.traci.edge.getLastStepVehicleNumber('south_in'),
        self.traci.edge.getLastStepVehicleNumber('east_in'),
        self.traci.edge.getLastStepVehicleNumber('west_in')
    ]
    
    # Current phase and timing
    current_phase = self.current_phase
    time_since_change = min(self.time_since_last_change / 60.0, 1.0)  # Normalize
    
    state = queues + speeds + [current_phase, time_since_change]
    
    return np.array(state, dtype=np.float32)
```

**Update STATE_DIM in main.py:** Change from 6 to 10

**Impact:** Better decision-making, +15-30% improvement

---

### ⚡ MEDIUM IMPACT

#### 4. **Tune Learning Rate**
**Location:** `main.py`, hyperparameters section

**Current:** 0.001

**Try:**
```python
# Slower, more stable learning
LEARNING_RATE = 0.0005

# Faster learning (may be unstable)
LEARNING_RATE = 0.002

# Learning rate decay (best of both worlds)
# Add to agent.py:
from torch.optim.lr_scheduler import StepLR
self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)
# Call after each training step: self.scheduler.step()
```

**Impact:** +5-15% improvement

---

#### 5. **Adjust Discount Factor (Gamma)**
**Location:** `main.py`

**Current:** 0.95

**Try:**
```python
# More forward-looking (considers long-term impact)
GAMMA = 0.99

# More short-sighted (good for immediate rewards)
GAMMA = 0.90
```

**Impact:** +5-10% improvement

---

#### 6. **Increase Network Capacity**
**Location:** `main.py` and `network.py`

**Current:** 128 hidden units, 2 layers

**Option A - Wider:**
```python
HIDDEN_DIM = 256  # Double the neurons
```

**Option B - Deeper:**
```python
# In network.py, add 3rd layer:
self.fc1 = nn.Linear(state_dim, hidden_dim)
self.fc2 = nn.Linear(hidden_dim, hidden_dim)
self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # NEW
self.fc4 = nn.Linear(hidden_dim, action_dim)

def forward(self, state):
    x = torch.relu(self.fc1(state))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))  # NEW
    q_values = self.fc4(x)
    return q_values
```

**Impact:** +10-15% improvement (but slower training)

---

#### 7. **Optimize Epsilon Decay**
**Location:** `main.py`

**Current:** decay=0.995, min=0.01

**Try:**
```python
# Slower exploration reduction (more learning time)
EPSILON_DECAY = 0.998
EPSILON_MIN = 0.05

# Faster convergence to exploitation
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
```

**Impact:** +5-10% improvement

---

### 🔧 LOW IMPACT (Fine-tuning)

#### 8. **Batch Size and Buffer Tuning**
```python
BATCH_SIZE = 128  # From 64 (more stable gradients)
BUFFER_CAPACITY = 50000  # From 10000 (more diverse experiences)
```

#### 9. **Target Network Update Frequency**
```python
TARGET_UPDATE_FREQ = 5  # From 10 (more frequent updates)
```

#### 10. **Simulation Parameters**
**Location:** `main.py`, SumoEnvironment initialization

```python
env = SumoEnvironment(
    net_file='sumo_files/intersection.net.xml',
    route_file='sumo_files/routes.rou.xml',
    use_gui=False,
    num_seconds=7200,  # From 3600 (longer episodes)
    delta_time=3  # From 5 (more frequent decisions)
)
```

---

## 🚀 Advanced Techniques (For Major Improvements)

### 11. **Prioritized Experience Replay**
Samples important experiences more frequently. Can give +20-30% improvement.

### 12. **Dueling DQN Architecture**
Separates value and advantage streams. Better for traffic control.

### 13. **Multi-Step Returns (N-step DQN)**
Looks ahead multiple steps for better long-term planning.

### 14. **Noisy Networks**
Replaces epsilon-greedy with learned exploration.

---

## 📋 Recommended Testing Protocol

### Step 1: Test Current Baseline
```powershell
python main.py --mode evaluate --model-path models/ddqn_traffic_final.pth --eval-episodes 100
```

### Step 2: Try One Change at a Time
```powershell
# Example: Test reward function change
# 1. Modify sumo_environment.py reward function
# 2. Train new model
python main.py --mode train --episodes 500
# 3. Rename model to avoid overwriting
mv models/ddqn_traffic_final.pth models/ddqn_reward_v2.pth
# 4. Evaluate
python main.py --mode evaluate --model-path models/ddqn_reward_v2.pth --eval-episodes 100
```

### Step 3: Compare Results
Keep a log file with:
- Modification made
- Average reward
- Average waiting time
- Average queue length
- Training time

### Step 4: Combine Best Changes
Once you find improvements, combine them carefully.

---

## 🎯 Quick Win Recommendations (Try in Order)

1. **Increase episodes to 1000** (2 hours training) - Easy, consistent improvement
2. **Adjust reward weights** - Test 3-4 variations, pick best
3. **Add speed features to state** - Better traffic awareness
4. **Increase HIDDEN_DIM to 256** - More learning capacity
5. **Use GAMMA=0.99** - Better long-term planning

---

## 📊 Expected Performance Targets

| Metric | Current | Good | Excellent |
|--------|---------|------|-----------|
| Avg Reward | -6,850 | -5,000 | -3,500 |
| Waiting Time | 0.21s | 0.10s | 0.05s |
| Queue Length | 0.50 | 0.30 | 0.15 |
| vs Fixed-Time | TBD | 30% better | 50% better |

---

## ⚠️ Important Notes

1. **Save each experiment separately:**
   ```python
   agent.save('models/ddqn_experiment_1.pth')
   agent.save('models/ddqn_experiment_2.pth')
   # etc.
   ```

2. **Keep training_history.csv for each run:**
   ```powershell
   cp results/training_history.csv results/training_history_experiment_1.csv
   ```

3. **One change at a time** - Otherwise you won't know what helped!

4. **Use same seed for fair comparison:**
   ```powershell
   python main.py --mode train --episodes 500 --seed 42
   ```

5. **Always evaluate on same number of episodes** (e.g., 100) for fair comparison.
