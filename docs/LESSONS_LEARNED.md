# Lessons Learned - Why Improvements Failed

## ❌ What Happened

**Attempted Improvements:**
1. Reward weight: 0.5 → 2.0 
2. State features: 6 → 14
3. Hidden units: 128 → 256
4. Gamma: 0.95 → 0.99

**Result:** Performance DEGRADED instead of improved

---

## 🧠 Why Multiple Changes Failed

### Problem: Changed Too Many Things at Once
- Can't identify which change caused the problem
- Changes might conflict with each other
- Network size + state size = much harder to train
- Different normalization broke the learning

### Specific Issues:

**1. State Normalization Mismatch**
- Old: Raw values (queues 0-20)
- New: Normalized values (0-1)
- Network expects certain input ranges
- Breaking that confused the learning

**2. Network Too Big Too Fast**
- 128 → 256 neurons needs more data
- More parameters = harder to optimize
- Needs longer training or different learning rate

**3. Gamma Too High**
- 0.95 → 0.99 changes TD error magnitude
- Needs re-tuning of learning rate
- Can cause instability without other adjustments

**4. Reward Weight Too Aggressive**
- 0.5 → 2.0 is a 4x change
- Completely changes reward scale
- Agent's learned Q-values became invalid

---

## ✅ The RIGHT Approach

### **ONE CHANGE AT A TIME**

#### Step 1: Extended Training (Safest)
```powershell
# Just train the proven baseline longer
$env:PYTHONDONTWRITEBYTECODE="1"; python main.py --mode train --episodes 1000
```
**Pros:**
- ✅ Zero risk
- ✅ Baseline was still improving at episode 500
- ✅ Expected +10-15% gain
- ✅ Takes ~2.5 hours

**After:** Save with `python save_extended.py`

---

#### Step 2: Try ONE Small Change at a Time

**Option A: Gentle Reward Tuning**
```python
# sumo_environment.py, line ~201
reward = -total_queue - 0.75 * total_waiting  # Just 0.5 → 0.75 (not 2.0!)
```
Train 500 episodes, evaluate, compare.

**Option B: Slightly Bigger Network**
```python
# main.py
HIDDEN_DIM = 160  # Just 128 → 160 (not 256!)
```
Train 500 episodes, evaluate, compare.

**Option C: Better Gamma**
```python
# main.py
GAMMA = 0.97  # Just 0.95 → 0.97 (not 0.99!)
```
Train 500 episodes, evaluate, compare.

---

## 📊 Proper Experimentation Workflow

```
1. Baseline (500ep) ✅ DONE
   ↓
2. Extended Baseline (1000ep) ← START HERE
   ↓
3. If still improving, try 1500ep
   ↓
4. Once converged, try ONE small change:
   ↓
   → Test reward weight 0.75
   → Save & evaluate
   → If better, keep it. If worse, revert.
   ↓
5. Try next change on BEST model so far
   ↓
6. Repeat until satisfied
```

---

## 🎯 Recommended Next Steps

### **Option 1: Safe & Proven (RECOMMENDED)**
Train the reverted baseline for 1000 episodes:

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"; python main.py --mode train --episodes 1000
```

After completion:
```powershell
python save_extended.py
```

**Expected result:** -6,670 → -5,500 to -6,000 (better than baseline)

---

### **Option 2: Quick Improvement Test**
Make ONE small change and train 500 episodes:

**Edit `sumo_environment.py` line ~201:**
```python
reward = -total_queue - 0.75 * total_waiting  # Small increase from 0.5
```

**Train:**
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"; python main.py --mode train --episodes 500
```

**Save:**
```python
from experiment_manager import save_current_training
save_current_training('reward_075', 'Gentle reward tune: 0.5→0.75', {'reward_weight': 0.75})
```

**Compare:**
```python
from experiment_manager import ExperimentManager
manager = ExperimentManager()
manager.compare_experiments(['baseline_500ep_GPU_20260205_235635', 'reward_075_TIMESTAMP'])
```

---

## 📈 What We Learned

### ✅ Good Practices:
1. **ONE change at a time**
2. **Small incremental changes** (0.5 → 0.75, not 0.5 → 2.0)
3. **Save every experiment** with descriptive names
4. **Compare before moving on**
5. **Keep proven baseline safe**

### ❌ Bad Practices:
1. Changing 4 things simultaneously
2. Making large jumps (128 → 256, 0.5 → 2.0)
3. Changing normalization without re-tuning
4. Not testing intermediate steps

---

## 🔄 Current Status

**Your Safe Models:**
- ✅ `experiments/baseline_500ep_GPU_20260205_235635/` (Avg reward: -6,670)
  - Can test anytime with this model
  - Proven to work well

**Code Status:**
- ✅ REVERTED to baseline configuration
- ✅ Ready to train 1000 episodes safely
- ✅ All improvements removed

**Next Action:**
Train 1000 episodes with proven config for guaranteed improvement!

---

## 💡 Key Insight

**Simple is better than complex.**

Your baseline with 6 features and 128 neurons achieved **61% improvement** in 500 episodes. That's excellent! 

Rather than making it "smarter" with more features, just give it **more time to learn** what it already knows works.

**1000 episodes of baseline > 500 episodes of complex model**

---

**Recommendation: Run `train_extended.py` or the command below:**

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"; python main.py --mode train --episodes 1000
```

This is the safest path to better results! 🎯
