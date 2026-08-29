# 🛡️ Multi-Agent Traffic Control: Security Phase Extension

## 1. Problem Statement: The Threat of False Data Injection (FDI)
In modern smart city infrastructure, traffic light sensors (like induction loops or cameras) send real-time vehicle queue data to central AI controllers over networks. However, these networks are vulnerable to cyberattacks.

In this extension, we simulate a **False Data Injection (FDI)** attack. An attacker intercepts the communication between the SUMO sensors and the DDQN agents, injecting falsified (massively inflated) vehicle queue numbers. 
* **The Impact:** The agent trusts this fake data, switches phases unnecessarily, builds massive real-world congestion, and corrupts the 24-dimensional state passed upward to the Local/Global Supervisor agents, breaking hierarchical coordination.

## 2. The Defense Architecture: Two-Layer Security System

To combat this, we engineered a robust, automated two-layer defense mechanism that intercepts all state arrays before they reach the decision-making RL algorithms.

### Layer 1: Statistical Watchman (Anomaly Detector)
We avoid hardcoded thresholds because traffic naturally fluctuates between quiet nights and busy rush hours. Instead, we use a **Rolling Window Z-Score** approach:
* The system maintains a sliding window of the last 20 clean queue readings for every lane at every intersection.
* It continuously updates the `mean` and `standard deviation` of this window.
* When a new reading arrives, it calculates the Z-Score: `z = |value - mean| / std`.
* If `z > 3.0`, the system flags the data point as a malicious anomaly.

### Layer 2: Time-Series Recovery (LSTM Predictor)
Simply dropping anomalous data or replacing it with the mean is insufficient for realistic traffic, which follows time-series trends (e.g., a real traffic jam building up versus an instantaneous hacker spike).
* We trained a **Long Short-Term Memory (LSTM)** neural network on clean baseline traffic data.
* When Layer 1 flags an index as malicious, Layer 2 steps in. It feeds the recent 20-step history into the pre-trained LSTM, which predicts what the queue length *should* be in reality.
* The system replaces the poisoned value with the LSTM's prediction and passes the cleansed state vector back into the RL agent.

## 3. Experimental Scenarios

To quantitatively prove the effectiveness of this defense, we implemented an automated test runner (`main_security.py`) that evaluates the 8-intersection hierarchy under 5 distinct conditions:

1. **`baseline`**: Normal system, no attacks, no defense. Represents the theoretical maximum performance.
2. **`attack`**: FDI attack active, no defense. Demonstrates the catastrophic damage hackers can cause to the network.
3. **`defense`**: FDI attack active + anomaly detector + LSTM correction. Demonstrates the mathematical recovery provided by our security layer.
4. **`unreliable`**: No attack, just realistic network packet loss and zero-order hold delays. Tests system resilience to normal hardware degradation.
5. **`secure`**: Full FDI attack + full defense + slight network noise. The most realistic modern deployment scenario.

## 4. Analytical Visualization
We built a custom analysis suite (`analyze_security.py`) that consumes the CSV outputs of the 5 scenarios and generates presentation-ready Matplotlib charts:
* **Average Network Reward**: Proves how close the `defense` scenario recovers toward the `baseline`.
* **Average Wait Time**: Shows the real-world reduction in vehicle delay.
* **Per-Intersection Breakdown**: Granular view of the supervisor coordination.
* **LSTM Detection Accuracy**: Validates the Z-score and Mean Absolute Error (MAE) of the LSTM's predictions against the mathematical truth.

## 5. Conclusion
This phase successfully proves that Deep Reinforcement Learning traffic controllers can be hardened against sophisticated cyberattacks without requiring the RL algorithm itself to be retrained. By intercepting and cleansing state data using statistical and predictive modeling, the system maintains hierarchical coordination even under active attack.
