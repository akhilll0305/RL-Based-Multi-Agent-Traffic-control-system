"""
Security layer for traffic-state protection in security-phase experiments.

Step 5.1 scope:
- Define SecurityLayer class and initialization contract
- Accept all tunable parameters for attack/defense/network reliability
- Initialize rolling history, last-known state, and delay buffers
- Load pre-trained LSTM predictor checkpoint

Note:
- Full behavior routing and algorithmic processing is implemented in Step 5.2+
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from traffic_rl.security.lstm import TrafficLSTM


SUPPORTED_MODES = {"baseline", "attack", "defense", "unreliable", "secure"}
DEFENSE_MODES = {"defense", "secure"}


@dataclass
class SecurityLayerConfig:
    """Container for security layer tunable parameters."""

    mode: str = "baseline"
    fdi_prob: float = 0.15
    fdi_min: float = 10.0
    fdi_max: float = 15.0
    window_size: int = 20
    z_threshold: float = 3.0
    packet_loss_prob: float = 0.05
    max_delay_steps: int = 3
    lstm_checkpoint: str = "outputs/checkpoints/security/lstm_predictor.pth"
    seed: int = 42


class SecurityLayer:
    """
    Security wrapper for raw local traffic states.

    Input format:
      {tls_id: np.ndarray shape (6,)}

    Output format:
      {tls_id: np.ndarray shape (6,)}

    Design constraints:
    - Queue features are indices [0..3]
    - Phase/timer features are indices [4..5] and must remain untouched by attack/defense
    """

    def __init__(
        self,
        mode: str = "baseline",
        fdi_prob: float = 0.15,
        fdi_min: float = 10.0,
        fdi_max: float = 15.0,
        window_size: int = 20,
        z_threshold: float = 3.0,
        packet_loss_prob: float = 0.05,
        max_delay_steps: int = 3,
        lstm_checkpoint: str = "outputs/checkpoints/security/lstm_predictor.pth",
        seed: int = 42,
    ) -> None:
        self.config = SecurityLayerConfig(
            mode=mode,
            fdi_prob=fdi_prob,
            fdi_min=fdi_min,
            fdi_max=fdi_max,
            window_size=window_size,
            z_threshold=z_threshold,
            packet_loss_prob=packet_loss_prob,
            max_delay_steps=max_delay_steps,
            lstm_checkpoint=lstm_checkpoint,
            seed=seed,
        )

        self._validate_config()

        self.rng = np.random.default_rng(seed=self.config.seed)

        # Initialized on first process(...) call based on observed tls_ids.
        self._tls_ids: Optional[List[str]] = None

        # Rolling queue history: tls_id -> deque of queue vectors shape (4,)
        self.history_buffers: Dict[str, Deque[np.ndarray]] = {}

        # Last successfully delivered raw state: tls_id -> np.ndarray shape (6,)
        self.last_known_states: Dict[str, np.ndarray] = {}

        # Delay buffers for unreliable mode:
        # tls_id -> deque of tuples (deliver_step, state_vector)
        self.delay_buffers: Dict[str, Deque[tuple[int, np.ndarray]]] = {}

        # Event logs and counters
        self.attack_log: List[dict] = []
        self.detection_log: List[dict] = []
        self.false_positive_count: int = 0

        self.total_attack_events: int = 0
        self.total_detection_events: int = 0

        self.lstm_model: Optional[TrafficLSTM] = self._load_lstm_model()

    def _validate_config(self) -> None:
        """Validate mode and numeric parameter ranges."""
        c = self.config

        if c.mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode '{c.mode}'. Allowed: {sorted(SUPPORTED_MODES)}")

        if not (0.0 <= c.fdi_prob <= 1.0):
            raise ValueError("fdi_prob must be in [0, 1]")

        if c.fdi_min < 0 or c.fdi_max < 0 or c.fdi_min > c.fdi_max:
            raise ValueError("FDI range must satisfy 0 <= fdi_min <= fdi_max")

        if c.window_size < 2:
            raise ValueError("window_size must be >= 2")

        if c.z_threshold <= 0:
            raise ValueError("z_threshold must be > 0")

        if not (0.0 <= c.packet_loss_prob <= 1.0):
            raise ValueError("packet_loss_prob must be in [0, 1]")

        if c.max_delay_steps < 0:
            raise ValueError("max_delay_steps must be >= 0")

    def _load_lstm_model(self) -> Optional[TrafficLSTM]:
        """Load LSTM checkpoint according to mode requirements."""
        ckpt = self.config.lstm_checkpoint

        if os.path.exists(ckpt):
            return TrafficLSTM.load(ckpt)

        if self.config.mode in DEFENSE_MODES:
            raise FileNotFoundError(
                f"LSTM checkpoint required for mode '{self.config.mode}' but not found: {ckpt}"
            )

        return None

    def _initialize_state_buffers(self, raw_states: Dict[str, np.ndarray]) -> None:
        """Initialize per-intersection buffers on first observation."""
        self._tls_ids = list(raw_states.keys())

        for tls in self._tls_ids:
            state = np.array(raw_states[tls], dtype=np.float32).copy()
            if state.shape != (6,):
                raise ValueError(f"State for {tls} must have shape (6,), got {state.shape}")

            self.history_buffers[tls] = deque(maxlen=self.config.window_size)
            self.last_known_states[tls] = state.copy()
            self.delay_buffers[tls] = deque()

    def _validate_input_states(self, raw_states: Dict[str, np.ndarray]) -> None:
        """Validate incoming raw state dict format."""
        if not isinstance(raw_states, dict) or not raw_states:
            raise ValueError("raw_states must be a non-empty dict: {tls_id: np.ndarray(6,)}")

        for tls, state in raw_states.items():
            arr = np.array(state)
            if arr.shape != (6,):
                raise ValueError(f"Invalid state shape for {tls}: expected (6,), got {arr.shape}")

    def _copy_states(self, raw_states: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Return deep-copied state dict to preserve caller immutability."""
        return {tls: np.array(state, dtype=np.float32).copy() for tls, state in raw_states.items()}

    def process(self, raw_states: Dict[str, np.ndarray], step: int) -> Dict[str, np.ndarray]:
        """
        Entry point for state processing.

        Step 5.2 behavior:
        - validates and initializes internal buffers
        - routes by security mode
        - guarantees output format {tls_id: np.ndarray(6,)}
        """
        self._validate_input_states(raw_states)

        if self._tls_ids is None:
            self._initialize_state_buffers(raw_states)

        states = self._copy_states(raw_states)

        if self.config.mode == "baseline":
            output = states
            self._update_history_buffers(output)

        elif self.config.mode == "attack":
            attacked = self._apply_fdi_attack(states, step)
            output = attacked
            self._update_history_buffers(output)

        elif self.config.mode == "defense":
            attacked = self._apply_fdi_attack(states, step)
            output = self._detect_and_correct(attacked, step)

        elif self.config.mode == "unreliable":
            output = self._apply_network_unreliability(states, step)
            self._update_history_buffers(output)

        elif self.config.mode == "secure":
            attacked = self._apply_fdi_attack(states, step)
            output = self._detect_and_correct(attacked, step)

        else:
            raise ValueError(f"Unhandled mode: {self.config.mode}")

        # Final guard: keep strict output format.
        self._validate_input_states(output)
        return output

    def _update_history_buffers(self, states: Dict[str, np.ndarray]) -> None:
        """Append current queue vectors (indices 0..3) to rolling history buffers."""
        for tls, state in states.items():
            self.history_buffers[tls].append(np.array(state[:4], dtype=np.float32).copy())

    def _apply_fdi_attack(self, states: Dict[str, np.ndarray], step: int) -> Dict[str, np.ndarray]:
        """Apply probabilistic false-data-injection attack on queue indices 0..3."""
        attacked = self._copy_states(states)

        for tls in attacked:
            if self.rng.random() > self.config.fdi_prob:
                continue

            lane_idx = int(self.rng.integers(0, 4))
            injection = float(self.rng.uniform(self.config.fdi_min, self.config.fdi_max))

            old_val = float(attacked[tls][lane_idx])
            new_val = float(np.clip(old_val + injection, 0.0, 20.0))
            attacked[tls][lane_idx] = new_val

            event = {
                "step": int(step),
                "tls_id": tls,
                "lane_idx": lane_idx,
                "old_value": old_val,
                "new_value": new_val,
                "injection": injection,
            }
            self.attack_log.append(event)
            self.total_attack_events += 1

        return attacked

    def _detect_and_correct(self, states: Dict[str, np.ndarray], step: int) -> Dict[str, np.ndarray]:
        """Detect anomalous queue values and correct flagged lanes with LSTM prediction."""
        corrected = self._copy_states(states)

        for tls in corrected:
            queue_now = np.array(corrected[tls][:4], dtype=np.float32)
            hist = self.history_buffers[tls]

            # Need sufficient history before z-score detection.
            min_history = min(10, self.config.window_size)
            if len(hist) < min_history:
                hist.append(queue_now.copy())
                continue

            history_arr = np.stack(list(hist), axis=0)  # (k, 4)
            mean = history_arr.mean(axis=0)
            std = history_arr.std(axis=0) + 1e-6
            z = np.abs(queue_now - mean) / std

            flagged = np.where(z > self.config.z_threshold)[0]

            if flagged.size > 0:
                if self.lstm_model is not None:
                    # Use most recent window_size history for prediction.
                    if len(hist) >= self.config.window_size:
                        seq = np.stack(list(hist)[-self.config.window_size :], axis=0)
                    else:
                        seq = history_arr
                    pred = self.lstm_model.predict(seq)

                    old_values = queue_now.copy()
                    for lane in flagged:
                        corrected[tls][lane] = float(np.clip(pred[lane], 0.0, 20.0))
                else:
                    old_values = queue_now.copy()

                event = {
                    "step": int(step),
                    "tls_id": tls,
                    "flagged_lanes": flagged.astype(int).tolist(),
                    "z_scores": z.astype(float).tolist(),
                    "before": old_values.astype(float).tolist(),
                    "after": np.array(corrected[tls][:4], dtype=np.float32).astype(float).tolist(),
                }
                self.detection_log.append(event)
                self.total_detection_events += 1
                
                # Visual Confirmation Logging (Step 8.2)
                # We identify all flagged lanes and print exactly what was blocked
                for lane_idx in np.where(flagged)[0]:
                    print(f"\n[DEFENSE] 🚨 Anomaly detected at {tls}! Lane: {lane_idx}")
                    print(f"    Value: {old_values[lane_idx]:.1f} (Z-score: {z[lane_idx]:.2f}) -> Replaced with LSTM prediction: {corrected[tls][lane_idx]:.1f}")
                

            hist.append(np.array(corrected[tls][:4], dtype=np.float32).copy())

        return corrected

    def _apply_network_unreliability(self, states: Dict[str, np.ndarray], step: int) -> Dict[str, np.ndarray]:
        """Apply packet loss + bounded delay and return delivered states."""
        output = {}

        for tls in states:
            incoming = np.array(states[tls], dtype=np.float32).copy()

            # Packet loss: reuse last known state.
            if self.rng.random() < self.config.packet_loss_prob:
                output[tls] = self.last_known_states[tls].copy()
                continue

            # Delay insertion for non-lost packets.
            delay_steps = 0
            if self.config.max_delay_steps > 0:
                delay_steps = int(self.rng.integers(0, self.config.max_delay_steps + 1))

            if delay_steps > 0:
                deliver_step = int(step + delay_steps)
                self.delay_buffers[tls].append((deliver_step, incoming.copy()))

                # If nothing due yet, serve last known state.
                if not self.delay_buffers[tls] or self.delay_buffers[tls][0][0] > step:
                    output[tls] = self.last_known_states[tls].copy()
                    continue

            # Deliver the most recent packet due at current step.
            delivered = None
            while self.delay_buffers[tls] and self.delay_buffers[tls][0][0] <= step:
                _, delivered_state = self.delay_buffers[tls].popleft()
                delivered = delivered_state

            if delivered is None:
                delivered = incoming

            output[tls] = np.array(delivered, dtype=np.float32).copy()
            self.last_known_states[tls] = output[tls].copy()

        return output

    def get_metrics(self) -> Dict[str, float]:
        """
        Return current security-layer metrics for experiment reporting.

        Keys:
          total_attack_events
          total_detection_events
          false_positive_count
          detection_rate
          false_positive_rate
        """
        detection_rate = (
            float(self.total_detection_events) / float(self.total_attack_events)
            if self.total_attack_events > 0
            else 0.0
        )

        clean_states_seen = max(0, self.total_detection_events + self.false_positive_count)
        false_positive_rate = (
            float(self.false_positive_count) / float(clean_states_seen)
            if clean_states_seen > 0
            else 0.0
        )

        return {
            "total_attack_events": float(self.total_attack_events),
            "total_detection_events": float(self.total_detection_events),
            "false_positive_count": float(self.false_positive_count),
            "detection_rate": float(detection_rate),
            "false_positive_rate": float(false_positive_rate),
        }
