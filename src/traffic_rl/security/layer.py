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


SUPPORTED_MODES = {"baseline", "guard", "attack", "defense", "unreliable", "secure"}
# guard = the detector with no attack: what its false alarms cost on clean data
DEFENSE_MODES = {"guard", "defense", "secure"}

# random         : one lane gets +U(fdi_min, fdi_max) with probability fdi_prob per step
# detector_aware : white-box; raises one lane to just below the z-score threshold
#                  (mean + aware_margin * theta_z * std of the detector's own window)
# ramp           : a campaign of campaign_steps on one lane, injection growing linearly
#                  from 0 to fdi_max, so the detector's history drifts with it
# replay         : a campaign that replays the true queue vectors recorded campaign_steps
#                  earlier (all 4 lanes), i.e. a stale but internally consistent state
# hide           : a campaign of campaign_steps that reports 0 vehicles on the lane with the
#                  longest queue when it starts, so the controller stops serving that approach
ATTACK_TYPES = {"random", "detector_aware", "ramp", "replay", "hide"}


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
    attack_type: str = "random"
    campaign_steps: int = 20
    aware_margin: float = 0.95
    std_floor: float = 1e-6
    min_deviation: float = 0.0     # also require |value - window mean| >= this many vehicles
    persistence: int = 1           # correct a lane only after this many flagged steps in a row
    detector: str = "zscore"       # "zscore" (window z-score) or "residual" (|reading - LSTM forecast|)
    residual_threshold: float = 5.0  # vehicles, for detector="residual"
    spike_rollback: bool = False   # residual detector: erase one-step spikes from the history
    verbose: bool = False


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
        attack_type: str = "random",
        campaign_steps: int = 20,
        aware_margin: float = 0.95,
        std_floor: float = 1e-6,
        min_deviation: float = 0.0,
        persistence: int = 1,
        detector: str = "zscore",
        residual_threshold: float = 5.0,
        spike_rollback: bool = False,
        verbose: bool = False,
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
            attack_type=attack_type,
            campaign_steps=campaign_steps,
            aware_margin=aware_margin,
            std_floor=std_floor,
            min_deviation=min_deviation,
            persistence=persistence,
            detector=detector,
            residual_threshold=residual_threshold,
            spike_rollback=spike_rollback,
            verbose=verbose,
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

        # Event logs and counters (cumulative over every episode this layer sees)
        self.attack_log: List[dict] = []
        self.detection_log: List[dict] = []
        self.false_positive_count: int = 0

        self.total_attack_events: int = 0
        self.total_detection_events: int = 0

        # Per-lane confusion matrix over every (step, tls, lane) the detector screens.
        # Ground truth = the attacker actually changed that lane's value this step.
        self.tp = self.fp = self.fn = self.tn = 0
        # (z-score, label) for every screened lane, for ROC curves over theta_z.
        # Warm-up steps (too little history) are scored z = 0, i.e. "not flagged".
        self.scores: List[float] = []
        self.labels: List[int] = []

        # Ground-truth mask of the current step: tls -> bool array (4,)
        self._attacked_mask: Dict[str, np.ndarray] = {}
        # Multi-step campaigns (ramp / replay): tls -> dict
        self._campaigns: Dict[str, dict] = {}
        # True queue vectors seen by the attacker, for replay
        self._true_history: Dict[str, Deque[np.ndarray]] = {}
        # Consecutive flagged steps per lane: tls -> int array (4,)
        self._streak: Dict[str, np.ndarray] = {}
        # Lanes flagged but not yet corrected last step, with their forecast: tls -> (mask, values)
        self._pending: Dict[str, tuple] = {}

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

        if c.attack_type not in ATTACK_TYPES:
            raise ValueError(f"attack_type must be one of {sorted(ATTACK_TYPES)}")

        if c.campaign_steps < 1:
            raise ValueError("campaign_steps must be >= 1")

        if c.min_deviation < 0 or c.persistence < 1:
            raise ValueError("min_deviation must be >= 0 and persistence >= 1")

        if c.detector not in ("zscore", "residual"):
            raise ValueError("detector must be 'zscore' or 'residual'")

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
            self._true_history[tls] = deque(maxlen=self.config.campaign_steps)
            self._streak[tls] = np.zeros(4, dtype=np.int32)
        self._campaigns = {}

    def reset(self) -> None:
        """Start a new episode: clear rolling buffers, keep cumulative counters."""
        self._tls_ids = None
        self.history_buffers, self.last_known_states, self.delay_buffers = {}, {}, {}
        self._true_history, self._campaigns, self._streak, self._pending = {}, {}, {}, {}

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

        # The envs encode queues as log1p(count). Attacks, detection and the LSTM work on the
        # car counts a detector would report, so decode here and encode again on the way out.
        states = self._copy_states(raw_states)
        for s in states.values():
            s[:4] = np.expm1(s[:4])

        if self._tls_ids is None:
            self._initialize_state_buffers(states)

        if self.config.mode == "baseline":
            output = states
            self._update_history_buffers(output)

        elif self.config.mode == "guard":
            self._attacked_mask = {}
            output = self._detect_and_correct(states, step)

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
            # attack + defence over the unreliable network: packets are lost or delayed on the way,
            # then the attacker alters what reaches the controller (so the attack mask stays exact)
            delivered = self._apply_network_unreliability(states, step)
            attacked = self._apply_fdi_attack(delivered, step)
            output = self._detect_and_correct(attacked, step)

        else:
            raise ValueError(f"Unhandled mode: {self.config.mode}")

        # Final guard: keep strict output format.
        self._validate_input_states(output)
        output = self._copy_states(output)
        for s in output.values():
            s[:4] = np.log1p(np.maximum(s[:4], 0.0))
        return output

    def _update_history_buffers(self, states: Dict[str, np.ndarray]) -> None:
        """Append current queue vectors (indices 0..3) to rolling history buffers."""
        for tls, state in states.items():
            self.history_buffers[tls].append(np.array(state[:4], dtype=np.float32).copy())

    def _apply_fdi_attack(self, states: Dict[str, np.ndarray], step: int) -> Dict[str, np.ndarray]:
        """Apply the configured false-data-injection attack to queue indices 0..3.

        Sets ``self._attacked_mask`` (ground truth for the detector's confusion matrix).
        """
        attacked = self._copy_states(states)
        attack = {
            "random": self._attack_random,
            "detector_aware": self._attack_detector_aware,
            "ramp": self._attack_ramp,
            "replay": self._attack_replay,
            "hide": self._attack_hide,
        }[self.config.attack_type]

        self._attacked_mask = {}
        for tls in attacked:
            true_q = np.array(states[tls][:4], dtype=np.float32)
            new_q = np.maximum(attack(tls, true_q.copy()), 0.0).astype(np.float32)
            self._true_history[tls].append(true_q)
            attacked[tls][:4] = new_q
            mask = np.abs(new_q - true_q) > 1e-6
            self._attacked_mask[tls] = mask
            for lane in np.flatnonzero(mask):
                self.attack_log.append({
                    "step": int(step),
                    "tls_id": tls,
                    "attack_type": self.config.attack_type,
                    "lane_idx": int(lane),
                    "old_value": float(true_q[lane]),
                    "new_value": float(new_q[lane]),
                    "injection": float(new_q[lane] - true_q[lane]),
                })
                self.total_attack_events += 1

        return attacked

    def _attack_random(self, tls: str, q: np.ndarray) -> np.ndarray:
        if self.rng.random() <= self.config.fdi_prob:
            lane = int(self.rng.integers(0, 4))
            q[lane] += float(self.rng.uniform(self.config.fdi_min, self.config.fdi_max))
        return q

    def _attack_detector_aware(self, tls: str, q: np.ndarray) -> np.ndarray:
        """Largest injection the z-score screen would still accept (white-box attacker)."""
        if self.rng.random() > self.config.fdi_prob:
            return q
        hist = self.history_buffers.get(tls)
        if hist is None or len(hist) < 2:
            return q
        h = np.stack(list(hist), axis=0)
        std = np.maximum(h.std(axis=0), self.config.std_floor)
        ceiling = h.mean(axis=0) + self.config.aware_margin * self.config.z_threshold * std
        gain = ceiling - q
        lane = int(np.argmax(gain))
        if gain[lane] > 0:
            q[lane] = ceiling[lane]
        return q

    def _start_campaign(self, tls: str) -> Optional[dict]:
        # Campaigns start at rate fdi_prob / campaign_steps, so the fraction of attacked
        # steps stays comparable to the random attack.
        camp = self._campaigns.get(tls)
        if camp is None and self.rng.random() <= self.config.fdi_prob / self.config.campaign_steps:
            camp = {"t": 0, "lane": int(self.rng.integers(0, 4))}
            if self.config.attack_type == "replay":
                if len(self._true_history[tls]) < self.config.campaign_steps:
                    return None
                camp["tape"] = list(self._true_history[tls])
            self._campaigns[tls] = camp
        return camp

    def _end_campaign_step(self, tls: str, camp: dict) -> None:
        camp["t"] += 1
        if camp["t"] >= self.config.campaign_steps:
            del self._campaigns[tls]

    def _attack_ramp(self, tls: str, q: np.ndarray) -> np.ndarray:
        camp = self._start_campaign(tls)
        if camp is not None:
            q[camp["lane"]] += self.config.fdi_max * (camp["t"] + 1) / self.config.campaign_steps
            self._end_campaign_step(tls, camp)
        return q

    def _attack_replay(self, tls: str, q: np.ndarray) -> np.ndarray:
        camp = self._start_campaign(tls)
        if camp is not None:
            q = np.array(camp["tape"][camp["t"]], dtype=np.float32)
            self._end_campaign_step(tls, camp)
        return q

    def _attack_hide(self, tls: str, q: np.ndarray) -> np.ndarray:
        camp = self._start_campaign(tls)
        if camp is not None:
            if camp["t"] == 0:
                camp["lane"] = int(np.argmax(q))     # the longest queue when the campaign starts
            q[camp["lane"]] = 0.0
            self._end_campaign_step(tls, camp)
        return q

    def _detect_and_correct(self, states: Dict[str, np.ndarray], step: int) -> Dict[str, np.ndarray]:
        """Detect anomalous queue values and correct flagged lanes with LSTM prediction."""
        corrected = self._copy_states(states)

        for tls in corrected:
            queue_now = np.array(corrected[tls][:4], dtype=np.float32)
            hist = self.history_buffers[tls]

            truth = self._attacked_mask.get(tls, np.zeros(4, dtype=bool))

            # Need sufficient history before z-score detection.
            min_history = min(10, self.config.window_size)
            if len(hist) < min_history:
                self._score(np.zeros(4), np.zeros(4, dtype=bool), truth)
                hist.append(queue_now.copy())
                continue

            history_arr = np.stack(list(hist), axis=0)  # (k, 4)
            # Most recent window_size history for the LSTM prediction.
            if len(hist) >= self.config.window_size:
                seq = np.stack(list(hist)[-self.config.window_size :], axis=0)
            else:
                seq = history_arr

            if self.config.detector == "residual":
                # Score = vehicles between the reading and the LSTM's one-step forecast. Unlike
                # the z-score this sees a busy lane that suddenly reads 0 (queues also reach 0
                # at every green, so 0 is not unusual against the window's spread).
                pred = self.lstm_model.predict(seq)
                pending = self._pending.pop(tls, None)
                if self.config.spike_rollback and pending is not None:
                    # Lanes flagged last step but not corrected (persistence > 1): if this reading
                    # fits the forecast made as if last step's value had been the forecast, that
                    # value was a one-step spike. Erase it from the history so it does not throw
                    # off the forecasts that follow. A lasting change (attack or real) does not fit
                    # and is handled as before.
                    mask, values = pending
                    alt = seq.copy()
                    alt[-1, mask] = values[mask]
                    pred_alt = self.lstm_model.predict(alt)
                    spike = mask & (np.abs(queue_now - pred_alt) < self.config.residual_threshold)
                    if spike.any():
                        hist[-1][spike] = values[spike]
                        seq[-1, spike] = values[spike]
                        pred = np.where(spike, pred_alt, pred)
                z = np.abs(queue_now - pred)
                raw_flag = z >= self.config.residual_threshold
            else:
                mean = history_arr.mean(axis=0)
                # std_floor = 1e-6 is the original detector. A lane that sat at a constant
                # value for the whole window then gives z ~ 1e6 for a 1-vehicle change.
                std = np.maximum(history_arr.std(axis=0), self.config.std_floor)
                z = np.abs(queue_now - mean) / std
                # Flagged when the z-score is high AND the lane moved by at least min_deviation
                # vehicles (min_deviation 0 = the original detector).
                raw_flag = (z > self.config.z_threshold) & (np.abs(queue_now - mean) >= self.config.min_deviation)

            # Corrected once flagged `persistence` steps in a row (1 = the original detector).
            self._streak[tls] = np.where(raw_flag, self._streak[tls] + 1, 0)
            flag_mask = self._streak[tls] >= self.config.persistence
            flagged = np.flatnonzero(flag_mask)
            self._score(z, flag_mask, truth)

            if flagged.size > 0:
                if self.lstm_model is not None:
                    if self.config.detector != "residual":
                        pred = self.lstm_model.predict(seq)

                    old_values = queue_now.copy()
                    for lane in flagged:
                        corrected[tls][lane] = float(max(pred[lane], 0.0))
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
                
                if self.config.verbose:
                    for lane_idx in flagged:      # flagged already holds lane indices
                        print(f"[DEFENSE] anomaly at {tls} lane {lane_idx}: "
                              f"{old_values[lane_idx]:.1f} (z={z[lane_idx]:.2f}) "
                              f"-> LSTM {corrected[tls][lane_idx]:.1f}")

            if self.config.detector == "residual" and (raw_flag & ~flag_mask).any():
                self._pending[tls] = (raw_flag & ~flag_mask, np.maximum(pred, 0.0).astype(np.float32))
            hist.append(np.array(corrected[tls][:4], dtype=np.float32).copy())

        return corrected

    def _score(self, z: np.ndarray, flagged: np.ndarray, truth: np.ndarray) -> None:
        """Update the per-lane confusion matrix and the ROC score log."""
        self.tp += int(np.sum(flagged & truth))
        fp = int(np.sum(flagged & ~truth))
        self.fp += fp
        self.false_positive_count += fp
        self.fn += int(np.sum(~flagged & truth))
        self.tn += int(np.sum(~flagged & ~truth))
        self.scores.extend(np.asarray(z, dtype=float).tolist())
        self.labels.extend(truth.astype(int).tolist())

    def roc_curve(self, thresholds=None):
        """(thresholds, tpr, fpr) from the logged z-scores.

        Approximate for thresholds other than the one the run used: in defense mode a
        flagged lane is corrected before it enters the detector's history, so the
        history (and later z-scores) depend on the threshold that was actually used.
        """
        z = np.asarray(self.scores)
        y = np.asarray(self.labels).astype(bool)
        if thresholds is None:
            thresholds = np.linspace(0.0, 10.0, 101)
        pos, neg = max(int(y.sum()), 1), max(int((~y).sum()), 1)
        tpr = np.array([np.sum((z > t) & y) / pos for t in thresholds])
        fpr = np.array([np.sum((z > t) & ~y) / neg for t in thresholds])
        return np.asarray(thresholds), tpr, fpr

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
        Detection quality over every screened (step, intersection, lane).

        precision, recall (= detection rate), F1, FPR = FP / (FP + TN), MCC, and ROC AUC
        from the logged z-scores. ``flags_per_attack`` is the quantity the earlier code
        reported as "detection_rate" (detections / attacks), kept for comparison only.
        """
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denom if denom else 0.0
        auc = float("nan")
        y = np.asarray(self.labels, dtype=bool)
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        if n_pos and n_neg:
            # Mann-Whitney U with tie-averaged ranks == area under the ROC curve
            _, inv, counts = np.unique(np.asarray(self.scores), return_inverse=True, return_counts=True)
            avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
            auc = float((avg_rank[inv][y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
        return {
            "total_attack_events": float(self.total_attack_events),
            "total_detection_events": float(self.total_detection_events),
            "false_positive_count": float(self.false_positive_count),
            "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
            "precision": float(precision),
            "recall": float(recall),
            "detection_rate": float(recall),
            "f1": float(f1),
            "false_positive_rate": float(fpr),
            "mcc": float(mcc),
            "roc_auc": auc,
            "flags_per_attack": (float(self.total_detection_events) / self.total_attack_events
                                 if self.total_attack_events else 0.0),
        }
