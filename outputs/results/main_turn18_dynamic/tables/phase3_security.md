# Phase 3 security, main scenario (grid8, 18% turning + time-varying), local-supervisor system
Test seeds 100000-100019, 20 episodes per cell, mean_time_in_system_all (s). Paired Wilcoxon on episode seed.
No attack: 190.81 s. Unreliable network only: 199.57 s.
Guard (detector on, no attack): original z-score 195.76 (FPR 13%, p<0.001); final detector 190.83 (FPR 0.1%, p=0.32).

| Attack | Attack, no defence | Original z-score defence | Final defence (residual + spike rollback) | Final vs attack | Final recall / FPR | Secure: original → final |
|---|---|---|---|---|---|---|
| random | 190.51 (p=0.70 vs none) | 196.07 | 190.81 | p=0.52 | 2% / 0.8% | 216.8 → 200.7 |
| detector-aware | 190.94 (p=0.78) | 197.22 | 191.01 | p=0.81 | 10% / 1.6% | 206.4 → 200.1 |
| ramp | 191.73 (p=0.04) | 197.15 | 191.26 | p=0.20 | 4% / 0.9% | 215.8 → 201.9 |
| replay | 191.68 (p=0.01) | 197.68 | 191.84 | p=0.38 | 1% / 0.4% | 217.3 → 202.8 |
| hide | 197.96 (p<0.001) | 210.16 | 195.88 | p=0.002 (29% of damage recovered) | 5% / 1.8% | 224.5 → 203.7 |

Final detector: |reading - LSTM one-step forecast| >= 5 vehicles for 2 steps in a row, one-step spikes erased from
the history; settings chosen on validation seeds 200000+. Recall is per attacked lane-step: most hidden lane-steps
hold only a few vehicles, which the forecast cannot tell from normal traffic (and which do little harm).
