# Project Report: Before vs After the Revision

**Project:** Hierarchical Cooperative Multi-Agent RL for Secure Traffic Signal Control
**Paper:** NCA 2026 submission #106 (rejected 2026-09-15), now being revised for a journal
**Report date:** 2026-10-22

This report explains, in simple words:
1. what the old system was and what it claimed,
2. every problem we found (the reviewers' comments and the bugs we found ourselves),
3. what we fixed,
4. the new results,
5. how to run the project and show it live.

---

## 1. Summary in one table

| | Old version (submitted to NCA 2026) | New version (now) |
|---|---|---|
| Did RL really control the lights? | **No.** SUMO kept running its own fixed 42 s program; RL "keep" did not hold the green | **Yes.** A new signal controller holds the green, adds a 3 s yellow, 10 s min green, 60 s max green |
| Traffic | Very light in the grids (v/c below 0.1), constant for the whole hour, up to 55% turning | Realistic: v/c ≈ 0.6, **18% turning**, **demand that changes during the hour** |
| Metrics | Snapshot at the last second; "throughput" was really "cars in the network"; a sum compared with a mean | SUMO trip data for every car (including cars that never got in); real throughput; time-averaged queue |
| Random traffic per episode | No. Phases 0–1 replayed the same traffic, so "20 episodes" was really 1 | Yes. Separate seeds for training, validation (choose settings) and test (final tables) |
| Baselines | Fixed 30 s timer, wrongly called "Webster" | Fixed-time 30 s, **real Webster (1958)**, **max-pressure** |
| Statistics | Mean ± std, numbers typed by hand | Mean ± std, 95% CI, paired Wilcoxon test with Holm correction, tables made by a script |
| Supervisor | Not a valid RL update (tanh output, same target for all 4 outputs) | Proper Double DQN over 4 discrete options per agent |
| Security attacks | 1 easy attack, built to be caught | 5 attacks: random, detector-aware, ramp, replay, **hide-the-jam** |
| Security metrics | None (the false-positive counter was never updated) | Precision, recall, F1, false-positive rate, MCC, ROC |
| Defence | Z-score detector: 13% false alarms, made traffic **worse** | LSTM forecast check: ~0.1% false alarms, **no cost**, recovers 29% of the real attack's damage |
| Headline results | Claimed "94% better than fixed timers" | Honest: RL beats the best baseline by 2.6–4.6% in the grids; ties Webster on one intersection |

---

## 2. The old system (what we submitted)

**Phases:**
- **Phase 0:** 1 intersection, 1 DDQN agent. State = 4 queues + current phase + time since switch. Action = keep or switch.
- **Phase 1:** 2×2 grid (4 lights). Phase 0 weights copied to every light. There were two versions:
  - **independent:** each light sees only itself;
  - **cooperative:** each light also sees its neighbours' queues.
- **Phase 2:** 4×2 grid (8 lights) in 2 groups of 4. There were three versions:
  - **plain DDQN:** no supervisors;
  - **local supervisors:** one per group, sending each light a signal;
  - **global supervisors:** the 2 supervisors also share a summary with each other.
- **Phase 3:** a hacker changes sensor data (false-data injection, FDI). A z-score detector finds strange values, and an LSTM replaces them.

**What the paper/README claimed:**
- Phase 0: **94.3% improvement** over fixed timers.
- Phase 1: **68% instant improvement** with zero extra training.
- Phase 2: local supervisor **+4.6%**, global supervisor **+2.8%**.
- Phase 3: **"full reward recovery"** under attack, and the defended system even **beat the no-attack baseline**.

As the sections below show, none of these numbers could be trusted.

---

## 3. Problems found

### 3.1 The reviewers' comments (2 reviewers, both "Reject (2)")

Both reviewers liked the idea. All their objections were about rigour. The 15 comments come from 4 root causes:

| Root cause | Meaning in simple words | Comments |
|---|---|---|
| **A** | The supervisor's learning rule was not real RL | R1-W1, R1-D2, R2-W1, R2-D2 |
| **B** | Metrics were wrong (snapshots, sum vs mean, no real throughput) | R1-W2, R1-D4, R2-W3, R2-D4 |
| **C** | The grids had almost no traffic, so every controller looked perfect | R1-D2, R1-D4, R2-D4 |
| **D** | The security test used one easy attack and no detection metrics | R1-W3, R1-D5, R2-W4, R2-D5, R2-D6 |

Every comment, one by one:

| ID | What the reviewer said | Status now |
|---|---|---|
| R1-W1 | The supervisor update is not a valid RL update | ✅ Fixed: discrete-option Double DQN supervisor. ❌ Not done: the ablation where the signal is replaced by a constant, noise or a shuffled value |
| R1-W2 / R1-D2 | Missing simulator, demand and hyperparameter details; only 5 seeds | ✅ All details are now defined in code and notes. ⚠️ Seeds: **1 training seed × 20 test episodes** (professor's decision); paired tests and 95% CIs added |
| R1-W3 | The attack is easy to detect; no detection accuracy, FPR, reconstruction error or overhead | ✅ Detection metrics and 4 new attacks added. ⚠️ Reconstruction error: only the LSTM forecast error is reported (MAE 0.78 vehicles). ⚠️ Overhead: measured in September on the old code; should be re-measured |
| R1-W4 | The intersection grouping is arbitrary | ❌ Not done (partition ablation, automatic grouping) |
| R1-D1 | "No runtime communication" contradicts the supervisor design | 📝 Text only: withdraw the claim and report bytes per step (table in the response document) |
| R1-D3 | Phase 2 rows of Table I not explained | 📝 Text only |
| R1-D4 | Near-zero waiting times and queues: are they correct? | ✅ Fixed: trip-based metrics and realistic demand (v/c ≈ 0.6). ❌ Not done: the demand sweep across several load levels |
| R1-D5 | Attack and network parameters are fixed, with no sensitivity analysis | ⚠️ Only the detector settings were tuned (on validation seeds). ❌ Not done: the full parameter sweep |
| R2-W1 | Novelty is incremental | 📝 Text only: reposition the paper |
| R2-W2 / R2-D3 | Scalability only up to 8 lights; overhead not measured | ⚠️ Overhead measured (September). ❌ Not done: 16+ intersections |
| R2-W3 / R2-D4 | Networks of different size compared directly; exact-zero waiting times | ✅ Fixed: per-car metrics, same load level on every network, real Webster + max-pressure |
| R2-D2 | The supervisor and cross-group summary are unclear | 📝 Text only (exact definition is in the response document) |
| R2-D5 | The security evaluation is too limited | ✅ Fixed: 5 attacks, detection metrics, false-alarm cost measured |
| R2-D6 | The defended system beats the no-attack baseline: separate mitigation from smoothing | ⚠️ Partly done: the new "guard" scenario (detector on, no attack) gives 190.83 vs 190.81 s, so there is **no smoothing effect** with the new detector. ❌ Not done: the full factorial (last-value hold, always-LSTM, moving average) |

Legend: ✅ done · ⚠️ partly done · ❌ not done yet · 📝 only needs writing in the paper.

### 3.2 Problems we found ourselves (the reviewers did not see these)

| # | Problem | Why it mattered |
|---|---|---|
| 1 | **RL never controlled the lights.** The network files had a fixed 42 s program, and the code only moved its pointer. "Keep" did not hold the green, and "switch" jumped into a yellow | **Every old RL number was mostly SUMO's own fixed-time plan** |
| 2 | Phases 0 and 1 had no random seed, so all episodes were identical | "20 episodes" was really 1 episode repeated |
| 3 | The tables in the paper matched neither the result files nor the README (3 different sets of numbers) | The numbers could not be reproduced |
| 4 | The "Webster" baseline was just a 30 s fixed timer | Wrong label and wrong citation |
| 5 | "Throughput" was the number of cars in the network at one moment | It did not measure cars served |
| 6 | Phase 0 reported a **sum** of waiting time; the other phases reported a **mean** | The rows of Table I were not comparable |
| 7 | Waiting time used SUMO's "consecutive halt" value, which resets when a car moves; it read exactly 0 in a quarter of the episodes | The "0.000 s" values were an artefact |
| 8 | The security detection rate was 2.49 (flags per attack), and the false-positive counter was never increased | The detector was mostly raising false alarms |
| 9 | Phase 3 used different traffic in each scenario | Scenarios could not be compared fairly |
| 10 | The grid agents learned to **keep cars outside the network** (cars never inserted are invisible to the queue and travel time) | Fake "improvement" |
| 11 | State inputs were capped (queue ≤ 20, time ≤ 60), so the agents froze on one green for ~55 minutes | Grid4 RL: 514 s with huge variance |
| 12 | Loading the Phase 0 model also loaded its old ε (0.01) and learning rate (1e-3) | Every grid fine-tune ignored the settings we chose |
| 13 | **Heavy permissive left turns:** 55% of the cars from the north at one junction turned, sharing a lane with straight traffic | Left-turners blocked the lane, and only synchronised fixed-time could handle it; **every** adaptive controller lost, not only RL |
| 14 | The "secure" scenario was exactly the same as "defense" (no network noise) | One of the 5 scenarios was a duplicate |
| 15 | The z-score detector could not see a fake "0" (queues are 0 at every green anyway) and raised false alarms on small changes | The defence made traffic 5–12 s worse |

---

## 4. What we fixed

### 4.1 Simulation and control

| Fix | Where |
|---|---|
| `SignalController` really holds the green, adds a 3 s yellow on every switch, min green 10 s, **max green 60 s** | `src/traffic_rl/envs/common.py` |
| Every episode gets a known random seed: training 42+k, validation 200000+k, **test 100000+k** | `envs/common.py` |
| State uses `log1p(queue)` with no cap, and the time feature is ÷60 | `envs/common.py` (`queue_feature`, `time_feature`) |
| A queue counts the cars waiting to enter the network too | `envs/common.py` (`_queue`) |
| Episodes are 3600 s in every phase | all envs |
| Grid episodes are 5–7× faster (unused SUMO calls removed; results unchanged) | all envs |

### 4.2 Metrics and statistics

| Fix | Where |
|---|---|
| SUMO trip output for every car, including unfinished and never-inserted cars | `envs/common.py` |
| Headline metric `mean_time_in_system_all`: average time per car, counting cars stuck outside | `envs/common.py` |
| Real throughput (arrived cars per hour), time-averaged queue, entry backlog | `envs/common.py` |
| Tables made only by a script: mean ± std, 95% CI, paired Wilcoxon test, Holm correction | `tools/make_tables.py` |

### 4.3 Traffic demand (scenario)

| Fix | Where |
|---|---|
| Demand calibrated to v/c ≈ 0.6 on every network (same load level everywhere) | `tools/calibrate_demand.py` → `sumo_files/*/calibrated.rou.xml` |
| **Turning share cut to 18%:** each turning flow ÷ 3, those cars sent straight, same total cars | `sumo_files/grid4/turn18.rou.xml`, `sumo_files/grid8/turn18.rou.xml` |
| **Time-varying demand:** 4 blocks of 15 minutes, same hourly total (NS ×1.3/1.1/0.7/0.9, EW ×0.7/1.1/1.3/0.9) | `tools/make_dynamic_routes.py` → `turn18_dynamic.rou.xml` |

Why we changed the scenario: with 55% turning on shared lanes, **every** adaptive controller lost to synchronised fixed-time, not just RL. This traffic would normally get a separate left-turn signal under common traffic-engineering practice. With through traffic only, RL beat fixed-time by 16%; with 18% turning plus changing demand, RL beats every baseline. **The paper must say that we changed the scenario and why.**

### 4.4 Baselines

| Fix | Where |
|---|---|
| Real Webster (1958) plan for each intersection (saturation flow measured in SUMO: 1764 veh/h/lane) | `src/traffic_rl/baselines.py` |
| New max-pressure controller (Varaiya 2013), under the same rules as RL (5 s decisions, min/max green) | `src/traffic_rl/baselines.py` |
| Baselines on time-varying traffic use plans computed from the hourly average | `scripts/evaluate_baselines.py --run-route-file` |

### 4.5 Learning

| Fix | Where |
|---|---|
| Supervisor = branching Double DQN over 4 options per agent (balance, prioritise-NS, prioritise-EW, hold), unbounded Q-values | `src/traffic_rl/supervisor/agent.py` |
| Queue-only reward (decided 2026-10-17; the old reward did not beat Webster) | all envs |
| Fine-tuning resets ε and learning rate after loading a model | `phase1_grid4/run.py`, `phase2_hierarchy/baseline.py` |
| A control run: grid8 DDQN trained 300 more episodes **without** supervisors, so a supervisor gain can't be explained by extra training | Phase 2 runs |

### 4.6 Security (Phase 3)

| Fix | Where |
|---|---|
| Per-lane confusion matrix: precision, recall, F1, FPR, MCC, ROC | `src/traffic_rl/security/layer.py` |
| New attacks: detector-aware, ramp, replay, **hide** (report 0 cars on the busiest lane for 100 s) | `layer.py` |
| New scenario **guard** (detector on, no attack) = the cost of false alarms | `layer.py`, `run_scenarios.py` |
| "secure" now really = network loss/delay + attack + defence | `layer.py` |
| **New detector:** flag a lane when the reading differs from the **LSTM forecast** by ≥ 5 cars, **2 steps in a row**; one-step spikes are erased from its memory. Settings chosen on validation seeds only | `layer.py` (`--detector residual --residual-threshold 5 --persistence 2 --spike-rollback`) |
| LSTM trained on 100 clean episodes with their own seeds (300000+), never on test seeds | `collect_data.py --sumo-seed` |
| All 5 scenarios use the same 20 test seeds (fair paired comparison) | `run_scenarios.py --eval-seed` |

---

## 5. New results (20 test episodes, seeds 100000–100019, one training seed)

Main metric: **average time per car in the system (s), lower is better**. Every "p" is a paired test against the best baseline.

### Phase 0: single intersection
| Controller | Constant demand | Time-varying demand |
|---|---|---|
| Fixed-time 30 s | 50.90 | 60.04 |
| **Webster** | **45.47** | **54.51** |
| DDQN | 45.58 (+0.2%, p=0.001) | 55.07 (n.s.) |

➡️ On one intersection, RL **ties Webster**; Webster is near-optimal there.

### Phase 1: grid4, 18% turning + time-varying demand
| Controller | Time per car | vs Webster |
|---|---|---|
| Fixed-time 30 s | 167.8 | +6.9% |
| Max-pressure | 170.0 | +8.3% |
| Webster | 157.0 | – |
| **DDQN independent** | **152.8** | **−2.7% (p<0.001)** |
| DDQN cooperative | 154.4 | −1.6% (p=0.015) |

Independent DDQN is also better on travel time (−2.9%), waiting (−14%) and queue (−13%), with equal throughput.

### Phase 2: grid8, 18% turning + time-varying demand
| Controller | Time per car | vs Webster |
|---|---|---|
| Fixed-time 30 s | 213.1 | +9.8% |
| Max-pressure | 196.7 | +1.4% (n.s.) |
| Webster | 194.0 | – |
| DDQN (no supervisor) | 188.9 | −2.6% (p<0.001) |
| DDQN + 300 more episodes (control) | 192.7 | −0.7% (n.s.) |
| Local supervisors | 190.8 | −1.6% (p<0.001) |
| **Global supervisors** | **185.0** | **−4.6% (p<0.001)** |

- Global supervisors vs plain DDQN: −3.9 s per car and −16% waiting (both p<0.001), with the same throughput (p=0.98).
- More training alone made things worse, so the gain comes from the supervisors.
- Local supervisors beat the control run but are 1.9 s worse than plain DDQN.
- ⚠️ Every RL version serves about 5–8 fewer cars per hour than Webster (out of ~8,700; p≈0.02–0.05). This must be reported.

### Phase 3: security (grid8 with local supervisors)
No attack: 190.81 s · network problems only: 199.57 s.

| Attack | Attack, no defence | Old defence (z-score) | **New defence** |
|---|---|---|---|
| random | 190.51 | 196.07 | **190.81** (no cost) |
| detector-aware | 190.94 | 197.22 | **191.01** (no cost) |
| ramp | 191.73 | 197.15 | **191.26** |
| replay | 191.68 | 197.68 | **191.84** (no cost) |
| **hide the jam** | **197.96** | 210.16 | **195.88** (29% of the damage recovered, p=0.002) |
| guard (no attack, detector on) | – | 195.76 (13% false alarms) | **190.83** (0.1% false alarms, n.s.) |
| secure (network problems + attack + defence) | – | 206–224 | **200–204** |

➡️ The new defence never makes traffic worse, and it helps against the one attack that really hurts. It misses small hidden queues (a few cars), which look like normal traffic and do little harm; this is a limitation to state in the paper.

### Results on the old scenario (55% turning; must be reported as the reason for the change)
| Network | Fixed-time | Webster | Max-pressure | DDQN |
|---|---|---|---|---|
| grid4 | **240.9** | 314.0 | 358.2 | 298.9 (independent) / 330.2 (cooperative) |
| grid8 | **214.9** | 265.2 | 238.6 | 241.8 |

---

## 6. Still open (honest list)

1. **Multi-seed training.** Everything is one training seed. On hold by the professor's decision; the paper must say so.
2. **Not done yet:**
   - supervisor signal ablation (constant, noise, shuffled);
   - grouping ablation;
   - demand sweep;
   - parameter sensitivity sweeps;
   - 16+ intersection grids;
   - coordinated multi-lane attack;
   - full smoothing-vs-mitigation factorial;
   - re-measuring overhead on the new code.
3. **Paper text:** rewrite the results from the tables above. Withdraw "no runtime communication", "94% improvement" and "full recovery". Explain the scenario change.
4. **Nothing is committed to git since `f06370f`.** That waits for your approval.

---

## 7. How to run the project

### 7.1 One-time setup (Windows, PowerShell)
```powershell
pip install -r requirements.txt
pip install eclipse-sumo==1.25.0
```
On this laptop the environment variables are already set. On a new machine, set them in each new terminal:
```powershell
$env:SUMO_HOME = "C:\Users\<you>\AppData\Local\Programs\Python\Python313\Lib\site-packages\sumo"
$env:PATH += ";$env:SUMO_HOME\bin"
$env:MPLBACKEND = "Agg"
$env:PYTHONIOENCODING = "utf-8"
$env:CUDA_VISIBLE_DEVICES = "-1"   # CPU is faster than GPU for this project
```
Run every command from the project folder.

### 7.2 Show it live (SUMO window)
The trained models of the final scenario are already in the folders the scripts load from. Each command opens the SUMO window and starts the traffic right away; the window closes at the end of the episode.

**Tips for the demo:**
- Use the **Delay** slider at the top of the SUMO window to slow the traffic down or speed it up.
- Zoom with the mouse wheel. You will see the lights switching between green, yellow and red as the agents decide.
- One episode is 1 simulated hour. In the first 15 minutes the north–south roads are busy; in minutes 30–45 the east–west roads are.
- When the episode ends, the console prints the results (travel time, waiting, throughput).

**1. Phase 0: one intersection, one RL agent**
```powershell
python scripts/phase0_single/run.py --mode evaluate --eval-episodes 1 --gui
```

**2. Phase 1: grid4, 4 RL agents (add `--cooperative` for the cooperative version)**
```powershell
python scripts/phase1_grid4/run.py --mode evaluate --load-finetuned --eval-episodes 1 --gui --route-file sumo_files/grid4/turn18_dynamic.rou.xml
```

**3. Phase 2: grid8 with global supervisors (best system)**
```powershell
python scripts/phase2_hierarchy/global_supervisor.py --mode evaluate --load-final --eval-episodes 1 --gui --route-file sumo_files/grid8/turn18_dynamic.rou.xml
```
Same for `local_supervisor.py`, or plain DDQN with `baseline.py --mode evaluate --load-final ...`. Add `--num-seconds 900` for a shorter, 15-minute demo.

**4. Phase 3: "hide the jam" attack, without and then with the defence**
```powershell
python scripts/phase3_security/run_scenarios.py --gui --episodes 1 --attack-type hide --scenarios attack defense --detector residual --residual-threshold 5 --persistence 2 --spike-rollback --tag _demo --route-file sumo_files/grid8/turn18_dynamic.rou.xml
```
Two SUMO windows open one after the other: first under attack, then with the defence. The console prints both results.

⚠️ The demo commands write small result files into `outputs/results/<phase>/`. That is fine: the paper numbers live in separate folders (section 7.4) that the demos never touch.

### 7.3 Reproduce the main results (long runs, several hours)
Use `--eval-seed 200000` (validation seeds) for any new tuning; keep the test seeds (100000) for final tables only.
```powershell
# Phase 0 (about 1 h)
python scripts/phase0_single/run.py --mode all --episodes 900

# Phase 1, grid4 (about 1 h each)
python scripts/phase1_grid4/run.py --mode all --episodes 150 --route-file sumo_files/grid4/turn18_dynamic.rou.xml
python scripts/phase1_grid4/run.py --mode all --cooperative --episodes 150 --route-file sumo_files/grid4/turn18_dynamic.rou.xml

# Phase 2, grid8 (about 2-4 h each; the supervisors start from the plain-DDQN result)
python scripts/phase2_hierarchy/baseline.py --mode all --episodes 300 --route-file sumo_files/grid8/turn18_dynamic.rou.xml
python scripts/phase2_hierarchy/local_supervisor.py --mode all --episodes 300 --route-file sumo_files/grid8/turn18_dynamic.rou.xml
python scripts/phase2_hierarchy/global_supervisor.py --mode all --episodes 300 --route-file sumo_files/grid8/turn18_dynamic.rou.xml

# Baselines on the same 20 test seeds (Webster plans come from the hourly-average file)
python scripts/evaluate_baselines.py --net grid8 --controller fixed_time --controller webster --controller max_pressure --route-file sumo_files/grid8/turn18.rou.xml --run-route-file sumo_files/grid8/turn18_dynamic.rou.xml --tag _turn18_dynamic

# Phase 3 (data -> LSTM -> scenarios)
python scripts/phase3_security/collect_data.py --episodes 100 --route-file sumo_files/grid8/turn18_dynamic.rou.xml
python scripts/phase3_security/split_data.py
python scripts/phase3_security/train_lstm.py
python scripts/phase3_security/run_scenarios.py --episodes 20 --attack-type hide --detector residual --residual-threshold 5 --persistence 2 --spike-rollback --tag _residual_v2 --route-file sumo_files/grid8/turn18_dynamic.rou.xml
#   (repeat with --attack-type random / detector_aware / ramp / replay)

# Any table from the result CSVs
python tools/make_tables.py --name my_table --ref "Webster=<csv>" "DDQN=<csv>" --metrics mean_time_in_system_all,mean_travel_time,mean_waiting_time,throughput_veh_h
```
On this 16 GB laptop, run at most 4–5 jobs at the same time.

### 7.4 Where the results are
| What | File |
|---|---|
| Phase 0 tables | `outputs/tables/phase0.md`, `outputs/tables/phase0_dynamic.md` |
| Phase 1 table (main scenario) | `outputs/results/candidate_turn18_dynamic/tables/candidate_turn18_dynamic_grid4.md` |
| Phase 2 table (main scenario) | `outputs/results/main_turn18_dynamic/tables/main_grid8_phase2.md` (and `main_grid8_vs_ddqn.md`) |
| Phase 3 table | `outputs/results/main_turn18_dynamic/tables/phase3_security.md` |
| Phase 3 raw CSVs | `outputs/results/main_turn18_dynamic/security/<attack>[_residual_v2]/` |
| Old-scenario (55% turning) results | `outputs/tables/phase1_grid4.md`, `outputs/archive/standard_folders_before_2026-10-22/` |
| Diagnosis runs (through-only, 18% constant) | `outputs/results/candidate_through_only/`, `outputs/results/candidate_turn18/` |
| Trained models used by the demos | `outputs/checkpoints/{single,grid4,cooperative,grid8,supervisor,global_supervisor,security}/` |
| Day-by-day notes | `paper/SESSION_2026-10-17.md`, `paper/SESSION_2026-10-21.md`, `paper/SESSION_2026-10-22.md` |
| Reviewer response draft | `paper/Response_to_Reviewers.docx` |

Note: on 2026-10-22 the old contents of the standard checkpoint and result folders (the old-scenario models and the pre-revision supervisor/LSTM models) were **moved, not deleted**, to `outputs/archive/standard_folders_before_2026-10-22/`.
