import sys, time, statistics
sys.path.insert(0, "src")
import torch, numpy as np
from traffic_rl.core.network import DDQNNetwork
from traffic_rl.supervisor.agent import SupervisorNetwork
from traffic_rl.security.lstm import TrafficLSTM

torch.set_num_threads(1)
dev = torch.device("cpu")

def nparams(m): return sum(p.numel() for p in m.parameters())
def bytes_(m): return sum(p.numel()*p.element_size() for p in m.parameters())

def bench(m, x, n=2000, warm=200):
    m.eval()
    with torch.no_grad():
        for _ in range(warm): m(x)
        ts=[]
        for _ in range(n):
            t0=time.perf_counter(); m(x); ts.append((time.perf_counter()-t0)*1e6)
    ts.sort()
    return statistics.mean(ts), ts[int(.5*n)], ts[int(.95*n)], ts[int(.99*n)]

rows=[]
models = {
 "DDQN local (6->128->128->2), Phase 0/1-indep": (DDQNNetwork(6,2,128), torch.zeros(1,6)),
 "DDQN local (8->128->128->2), Phase 1-coop":    (DDQNNetwork(8,2,128), torch.zeros(1,8)),
 "DDQN local (7->128->128->2), Phase 2":         (DDQNNetwork(7,2,128), torch.zeros(1,7)),
 "Supervisor local (24->64->64->4)":             (SupervisorNetwork(24,64,4), torch.zeros(1,24)),
 "Supervisor global (28->64->64->4)":            (SupervisorNetwork(28,64,4), torch.zeros(1,28)),
 "LSTM predictor (4->64, +Linear 64->4)":        (TrafficLSTM(4,64,1,4,device="cpu"), torch.zeros(1,20,4)),
}
print(f"{'Model':46s} {'Params':>8s} {'KB(fp32)':>9s} {'mean us':>9s} {'p50':>8s} {'p95':>8s} {'p99':>8s}")
tot=0
for name,(m,x) in models.items():
    m=m.to(dev); x=x.to(dev)
    p=nparams(m); b=bytes_(m); mu,p50,p95,p99=bench(m,x)
    print(f"{name:46s} {p:8d} {b/1024:9.1f} {mu:9.2f} {p50:8.2f} {p95:8.2f} {p99:8.2f}")

# z-score cost (numpy, 4 lanes, window 20)
h=np.random.rand(20,4).astype(np.float32); q=np.random.rand(4).astype(np.float32)
for _ in range(500):
    mu_=h.mean(0); sd=h.std(0)+1e-6; _=np.abs(q-mu_)/sd
ts=[]
for _ in range(20000):
    t0=time.perf_counter(); mu_=h.mean(0); sd=h.std(0)+1e-6; z=np.abs(q-mu_)/sd; ts.append((time.perf_counter()-t0)*1e6)
ts.sort()
print(f"\nZ-score per intersection (4 lanes, W=20): mean {statistics.mean(ts):.2f} us, p95 {ts[int(.95*len(ts))]:.2f} us")

# whole-network per-decision cost, Phase 2 8 intersections
print("\n--- Per decision step (8 intersections), CPU single thread ---")
q7=DDQNNetwork(7,2,128).eval(); s28=SupervisorNetwork(28,64,4).eval()
x7=torch.zeros(1,7); x28=torch.zeros(1,28)
with torch.no_grad():
    for _ in range(200):
        s28(x28); q7(x7)
    ts=[]
    for _ in range(1000):
        t0=time.perf_counter()
        for _ in range(2): s28(x28)      # 2 supervisors
        for _ in range(8): q7(x7)        # 8 local agents
        ts.append((time.perf_counter()-t0)*1e6)
ts.sort()
print(f"2 supervisors + 8 agents: mean {statistics.mean(ts):.1f} us, p95 {ts[int(.95*len(ts))]:.1f} us  (decision interval = 5 000 000 us)")
