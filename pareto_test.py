"""
test_pareto.py — Kiểm tra tính đúng đắn của Pareto mechanism

Chạy: python test_pareto.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from utils.pareto import (
    dominates, non_dominated_indices, pareto_front,
    ParetoArchive, select_by_pareto_dominance
)

PASS = "✓ PASS"
FAIL = "✗ FAIL"

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status}  {name}", f"→ {detail}" if detail else "")
    return condition

all_pass = True

# ══════════════════════════════════════════════════════════════
# SECTION 1: dominates() — cả hai objective MAXIMIZE
# ══════════════════════════════════════════════════════════════
print("\n=== 1. dominates() — maximize obj0, maximize obj1 ===")

# A=[10,5], B=[8,5], C=[8,7]  (obj0 maximize, obj1 minimize → truyền -obj1)
# Nếu obj1 là MINIMIZE, cần đổi dấu trước khi gọi dominates()
# Ở đây test dominates() với giả định cả hai maximize

A = np.array([10, -5])   # obj1 minimize → dùng âm
B = np.array([8, -5])
C = np.array([8, -7])    # C có obj1=7 lớn hơn (minimize tốt hơn) → -7 < -5 → C tệ hơn

# A vs B: A[0]=10>8, A[1]=-5=-5 → A dominates B
all_pass &= check("A dominates B", dominates(A, B), f"A={A}, B={B}")
# B vs A: B[0]=8<10 → B không dominate A
all_pass &= check("B NOT dominates A", not dominates(B, A))
# A vs C: A[0]=10>8, A[1]=-5>-7 → A dominates C
all_pass &= check("A dominates C", dominates(A, C))
# B vs C: B[0]=8=8, B[1]=-5>-7 → B dominates C
all_pass &= check("B dominates C", dominates(B, C))
# C vs A: C[0]=8<10 → C không dominate A
all_pass &= check("C NOT dominates A", not dominates(C, A))

# Equal solutions
D = np.array([10, -5])
all_pass &= check("A NOT dominates D (equal)", not dominates(A, D),
                  "equal solutions don't dominate each other")
all_pass &= check("D NOT dominates A (equal)", not dominates(D, A))

# ══════════════════════════════════════════════════════════════
# SECTION 2: non_dominated_indices()
# ══════════════════════════════════════════════════════════════
print("\n=== 2. non_dominated_indices() ===")

# Chỉ A và B không bị dominated
pts = np.array([[10, -5], [8, -5], [8, -7]])  # A, B, C
nd = non_dominated_indices(pts)
nd_set = set(nd)
all_pass &= check("Non-dominated = {A, B}", nd_set == {0, 1},
                  f"got indices {nd_set}")
all_pass &= check("C is dominated", 2 not in nd_set)

# Tất cả trên Pareto front
pts2 = np.array([[10, 1], [7, 5], [3, 9]])  # Pareto front (thương đổi)
nd2 = non_dominated_indices(pts2)
all_pass &= check("All 3 non-dominated", set(nd2) == {0, 1, 2},
                  f"got {set(nd2)}")

# ══════════════════════════════════════════════════════════════
# SECTION 3: ParetoArchive
# ══════════════════════════════════════════════════════════════
print("\n=== 3. ParetoArchive ===")

arch = ParetoArchive(max_size=5)

# Thêm 3 solutions non-dominated
added_a = arch.add(np.array([10.0, -5.0]), label='A')
added_b = arch.add(np.array([7.0, -2.0]), label='B')
added_c = arch.add(np.array([3.0, 0.0]), label='C')
all_pass &= check("A added", added_a)
all_pass &= check("B added", added_b)
all_pass &= check("C added", added_c)
all_pass &= check("Archive size = 3", len(arch) == 3, f"got {len(arch)}")

# Thêm solution bị dominated bởi A
added_d = arch.add(np.array([8.0, -6.0]), label='D')  # A dominates D? A=[10,-5], D=[8,-6]: A[0]>D[0], A[1]=-5>-6 ✓
all_pass &= check("D rejected (dominated by A)", not added_d)
all_pass &= check("Archive still size 3", len(arch) == 3)

# Thêm solution dominate cả B và C
added_e = arch.add(np.array([9.0, 1.0]), label='E')  # E=[9,1] vs B=[7,-2]: E[0]>B[0] but E[1]=1<-2 → E not dominate B
# E=[9,1] vs C=[3,0]: E[0]>C[0], E[1]=1>0 → E dominates C
all_pass &= check("E added (dominates C)", added_e)
# C bị xoá
labels_in = [e.get('label') for e in arch.get_front()]
all_pass &= check("C removed from archive", 'C' not in labels_in, f"labels={labels_in}")

# Archive đầy
arch2 = ParetoArchive(max_size=3)
for i in range(4):
    arch2.add(np.array([float(10 - i), float(i)]), idx=i)
all_pass &= check("Archive capped at max_size=3", len(arch2) <= 3, f"size={len(arch2)}")

# ══════════════════════════════════════════════════════════════
# SECTION 4: select_by_pareto_dominance()
# ══════════════════════════════════════════════════════════════
print("\n=== 4. select_by_pareto_dominance() ===")

# Q-vectors: candidate 0 dominates candidates 1,2
q = np.array([
    [10.0, -5.0],   # 0: tốt nhất
    [8.0, -5.0],    # 1: dominated bởi 0
    [8.0, -7.0],    # 2: dominated bởi cả 0 và 1
])
mask = np.array([True, True, True])
chosen = select_by_pareto_dominance(q, mask)
all_pass &= check("select_by_pareto picks non-dominated best", chosen == 0,
                  f"chosen={chosen}")

# Test với mask: node 0 không valid
mask2 = np.array([False, True, True])
chosen2 = select_by_pareto_dominance(q, mask2)
all_pass &= check("Mask filters invalid nodes; picks node 1", chosen2 == 1,
                  f"chosen={chosen2}")

# Multiple non-dominated: tie-breaking bằng pareto_w
q2 = np.array([
    [10.0, 1.0],   # 0: high obj0, low obj1
    [3.0, 9.0],    # 1: low obj0, high obj1
])
mask3 = np.array([True, True])
# Prefer obj0 (w=[0.9, 0.1])
pw_prefer_0 = np.array([0.9, 0.1])
chosen_pref0 = select_by_pareto_dominance(q2, mask3, pw_prefer_0)
all_pass &= check("Tie-break: prefer obj0 → picks candidate 0",
                  chosen_pref0 == 0, f"chosen={chosen_pref0}")
# Prefer obj1 (w=[0.1, 0.9])
pw_prefer_1 = np.array([0.1, 0.9])
chosen_pref1 = select_by_pareto_dominance(q2, mask3, pw_prefer_1)
all_pass &= check("Tie-break: prefer obj1 → picks candidate 1",
                  chosen_pref1 == 1, f"chosen={chosen_pref1}")

# ══════════════════════════════════════════════════════════════
# SECTION 5: HL Agent vector Q shape test
# ══════════════════════════════════════════════════════════════
print("\n=== 5. HLSharedQScorer output shape ===")

from config import Config
from models.hl_scorer import HLSharedQScorer, N_OBJ_HL

cfg = Config()
net = HLSharedQScorer(cfg)
B = 5  # batch size = số SFC candidates
z_g = torch.randn(B, cfg.d_global)
sfc_f = torch.randn(B, cfg.qnet.d_sfc)
pw = torch.randn(B, 2)
out = net(z_g, sfc_f, pw)
all_pass &= check(
    f"HL Q output shape = [{B}, {N_OBJ_HL}]",
    out.shape == (B, N_OBJ_HL),
    f"got {tuple(out.shape)}"
)

# ══════════════════════════════════════════════════════════════
# SECTION 6: LL Agent vector Q shape test
# ══════════════════════════════════════════════════════════════
print("\n=== 6. LLNodeScorer output shape ===")

from models.ll_dqn import LLNodeScorer, N_OBJ

N = 50  # num nodes
ll_net = LLNodeScorer(cfg)
zg = torch.randn(N, cfg.d_global)
zp = torch.randn(N, cfg.vgae.d_latent)
zc = torch.randn(N, cfg.vgae.d_latent)
vf = torch.randn(N, cfg.qnet.d_vnf)
sf = torch.randn(N, cfg.qnet.d_sfc)
pw2 = torch.randn(N, 2)
out_ll = ll_net(zg, zp, zc, vf, sf, pw2)
all_pass &= check(
    f"LL Q output shape = [{N}, {N_OBJ}]",
    out_ll.shape == (N, N_OBJ),
    f"got {tuple(out_ll.shape)}"
)

# ══════════════════════════════════════════════════════════════
# SECTION 7: Pareto front từ evaluate.py helper
# ══════════════════════════════════════════════════════════════
print("\n=== 7. pareto_front() helper ===")

pts_legacy = [(10, -5), (8, -5), (8, -7), (9, -4)]
front = pareto_front(pts_legacy)
front_set = set(map(tuple, front))
all_pass &= check("(9,-4) on front", (9.0, -4.0) in front_set, f"front={front_set}")
all_pass &= check("(8,-7) not on front", (8.0, -7.0) not in front_set)

# ══════════════════════════════════════════════════════════════
# SECTION 8: HL Agent buffer stores vector reward
# ══════════════════════════════════════════════════════════════
print("\n=== 8. HLAgent stores vector reward [N_OBJ_HL] ===")

import torch
from agents.hl_agent import HLAgent

device = torch.device('cpu')
hl_agent = HLAgent(cfg, device)
z_g_cpu = torch.randn(cfg.d_global)
sfc_f_cpu = torch.randn(cfg.qnet.d_sfc)
pw_cpu = torch.tensor([0.6, 0.4])
reward_vec = np.array([1.5, -0.3], dtype=np.float32)

hl_agent.store_transition(
    z_g_cpu, sfc_f_cpu, pw_cpu,
    0,                    # action_idx
    reward_vec,           # vector reward
    z_g_cpu, [], pw_cpu,
    0.0
)
stored = hl_agent.buffer.buf[-1]
stored_rew = stored[4]   # index 4 = reward_vec
all_pass &= check(
    "HL buffer stores vector reward",
    isinstance(stored_rew, np.ndarray) and stored_rew.shape == (N_OBJ_HL,),
    f"shape={stored_rew.shape}, dtype={stored_rew.dtype}"
)

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
if all_pass:
    print("  ALL TESTS PASSED ✓")
else:
    print("  SOME TESTS FAILED ✗ — xem chi tiết bên trên")
print("="*55)
