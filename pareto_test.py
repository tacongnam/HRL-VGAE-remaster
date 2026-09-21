import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from utils.pareto import (
    dominates, non_dominated_indices, pareto_front,
    ParetoArchive, select_by_hypervolume,
    compute_hypervolume, hypervolume_contribution,
    prune_by_hypervolume, hv_score_for_action
)

PASS = "PASS"
FAIL = "FAIL"

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}]  {name}", f"-> {detail}" if detail else "")
    return condition

all_pass = True

print("\n=== 1. dominates() ===")
A = np.array([10, -5])
B = np.array([8, -5])
C = np.array([8, -7])
all_pass &= check("A dominates B", dominates(A, B))
all_pass &= check("B NOT dominates A", not dominates(B, A))
all_pass &= check("A dominates C", dominates(A, C))
all_pass &= check("B dominates C", dominates(B, C))
all_pass &= check("C NOT dominates A", not dominates(C, A))
D = np.array([10, -5])
all_pass &= check("A NOT dominates D (equal)", not dominates(A, D))
all_pass &= check("D NOT dominates A (equal)", not dominates(D, A))

print("\n=== 2. non_dominated_indices() ===")
pts = np.array([[10, -5], [8, -5], [8, -7]])
nd = set(non_dominated_indices(pts))
all_pass &= check("Non-dominated = {A} (A dominates B and C)", nd == {0}, f"got {nd}")
all_pass &= check("B dominated by A", 1 not in nd)
all_pass &= check("C dominated", 2 not in nd)
pts2 = np.array([[10, 1], [7, 5], [3, 9]])
nd2 = set(non_dominated_indices(pts2))
all_pass &= check("All 3 non-dominated", nd2 == {0, 1, 2}, f"got {nd2}")

print("\n=== 3. ParetoArchive ===")
arch = ParetoArchive(max_size=5)
added_a = arch.add(np.array([10.0, -5.0]), label='A')
added_b = arch.add(np.array([7.0, -2.0]), label='B')
added_c = arch.add(np.array([3.0, 0.0]), label='C')
all_pass &= check("A added", added_a)
all_pass &= check("B added", added_b)
all_pass &= check("C added", added_c)
all_pass &= check("Archive size = 3", len(arch) == 3)
added_d = arch.add(np.array([8.0, -6.0]), label='D')
all_pass &= check("D rejected (dominated by A)", not added_d)
all_pass &= check("Archive still size 3", len(arch) == 3)
added_e = arch.add(np.array([9.0, 1.0]), label='E')
all_pass &= check("E added", added_e)
labels_in = [e.get('label') for e in arch.get_front()]
all_pass &= check("C removed from archive", 'C' not in labels_in, f"labels={labels_in}")
arch2 = ParetoArchive(max_size=3)
for i in range(4):
    arch2.add(np.array([float(10 - i), float(i)]), idx=i)
all_pass &= check("Archive capped at max_size=3", len(arch2) <= 3)

print("\n=== 4. compute_hypervolume() ===")
ref = np.array([0.0, 0.0])
pts_hv = np.array([[2.0, 1.0], [1.0, 2.0]])
hv = compute_hypervolume(pts_hv, ref)
all_pass &= check("HV of 2D Pareto front > 0", hv > 0, f"hv={hv:.4f}")
hv_single = compute_hypervolume(np.array([[3.0, 2.0]]), ref)
all_pass &= check("HV single point = 3*2=6", abs(hv_single - 6.0) < 1e-9, f"hv={hv_single}")
hv_empty = compute_hypervolume(np.array([[0.5, 0.5]]), ref)
all_pass &= check("HV point at ref barely above = 0.25", abs(hv_empty - 0.25) < 1e-9, f"hv={hv_empty}")
hv_below = compute_hypervolume(np.array([[-1.0, -1.0]]), ref)
all_pass &= check("HV point below ref = 0", hv_below == 0.0, f"hv={hv_below}")
hv_with_dominated = compute_hypervolume(np.array([[3.0, 2.0], [1.0, 1.0]]), ref)
all_pass &= check("HV ignores dominated point", abs(hv_with_dominated - 6.0) < 1e-9, f"hv={hv_with_dominated}")
hv_dup = compute_hypervolume(np.array([[2.0, 1.0], [2.0, 1.0]]), ref)
all_pass &= check("HV duplicate points", hv_dup > 0, f"hv={hv_dup}")

print("\n=== 5. hypervolume_contribution() ===")
pts_c = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
ref_c = np.array([0.0, 0.0])
contrib_0 = hypervolume_contribution(pts_c, 0, ref_c)
contrib_1 = hypervolume_contribution(pts_c, 1, ref_c)
all_pass &= check("HV contribution of each point > 0",
                  contrib_0 > 0 and contrib_1 > 0,
                  f"c0={contrib_0:.3f} c1={contrib_1:.3f}")

print("\n=== 6. prune_by_hypervolume() ===")
vecs = [np.array([3.0, 1.0]), np.array([1.0, 3.0]),
        np.array([2.0, 2.0]), np.array([2.5, 1.5])]
ref_p = np.array([0.0, 0.0])
pruned = prune_by_hypervolume(vecs, max_size=2, reference_point=ref_p)
all_pass &= check("Pruned to max_size=2", len(pruned) <= 2, f"got {len(pruned)}")
try:
    prune_by_hypervolume(vecs, max_size=0, reference_point=ref_p)
    all_pass &= check("max_size=0 raises ValueError", False)
except ValueError:
    all_pass &= check("max_size=0 raises ValueError", True)

print("\n=== 7. select_by_hypervolume() ===")
ref_hv = np.array([-1.0, -1.0])
q_sets = [
    [np.array([10.0, -5.0])],
    [np.array([8.0, -5.0])],
    [np.array([8.0, -7.0])],
]
mask = np.array([True, True, True])
chosen = select_by_hypervolume(q_sets, mask, ref_hv)
all_pass &= check("HV selects best action (idx 0)", chosen == 0, f"chosen={chosen}")

mask2 = np.array([False, True, True])
chosen2 = select_by_hypervolume(q_sets, mask2, ref_hv)
all_pass &= check("HV with mask filters invalid (picks 1)", chosen2 == 1, f"chosen={chosen2}")

q_sets_empty_qset = [[], [np.array([1.0, 1.0])]]
mask3 = np.array([True, True])
chosen3 = select_by_hypervolume(q_sets_empty_qset, mask3, ref_hv)
all_pass &= check("Empty Q-set handled (picks non-empty)", chosen3 == 1, f"chosen={chosen3}")

try:
    select_by_hypervolume(q_sets, np.array([False, False, False]), ref_hv)
    all_pass &= check("No valid action raises ValueError", False)
except ValueError:
    all_pass &= check("No valid action raises ValueError", True)

print("\n=== 8. HLSharedQScorer output shape (N_OBJ_HL=4) ===")
from config import Config
from models.hl_scorer import HLSharedQScorer, N_OBJ_HL
cfg = Config()
net = HLSharedQScorer(cfg)
B = 5
z_g = torch.randn(B, cfg.d_global)
sfc_f = torch.randn(B, cfg.qnet.d_sfc)
pw = torch.randn(B, 2)
out = net(z_g, sfc_f, pw)
all_pass &= check(f"HL Q output shape = [{B}, {N_OBJ_HL}]",
                  out.shape == (B, N_OBJ_HL), f"got {tuple(out.shape)}")
all_pass &= check("N_OBJ_HL == 4", N_OBJ_HL == 4)

print("\n=== 9. LLNodeScorer output shape (N_OBJ=4) ===")
from models.ll_dqn import LLNodeScorer, N_OBJ
N = 50
ll_net = LLNodeScorer(cfg)
zg = torch.randn(N, cfg.d_global)
zp = torch.randn(N, cfg.vgae.d_latent)
zc = torch.randn(N, cfg.vgae.d_latent)
vf = torch.randn(N, cfg.qnet.d_vnf)
sf = torch.randn(N, cfg.qnet.d_sfc)
pw2 = torch.randn(N, 2)
out_ll = ll_net(zg, zp, zc, vf, sf, pw2)
all_pass &= check(f"LL Q output shape = [{N}, {N_OBJ}]",
                  out_ll.shape == (N, N_OBJ), f"got {tuple(out_ll.shape)}")
all_pass &= check("N_OBJ == 4", N_OBJ == 4)

print("\n=== 10. HLAgent buffer stores 4-obj reward vector ===")
from agents.hl_agent import HLAgent
device = torch.device('cpu')
hl_agent = HLAgent(cfg, device)
z_g_cpu = torch.randn(cfg.d_global)
sfc_f_cpu = torch.randn(cfg.qnet.d_sfc)
pw_cpu = torch.tensor([0.6, 0.4])
reward_vec = cfg.pareto.failure_penalty_vector()
hl_agent.store_transition(
    z_g_cpu, sfc_f_cpu, pw_cpu, 0, reward_vec, z_g_cpu, [], pw_cpu, 0.0)
stored = hl_agent.buffer.buf[-1]
stored_rew = stored[4]
all_pass &= check("HL buffer stores 4-obj vector",
                  isinstance(stored_rew, np.ndarray) and stored_rew.shape == (N_OBJ_HL,),
                  f"shape={stored_rew.shape}")

print("\n=== 11. pareto_front() helper ===")
pts_legacy = [(10, -5), (8, -5), (8, -7), (9, -4)]
front = pareto_front(pts_legacy)
front_set = set(map(tuple, front))
all_pass &= check("(9,-4) on front", (9.0, -4.0) in front_set)
all_pass &= check("(8,-7) not on front", (8.0, -7.0) not in front_set)

print("\n=== 12. epsilon=0 in inference (HLAgent) ===")
hl_agent_infer = HLAgent(cfg, device)
hl_agent_infer.epsilon = 0.0
all_pass &= check("epsilon=0 set correctly", hl_agent_infer.epsilon == 0.0)

print("\n=== 13. Config HV reference point ===")
ref = cfg.pareto.hv_reference_point()
all_pass &= check("HV ref point shape=4", ref.shape == (4,), f"shape={ref.shape}")
all_pass &= check("HV ref all negative (below valid range)", np.all(ref < 0), f"ref={ref}")

print("\n=== 14. Config failure penalty vector ===")
fpv = cfg.pareto.failure_penalty_vector()
all_pass &= check("Failure penalty shape=4", fpv.shape == (4,))
all_pass &= check("Failure penalty all negative", np.all(fpv < 0))

print("\n" + "=" * 55)
if all_pass:
    print("  ALL TESTS PASSED")
else:
    print("  SOME TESTS FAILED -- see details above")
print("=" * 55)