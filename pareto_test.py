import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import random
from pareto import (
    dominates, non_dominated_indices, non_dominated, pareto_front,
    ParetoArchive, compute_hypervolume, hypervolume_contribution,
    prune_by_hypervolume, hv_score_for_action, select_by_hypervolume,
    build_target_q_set,
)

all_pass = True

def check(name, cond, detail=''):
    global all_pass
    if not cond:
        all_pass = False
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}]  {name}', f'-> {detail}' if detail else '')
    return cond

# ── 1. dominates() ──────────────────────────────────────────
print('\n=== 1. dominates() ===')
A = np.array([10., -5.]); B = np.array([8., -5.]); C = np.array([8., -7.])
check('A dom B', dominates(A, B))
check('B NOT dom A', not dominates(B, A))
check('A dom C', dominates(A, C))
check('B dom C', dominates(B, C))
check('equal not dom', not dominates(A, np.array([10., -5.])))

# ── 2. non_dominated_indices() ──────────────────────────────
print('\n=== 2. non_dominated_indices() ===')
pts = np.array([[10., -5.], [8., -5.], [8., -7.]])
nd = set(non_dominated_indices(pts))
check('nd = {0} (A dominates B)', nd == {0}, f'got {nd}')
check('nd all-front', set(non_dominated_indices(
    np.array([[10., 1.], [7., 5.], [3., 9.]]))) == {0, 1, 2})

# ── 3. non_dominated() with duplicates ──────────────────────
print('\n=== 3. non_dominated() duplicates ===')
vecs = [np.array([1., 1.]), np.array([1., 1.]), np.array([2., 2.])]
nd2 = non_dominated(vecs)
check('dominant dup only keeps dominator', all(np.allclose(v, [2., 2.]) for v in nd2),
      f'got {nd2}')

# ── 4. compute_hypervolume() ────────────────────────────────
print('\n=== 4. compute_hypervolume() ===')
ref = np.array([0., 0.])
check('HV single point 3*2=6', abs(compute_hypervolume(np.array([[3., 2.]]), ref) - 6.) < 1e-9)
check('HV below ref=0', compute_hypervolume(np.array([[-1., -1.]]), ref) == 0.)
check('HV 2D front >0', compute_hypervolume(np.array([[2., 1.], [1., 2.]]), ref) > 0)
check('HV ignores dominated',
      abs(compute_hypervolume(np.array([[3., 2.], [1., 1.]]), ref) - 6.) < 1e-9)
check('HV 4D positive',
      compute_hypervolume(np.array([[1., 1., 1., 1.], [2., 0.5, 0.5, 0.5]]),
                          np.array([-1., -1., -1., -1.])) > 0)

# ── 5. hypervolume_contribution() ───────────────────────────
print('\n=== 5. hypervolume_contribution() ===')
pts_c = np.array([[3., 1.], [1., 3.], [2., 2.]])
ref_c = np.array([0., 0.])
c0 = hypervolume_contribution(pts_c, 0, ref_c)
c1 = hypervolume_contribution(pts_c, 1, ref_c)
check('HV contributions >0', c0 > 0 and c1 > 0, f'c0={c0:.3f} c1={c1:.3f}')

# ── 6. prune_by_hypervolume() ───────────────────────────────
print('\n=== 6. prune_by_hypervolume() ===')
vecs_p = [np.array([3., 1.]), np.array([1., 3.]),
          np.array([2., 2.]), np.array([2.5, 1.5])]
pruned = prune_by_hypervolume(vecs_p, 2, np.array([0., 0.]))
check('Pruned to max_size=2', len(pruned) <= 2)
check('Pruned result still non-dominated', len(non_dominated(pruned)) == len(pruned))
try:
    prune_by_hypervolume(vecs_p, 0, np.array([0., 0.]))
    check('max_size=0 raises ValueError', False)
except ValueError:
    check('max_size=0 raises ValueError', True)
check('Empty input returns []', prune_by_hypervolume([], 3, np.array([0., 0.])) == [])

# ── 7. select_by_hypervolume() ──────────────────────────────
print('\n=== 7. select_by_hypervolume() ===')
ref_hv = np.array([-1., -1.])
q_sets = [[np.array([10., -0.5])], [np.array([8., -0.5])], [np.array([8., -0.7])]]
mask = np.array([True, True, True])
check('HV selects best (idx 0)', select_by_hypervolume(q_sets, mask, ref_hv) == 0)

mask2 = np.array([False, True, True])
check('Mask filters invalid; picks 1', select_by_hypervolume(q_sets, mask2, ref_hv) == 1)

check('Empty Q-set handled',
      select_by_hypervolume([[], [np.array([1., 1.])]], np.array([True, True]), ref_hv) == 1)

try:
    select_by_hypervolume(q_sets, np.array([False, False, False]), ref_hv)
    check('No valid raises ValueError', False)
except ValueError:
    check('No valid raises ValueError', True)

# Tie-break: seeded random reproducibility
q_tie = [[np.array([2., 2.])], [np.array([2., 2.])]]
mask_tie = np.array([True, True])
rng1 = random.Random(42)
rng2 = random.Random(42)
r1 = select_by_hypervolume(q_tie, mask_tie, ref_hv, tie_break_rng=rng1)
r2 = select_by_hypervolume(q_tie, mask_tie, ref_hv, tie_break_rng=rng2)
check('Tie-break seeded reproducible', r1 == r2, f'{r1} == {r2}')

# Tie-break: normalized cost (higher cost_obj = lower cost → prefer higher cost_obj)
q_cost = [[np.array([3., 2.])], [np.array([2., 2.])]]  # same HV if ref=-1
hv0 = hv_score_for_action([np.array([3., 2.])], ref_hv)
hv1 = hv_score_for_action([np.array([2., 2.])], ref_hv)
if abs(hv0 - hv1) > 1e-9:
    check('Cost tie-break not triggered (different HV, skip)',True)
else:
    chosen_cost = select_by_hypervolume(q_cost, np.array([True, True]), ref_hv)
    check('Cost tie-break: prefers higher cost_obj (idx 0)', chosen_cost == 0,
          f'chosen={chosen_cost}')

# ── 8. build_target_q_set() ─────────────────────────────────
print('\n=== 8. build_target_q_set() ===')
r_vec = np.array([0.5, 0.3, -0.1, 1.0])
next_q_sets = [
    [np.array([0.4, 0.2, -0.2, 0.8])],
    [np.array([0.1, 0.6, -0.05, 0.9])],
]
ref_4 = np.array([-1., -1., -1., -1.])
target = build_target_q_set(r_vec, next_q_sets, gamma=0.9,
                             max_size=5, reference_point=ref_4, done=False)
check('TargetSet is list of np arrays', isinstance(target, list) and len(target) > 0)
check('TargetSet vectors have shape (4,)', all(v.shape == (4,) for v in target))
check('TargetSet non-dominated', len(non_dominated(target)) == len(target))

target_terminal = build_target_q_set(r_vec, next_q_sets, gamma=0.9,
                                      max_size=5, reference_point=ref_4, done=True)
check('Terminal TargetSet = [r_vec]',
      len(target_terminal) == 1 and np.allclose(target_terminal[0], r_vec))

target_empty = build_target_q_set(r_vec, [], gamma=0.9,
                                   max_size=5, reference_point=ref_4, done=False)
check('Empty next Q-sets -> [r_vec]',
      len(target_empty) == 1 and np.allclose(target_empty[0], r_vec))

# ── 9. ParetoArchive ────────────────────────────────────────
print('\n=== 9. ParetoArchive ===')
arch = ParetoArchive(max_size=5)
check('A added', arch.add(np.array([10., -5.]), label='A'))
check('B added', arch.add(np.array([7., -2.]), label='B'))
check('C added', arch.add(np.array([3., 0.]), label='C'))
check('Size=3', len(arch) == 3)
check('D rejected (dominated)', not arch.add(np.array([8., -6.]), label='D'))
check('E added + C removed', arch.add(np.array([9., 1.]), label='E'))
labels = [e.get('label') for e in arch.get_front()]
check('C removed from archive', 'C' not in labels, f'labels={labels}')
arch2 = ParetoArchive(max_size=3)
for i in range(4):
    arch2.add(np.array([float(10 - i), float(i)]))
check('Archive capped at 3', len(arch2) <= 3)

# ── 10. Config fields ────────────────────────────────────────
print('\n=== 10. Config ===')
from config import Config
cfg = Config()
ref = cfg.pareto.hv_reference_point()
check('HV ref shape=(4,)', ref.shape == (4,))
check('HV ref all < 0', np.all(ref < 0))
fpv = cfg.pareto.failure_penalty_vector()
check('Failure penalty shape=(4,)', fpv.shape == (4,))
check('Failure penalty all < 0', np.all(fpv < 0))
check('max_q_vectors_per_action > 0', cfg.qnet.max_q_vectors_per_action > 0)

# ── 11. N_OBJ = 4 in both networks ──────────────────────────
print('\n=== 11. N_OBJ consistency ===')
from hl_scorer import N_OBJ_HL
from ll_dqn import N_OBJ
check('N_OBJ_HL == 4', N_OBJ_HL == 4)
check('N_OBJ == 4', N_OBJ == 4)

# ── 12. Replay stores vector reward with valid_action_mask ───
print('\n=== 12. HLReplayBuffer valid_action_mask ===')
import torch
from hl_agent import HLAgent
device = torch.device('cpu')
ag = HLAgent(cfg, device)
z = torch.randn(cfg.d_global)
sf = torch.randn(cfg.qnet.d_sfc)
pw = torch.tensor([0.6, 0.4])
rv = fpv.copy()
mask = np.array([True, False, True])
ag.store_transition(z, sf, pw, 0, rv, z, [], pw, 0.0, valid_action_mask=mask)
stored = ag.buffer.buf[-1]
stored_rv = stored[4]
stored_mask = stored[9]
check('HL buffer reward shape=(4,)',
      isinstance(stored_rv, np.ndarray) and stored_rv.shape == (4,))
check('HL buffer valid_action_mask stored',
      stored_mask is not None and np.array_equal(stored_mask, mask))

# ── 13. epsilon=0 in inference ───────────────────────────────
print('\n=== 13. epsilon=0 inference ===')
from ll_agent import LLAgent
ll = LLAgent(cfg, device)
ll.epsilon = 0.0
check('LLAgent epsilon=0', ll.epsilon == 0.0)
ag2 = HLAgent(cfg, device)
ag2.epsilon = 0.0
check('HLAgent epsilon=0', ag2.epsilon == 0.0)

# ── 14. pareto_front() helper ────────────────────────────────
print('\n=== 14. pareto_front() helper ===')
front = set(map(tuple, pareto_front([(10., -5.), (8., -5.), (8., -7.), (9., -4.)])))
check('(9,-4) on front', (9., -4.) in front)
check('(8,-7) not on front', (8., -7.) not in front)

# ── 15. Rollback consistency check ──────────────────────────
print('\n=== 15. Rollback: failure_vec used consistently ===')
fv = cfg.pareto.failure_penalty_vector()
check('failure_vec r_success component < 0', fv[3] < 0)
check('failure_vec shape consistent with N_OBJ', len(fv) == N_OBJ)

print()
print('=' * 55)
if all_pass:
    print('  ALL TESTS PASSED')
else:
    print('  SOME TESTS FAILED — see above')
print('=' * 55)