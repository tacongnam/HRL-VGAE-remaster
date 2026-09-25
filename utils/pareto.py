import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from config import Config

N_OBJ = 4


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_rank(candidates: np.ndarray) -> np.ndarray:
    n = len(candidates)
    ranks = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(candidates[j], candidates[i]):
                ranks[i] += 1
    return ranks


def non_dominated_indices(points: np.ndarray) -> List[int]:
    n = len(points)
    keep = []
    for i in range(n):
        dominated = any(j != i and dominates(points[j], points[i]) for j in range(n))
        if not dominated:
            keep.append(i)
    return keep


def non_dominated(vectors: List[np.ndarray]) -> List[np.ndarray]:
    if not vectors:
        return []
    pts = np.array(vectors, dtype=np.float64)
    idx = non_dominated_indices(pts)
    return [vectors[i] for i in idx]


def compute_hypervolume(points: np.ndarray, reference_point: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    ref = np.asarray(reference_point, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    valid = np.all(points > ref, axis=1)
    points = points[valid]
    if len(points) == 0:
        return 0.0
    n_obj = points.shape[1]
    if n_obj == 1:
        return float(np.max(points[:, 0]) - ref[0])
    if n_obj == 2:
        return _hv_2d(points, ref)
    return _hv_wfg(points, ref)


def _hv_2d(points: np.ndarray, ref: np.ndarray) -> float:
    pts = points[np.argsort(points[:, 0])[::-1]]
    hv, prev_y = 0.0, ref[1]
    for p in pts:
        if p[1] > prev_y:
            hv += (p[0] - ref[0]) * (p[1] - prev_y)
            prev_y = p[1]
    return float(hv)


def _hv_wfg(points: np.ndarray, ref: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    if points.shape[1] == 1:
        return float(np.max(points[:, 0]) - ref[0])
    if points.shape[1] == 2:
        return _hv_2d(points, ref)
    nd_idx = non_dominated_indices(points)
    points = points[nd_idx]
    if len(points) == 1:
        return float(np.prod(points[0] - ref))
    order = np.argsort(points[:, 0])[::-1]
    points = points[order]
    n = len(points)
    hv = 0.0
    for i in range(n):
        x_next = points[i + 1][0] if i + 1 < n else ref[0]
        width = points[i][0] - x_next
        if width > 0:
            hv += width * _hv_wfg(points[: i + 1, 1:], ref[1:])
    return float(hv)


def hypervolume_contribution(
    points: np.ndarray, idx: int, reference_point: np.ndarray
) -> float:
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(reference_point, dtype=np.float64)
    hv_all = compute_hypervolume(pts, ref)
    pts_without = np.delete(pts, idx, axis=0)
    hv_without = compute_hypervolume(pts_without, ref) if len(pts_without) > 0 else 0.0
    return hv_all - hv_without


def prune_by_hypervolume(
    vectors: List[np.ndarray], max_size: int, reference_point: np.ndarray
) -> List[np.ndarray]:
    if max_size <= 0:
        raise ValueError("max_size must be > 0")
    if not vectors:
        return []
    vectors = non_dominated(vectors)
    while len(vectors) > max_size:
        pts = np.array(vectors, dtype=np.float64)
        contribs = [
            hypervolume_contribution(pts, i, reference_point)
            for i in range(len(vectors))
        ]
        vectors.pop(int(np.argmin(contribs)))
    return vectors


def hv_score_for_action(
    q_vectors: List[np.ndarray], reference_point: np.ndarray
) -> float:
    if not q_vectors:
        return 0.0
    pts = np.array(q_vectors, dtype=np.float64)
    nd_idx = non_dominated_indices(pts)
    nd_pts = pts[nd_idx]
    return compute_hypervolume(nd_pts, reference_point)


def select_by_hypervolume(
    q_sets: List[List[np.ndarray]],
    valid_mask: np.ndarray,
    reference_point: np.ndarray,
    tie_break_rng: Optional[random.Random] = None,
) -> int:
    valid_indices = [i for i in range(len(q_sets)) if valid_mask[i]]
    if not valid_indices:
        raise ValueError("No valid action available")
    if len(valid_indices) == 1:
        return valid_indices[0]

    hv_scores = np.array(
        [hv_score_for_action(q_sets[i], reference_point) for i in valid_indices],
        dtype=np.float64,
    )

    best_hv = np.max(hv_scores)
    tied = [valid_indices[j] for j, s in enumerate(hv_scores) if s >= best_hv - 1e-12]
    if len(tied) == 1:
        return tied[0]

    # Tie-break 1: lower normalized cost (higher cost objective = index 0)
    best_cost = None
    best_tied = tied[0]
    for idx in tied:
        qs = q_sets[idx]
        if not qs:
            continue
        mean_cost_obj = float(np.mean([v[0] for v in qs]))
        if best_cost is None or mean_cost_obj > best_cost:
            best_cost = mean_cost_obj
            best_tied = idx

    # Check if cost tie-break resolved it
    cost_winners = []
    if best_cost is not None:
        for idx in tied:
            qs = q_sets[idx]
            if qs:
                mco = float(np.mean([v[0] for v in qs]))
                if abs(mco - best_cost) < 1e-12:
                    cost_winners.append(idx)
    if not cost_winners:
        cost_winners = tied

    if len(cost_winners) == 1:
        return cost_winners[0]

    # Tie-break 2: seeded random
    rng = tie_break_rng if tie_break_rng is not None else random.Random(42)
    return rng.choice(cost_winners)


def build_target_q_set(
    reward_vec: np.ndarray,
    next_q_sets: List[List[np.ndarray]],
    gamma: float,
    max_size: int,
    reference_point: np.ndarray,
    done: bool = False,
) -> List[np.ndarray]:
    r = np.asarray(reward_vec, dtype=np.float64)
    if done or not next_q_sets or all(len(qs) == 0 for qs in next_q_sets):
        return [r.copy()]

    all_next: List[np.ndarray] = []
    for qs in next_q_sets:
        all_next.extend(qs)

    if not all_next:
        return [r.copy()]

    nd_next = non_dominated(all_next)
    targets = [r + gamma * np.asarray(q, dtype=np.float64) for q in nd_next]
    targets = non_dominated(targets)
    if len(targets) > max_size:
        targets = prune_by_hypervolume(targets, max_size, reference_point)
    return targets


class ParetoArchive:
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._entries: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, obj: np.ndarray, **metadata) -> bool:
        obj = np.asarray(obj, dtype=np.float64)
        for entry in self._entries:
            if dominates(entry["obj"], obj):
                return False
        self._entries = [e for e in self._entries if not dominates(obj, e["obj"])]
        entry = {"obj": obj.copy()}
        entry.update(metadata)
        self._entries.append(entry)
        if len(self._entries) > self.max_size:
            self._prune_by_crowding()
        return True

    def _prune_by_crowding(self):
        if len(self._entries) <= 1:
            return
        objs = np.array([e["obj"] for e in self._entries])
        n, m = objs.shape
        crowd = np.zeros(n)
        for k in range(m):
            col = objs[:, k]
            order = np.argsort(col)
            range_k = col[order[-1]] - col[order[0]]
            if range_k < 1e-12:
                continue
            crowd[order[0]] += np.inf
            crowd[order[-1]] += np.inf
            for r in range(1, n - 1):
                crowd[order[r]] += (col[order[r + 1]] - col[order[r - 1]]) / range_k
        finite_mask = np.isfinite(crowd)
        if finite_mask.any():
            remove_idx = int(np.where(finite_mask, crowd, np.inf).argmin())
        else:
            remove_idx = 0
        self._entries.pop(remove_idx)

    def get_front(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def get_obj_matrix(self) -> Optional[np.ndarray]:
        if not self._entries:
            return None
        return np.array([e["obj"] for e in self._entries])

    def best_by_weight(self, w: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self._entries:
            return None
        w = np.asarray(w, dtype=np.float64)
        scores = [float(np.dot(e["obj"], w)) for e in self._entries]
        return self._entries[int(np.argmax(scores))]

    def clear(self):
        self._entries.clear()

    def summary(self) -> str:
        if not self._entries:
            return "ParetoArchive(empty)"
        objs = self.get_obj_matrix()
        return (
            f"ParetoArchive(size={len(self._entries)}, "
            f"obj0=[{objs[:,0].min():.3f},{objs[:,0].max():.3f}], "
            f"obj1=[{objs[:,1].min():.3f},{objs[:,1].max():.3f}])"
        )


class ParetoScalarizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.w_accept = cfg.pareto.accept_weight_init
        self.w_cost = cfg.pareto.cost_weight_init
        self.utopia_accept = 0.0
        self.utopia_cost = 0.0
        self.nadir_accept = -cfg.reward.r_fail
        self.nadir_cost = -cfg.reward.r_fail
        self._rho = cfg.pareto.chebyshev_rho

    def current_weight_vector(self) -> np.ndarray:
        return np.array([self.w_accept, self.w_cost], dtype=np.float32)

    def sample_weight(self) -> Tuple[float, float]:
        bins = self.cfg.pareto.num_weight_bins
        k = random.randint(0, bins - 1)
        w = k / max(1, bins - 1)
        return w, 1.0 - w

    def update_utopia(self, r_accept: float, r_cost: float):
        m = self.cfg.pareto.utopia_momentum
        if r_accept > self.utopia_accept:
            self.utopia_accept = r_accept
        else:
            self.utopia_accept = m * self.utopia_accept + (1 - m) * r_accept
        if r_cost > self.utopia_cost:
            self.utopia_cost = r_cost
        else:
            self.utopia_cost = m * self.utopia_cost + (1 - m) * r_cost
        self.nadir_accept = min(self.nadir_accept, r_accept)
        self.nadir_cost = min(self.nadir_cost, r_cost)

    def scalarize(
        self, r_accept: float, r_cost: float, w_accept: float, w_cost: float
    ) -> float:
        range_accept = max(1e-6, self.utopia_accept - self.nadir_accept)
        range_cost = max(1e-6, self.utopia_cost - self.nadir_cost)
        d_accept = w_accept * (self.utopia_accept - r_accept) / range_accept
        d_cost = w_cost * (self.utopia_cost - r_cost) / range_cost
        max_term = max(d_accept, d_cost)
        sum_term = d_accept + d_cost
        return -(max_term + self._rho * sum_term)

    def adapt_weights(
        self,
        recent_accept_ratio: float,
        recent_cost_norm: float,
        target_accept: float = 0.9,
    ):
        rate = self.cfg.pareto.weight_adapt_rate
        gap = abs(recent_accept_ratio - target_accept)
        adaptive_rate = rate * (1.0 + 5.0 * gap)
        if recent_accept_ratio < target_accept:
            self.w_accept = min(0.95, self.w_accept + adaptive_rate)
        else:
            self.w_accept = max(0.05, self.w_accept - adaptive_rate * 0.5)
        self.w_cost = 1.0 - self.w_accept


def is_dominated(a: np.ndarray, b: np.ndarray) -> bool:
    return dominates(np.asarray(b), np.asarray(a))


def pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    pts = np.array(points, dtype=np.float64)
    nd = non_dominated_indices(pts)
    return [tuple(pts[i]) for i in nd]


# Backward-compat alias
def select_by_pareto_dominance(
    q_vectors, valid_mask, pareto_w=None, epsilon_decomp=0.0
):
    n = len(q_vectors)
    valid_indices = [i for i in range(n) if valid_mask[i]]
    if not valid_indices:
        raise ValueError("No valid action available")
    if len(valid_indices) == 1:
        return valid_indices[0]
    valid_qs = np.array([q_vectors[i] for i in valid_indices])
    nd_local = non_dominated_indices(valid_qs)
    nd_global = [valid_indices[i] for i in nd_local]
    if len(nd_global) == 1:
        return nd_global[0]
    nd_qs = np.array([q_vectors[i] for i in nd_global])
    if pareto_w is not None:
        w = np.asarray(pareto_w, dtype=np.float64)
        scores = nd_qs @ w
        best_local = int(np.argmax(scores))
    else:
        best_local = random.randint(0, len(nd_global) - 1)
    return nd_global[best_local]
