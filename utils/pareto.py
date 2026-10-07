import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from config import Config


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool((a[0] >= b[0] and a[1] >= b[1]) and (a[0] > b[0] or a[1] > b[1]))


def non_dominated_2d_fast(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    if n <= 1:
        return points
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    pts = points[order]
    keep = np.zeros(n, dtype=bool)
    max_y = -np.inf
    for i in range(n):
        y = pts[i, 1]
        if y > max_y:
            keep[i] = True
            max_y = y
    return pts[keep]


def non_dominated_indices(points: np.ndarray) -> List[int]:
    n = points.shape[0]
    if n <= 1:
        return list(range(n))
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    pts = points[order]
    survivors = []
    max_y = -np.inf
    for idx, (_, y) in enumerate(pts):
        if y > max_y:
            survivors.append(order[idx])
            max_y = y
    return sorted(survivors)


def non_dominated(vectors: List[np.ndarray]) -> List[np.ndarray]:
    if not vectors:
        return []
    pts = np.asarray(vectors, dtype=np.float64)
    if pts.ndim == 1:
        return [vectors[0]]
    if len(vectors) == 1:
        return vectors
    res = non_dominated_2d_fast(pts)
    return [res[i] for i in range(res.shape[0])]


def _hv_2d(points: np.ndarray, ref: np.ndarray) -> float:
    order = np.argsort(-points[:, 0])
    pts = points[order]
    hv = 0.0
    prev_y = ref[1]
    for i in range(pts.shape[0]):
        y = pts[i, 1]
        if y > prev_y:
            hv += (pts[i, 0] - ref[0]) * (y - prev_y)
            prev_y = y
    return float(hv)


def compute_hypervolume(points: np.ndarray, reference_point: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    ref = np.asarray(reference_point, dtype=np.float64)
    valid_mask = (pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])
    pts = pts[valid_mask]
    if pts.shape[0] == 0:
        return 0.0
    return _hv_2d(pts, ref)


def hypervolume_contribution(points: np.ndarray, idx: int, reference_point: np.ndarray) -> float:
    hv_all = compute_hypervolume(points, reference_point)
    pts_without = np.delete(points, idx, axis=0)
    if pts_without.shape[0] == 0:
        return hv_all
    return hv_all - compute_hypervolume(pts_without, reference_point)


def prune_by_hypervolume_fast(points: np.ndarray, max_size: int, reference_point: np.ndarray) -> np.ndarray:
    if points.shape[0] <= max_size:
        return points
    pts = points.copy()
    ref = np.asarray(reference_point, dtype=np.float64)
    while pts.shape[0] > max_size:
        n = pts.shape[0]
        hv_all = _hv_2d(pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])], ref)
        min_c = np.inf
        min_i = 0
        for i in range(n):
            sub = np.delete(pts, i, axis=0)
            sub_valid = sub[(sub[:, 0] > ref[0]) & (sub[:, 1] > ref[1])]
            c = hv_all - (_hv_2d(sub_valid, ref) if sub_valid.shape[0] > 0 else 0.0)
            if c < min_c:
                min_c = c
                min_i = i
        pts = np.delete(pts, min_i, axis=0)
    return pts


def prune_by_hypervolume(
    vectors: List[np.ndarray], max_size: int, reference_point: np.ndarray
) -> List[np.ndarray]:
    if not vectors:
        return []
    pts = non_dominated_2d_fast(np.asarray(vectors, dtype=np.float64))
    res = prune_by_hypervolume_fast(pts, max_size, reference_point)
    return [res[i] for i in range(res.shape[0])]


def hv_score_for_action(q_vectors: List[np.ndarray], reference_point: np.ndarray) -> float:
    if not q_vectors:
        return 0.0
    pts = np.asarray(q_vectors, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    pts_nd = non_dominated_2d_fast(pts)
    return compute_hypervolume(pts_nd, reference_point)


def select_by_hypervolume(
    q_sets: List[List[np.ndarray]],
    valid_mask: np.ndarray,
    reference_point: np.ndarray,
    tie_break_rng: Optional[random.Random] = None,
) -> int:
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        raise ValueError("No valid action available")
    if valid_indices.size == 1:
        return int(valid_indices[0])

    hv_scores = np.array(
        [hv_score_for_action(q_sets[i], reference_point) for i in valid_indices],
        dtype=np.float64,
    )
    best_hv = np.max(hv_scores)
    tied = valid_indices[hv_scores >= best_hv - 1e-12]
    if len(tied) == 1:
        return int(tied[0])

    best_cost = -np.inf
    best_tied = tied[0]
    for idx in tied:
        if q_sets[idx]:
            mco = float(np.mean([v[0] for v in q_sets[idx]]))
            if mco > best_cost:
                best_cost = mco
                best_tied = idx

    cost_winners = [
        idx
        for idx in tied
        if q_sets[idx] and abs(float(np.mean([v[0] for v in q_sets[idx]])) - best_cost) < 1e-12
    ]
    if not cost_winners:
        cost_winners = list(tied)
    if len(cost_winners) == 1:
        return int(cost_winners[0])

    rng = tie_break_rng or random.Random(42)
    return int(rng.choice(cost_winners))


def build_target_q_set_fast(
    r_vec: np.ndarray,
    cand_q_matrix: np.ndarray,
    gamma: float,
    max_size: int,
    reference_point: np.ndarray,
    done: bool = False,
) -> np.ndarray:
    if done or cand_q_matrix.size == 0:
        return r_vec.reshape(1, -1)
    nd_next = non_dominated_2d_fast(cand_q_matrix)
    targets = r_vec + gamma * nd_next
    targets_nd = non_dominated_2d_fast(targets)
    if targets_nd.shape[0] > max_size:
        return prune_by_hypervolume_fast(targets_nd, max_size, reference_point)
    return targets_nd


def build_target_q_set(
    reward_vec: np.ndarray,
    next_q_sets: List[List[np.ndarray]],
    gamma: float,
    max_size: int,
    reference_point: np.ndarray,
    done: bool = False,
) -> List[np.ndarray]:
    r = np.asarray(reward_vec, dtype=np.float64)
    if done or not next_q_sets:
        return [r]
    all_next = [q for qs in next_q_sets for q in qs if len(q) > 0]
    if not all_next:
        return [r]
    pts = np.asarray(all_next, dtype=np.float64)
    res = build_target_q_set_fast(r, pts, gamma, max_size, reference_point, done=done)
    return [res[i] for i in range(res.shape[0])]


class ParetoArchive:
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._entries: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, obj: np.ndarray, **metadata) -> bool:
        obj = np.asarray(obj, dtype=np.float64)
        for e in self._entries:
            if dominates(e["obj"], obj):
                return False
        self._entries = [e for e in self._entries if not dominates(obj, e["obj"])]
        self._entries.append({"obj": obj.copy(), **metadata})
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
            crowd[order[0]] = np.inf
            crowd[order[-1]] = np.inf
            for r in range(1, n - 1):
                crowd[order[r]] += (col[order[r + 1]] - col[order[r - 1]]) / range_k
        self._entries.pop(int(np.argmin(crowd)))

    def get_front(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def summary(self) -> str:
        if not self._entries:
            return "ParetoArchive(empty)"
        objs = np.array([e["obj"] for e in self._entries])
        return f"ParetoArchive(size={len(self._entries)}, obj0=[{objs[:,0].min():.3f},{objs[:,0].max():.3f}], obj1=[{objs[:,1].min():.3f},{objs[:,1].max():.3f}])"


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

    def sample_weight(self) -> Tuple[float, float]:
        k = random.randint(0, self.cfg.pareto.num_weight_bins - 1)
        w = k / max(1, self.cfg.pareto.num_weight_bins - 1)
        return w, 1.0 - w

    def update_utopia(self, r_accept: float, r_cost: float):
        m = self.cfg.pareto.utopia_momentum
        self.utopia_accept = max(self.utopia_accept, r_accept) if r_accept > self.utopia_accept else m * self.utopia_accept + (1.0 - m) * r_accept
        self.utopia_cost = max(self.utopia_cost, r_cost) if r_cost > self.utopia_cost else m * self.utopia_cost + (1.0 - m) * r_cost
        self.nadir_accept = min(self.nadir_accept, r_accept)
        self.nadir_cost = min(self.nadir_cost, r_cost)

    def scalarize(self, r_accept: float, r_cost: float, w_accept: float, w_cost: float) -> float:
        da = w_accept * (self.utopia_accept - r_accept) / max(1e-6, self.utopia_accept - self.nadir_accept)
        dc = w_cost * (self.utopia_cost - r_cost) / max(1e-6, self.utopia_cost - self.nadir_cost)
        return -(max(da, dc) + self._rho * (da + dc))

    def adapt_weights(self, recent_accept_ratio: float, recent_cost_norm: float, target_accept: float = 0.9):
        rate = self.cfg.pareto.weight_adapt_rate * (1.0 + 5.0 * abs(recent_accept_ratio - target_accept))
        if recent_accept_ratio < target_accept:
            self.w_accept = min(0.95, self.w_accept + rate)
        else:
            self.w_accept = max(0.05, self.w_accept - rate * 0.5)
        self.w_cost = 1.0 - self.w_accept


def pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    pts = np.asarray(points, dtype=np.float64)
    idx = non_dominated_indices(pts)
    return [tuple(pts[i]) for i in idx]


def normalize_objectives(
    points: np.ndarray,
    bounds_min: Optional[np.ndarray] = None,
    bounds_max: Optional[np.ndarray] = None,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    b_min = np.min(pts, axis=0) if bounds_min is None else np.asarray(bounds_min, dtype=np.float64)
    b_max = np.max(pts, axis=0) if bounds_max is None else np.asarray(bounds_max, dtype=np.float64)
    diff = b_max - b_min
    diff = np.where(diff < 1e-12, 1.0, diff)
    return (pts - b_min) / diff


def coverage_metric(
    set_a: Union[List[np.ndarray], np.ndarray],
    set_b: Union[List[np.ndarray], np.ndarray],
    maximize: bool = True,
) -> float:
    pts_a = np.unique(np.asarray(set_a, dtype=np.float64), axis=0) if len(set_a) > 0 else np.empty((0, 2))
    pts_b = np.unique(np.asarray(set_b, dtype=np.float64), axis=0) if len(set_b) > 0 else np.empty((0, 2))
    if pts_b.shape[0] == 0 or pts_a.shape[0] == 0:
        return 0.0
    if not maximize:
        pts_a = -pts_a
        pts_b = -pts_b
    dominated_count = sum(bool(np.any(np.all(pts_a >= b, axis=1))) for b in pts_b)
    return float(dominated_count / pts_b.shape[0])


def spacing_metric(points: Union[List[np.ndarray], np.ndarray], normalize: bool = True) -> float:
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0) if len(points) > 0 else np.empty((0, 2))
    n = pts.shape[0]
    if n <= 1:
        return 0.0
    if normalize:
        pts = normalize_objectives(pts)
    d = np.zeros(n)
    for i in range(n):
        diffs = np.sum(np.abs(pts - pts[i]), axis=1)
        diffs[i] = np.inf
        d[i] = np.min(diffs)
    d_mean = np.mean(d)
    return float(np.sqrt(np.sum((d - d_mean) ** 2) / (n - 1)))


def spread_metric(
    points: Union[List[np.ndarray], np.ndarray],
    extreme_points: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> float:
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0) if len(points) > 0 else np.empty((0, 2))
    n = pts.shape[0]
    if n <= 1:
        return 0.0
    if normalize:
        b_min = np.min(pts, axis=0)
        b_max = np.max(pts, axis=0)
        if extreme_points is not None:
            b_min = np.minimum(b_min, np.min(extreme_points, axis=0))
            b_max = np.maximum(b_max, np.max(extreme_points, axis=0))
        pts_norm = normalize_objectives(pts, b_min, b_max)
        ext_norm = normalize_objectives(extreme_points, b_min, b_max) if extreme_points is not None else None
    else:
        pts_norm = pts
        ext_norm = np.asarray(extreme_points, dtype=np.float64) if extreme_points is not None else None

    n, m = pts_norm.shape
    d = np.zeros(n)
    for i in range(n):
        diffs = np.sqrt(np.sum((pts_norm - pts_norm[i]) ** 2, axis=1))
        diffs[i] = np.inf
        d[i] = np.min(diffs)
    d_bar = np.mean(d)
    d_e_sum = 0.0
    if ext_norm is not None and len(ext_norm) >= m:
        for ep in ext_norm:
            d_e_sum += np.min(np.sqrt(np.sum((pts_norm - ep) ** 2, axis=1)))
    numerator = d_e_sum + np.sum(np.abs(d - d_bar))
    denominator = d_e_sum + n * d_bar
    return float(numerator / denominator) if denominator >= 1e-12 else 0.0