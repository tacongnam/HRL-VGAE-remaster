import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from config import Config


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_indices(points: np.ndarray) -> List[int]:
    n = len(points)
    return [
        i
        for i in range(n)
        if not any(j != i and dominates(points[j], points[i]) for j in range(n))
    ]


def non_dominated(vectors: List[np.ndarray]) -> List[np.ndarray]:
    if not vectors:
        return []
    idx = non_dominated_indices(np.array(vectors, dtype=np.float64))
    return [vectors[i] for i in idx]


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
    points = points[non_dominated_indices(points)]
    if len(points) == 1:
        return float(np.prod(points[0] - ref))
    points = points[np.argsort(points[:, 0])[::-1]]
    n, hv = len(points), 0.0
    for i in range(n):
        width = points[i][0] - (points[i + 1][0] if i + 1 < n else ref[0])
        if width > 0:
            hv += width * _hv_wfg(points[: i + 1, 1:], ref[1:])
    return float(hv)


def compute_hypervolume(points: np.ndarray, reference_point: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    ref = np.asarray(reference_point, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    points = points[np.all(points > ref, axis=1)]
    if len(points) == 0:
        return 0.0
    return _hv_2d(points, ref) if points.shape[1] == 2 else _hv_wfg(points, ref)


def hypervolume_contribution(
    points: np.ndarray, idx: int, reference_point: np.ndarray
) -> float:
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(reference_point, dtype=np.float64)
    hv_all = compute_hypervolume(pts, ref)
    pts_without = np.delete(pts, idx, axis=0)
    return hv_all - (
        compute_hypervolume(pts_without, ref) if len(pts_without) > 0 else 0.0
    )


def prune_by_hypervolume(
    vectors: List[np.ndarray], max_size: int, reference_point: np.ndarray
) -> List[np.ndarray]:
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
    return compute_hypervolume(pts[non_dominated_indices(pts)], reference_point)


def select_by_hypervolume(
    q_sets: List[List[np.ndarray]],
    valid_mask: np.ndarray,
    reference_point: np.ndarray,
    tie_break_rng: Optional[random.Random] = None,
) -> int:
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
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

    best_cost, best_tied = None, tied[0]
    for idx in tied:
        if q_sets[idx]:
            mco = float(np.mean([v[0] for v in q_sets[idx]]))
            if best_cost is None or mco > best_cost:
                best_cost, best_tied = mco, idx

    cost_winners = (
        [
            idx
            for idx in tied
            if q_sets[idx]
            and abs(float(np.mean([v[0] for v in q_sets[idx]])) - best_cost) < 1e-12
        ]
        if best_cost is not None
        else tied
    )
    if len(cost_winners) == 1:
        return cost_winners[0]

    rng = tie_break_rng or random.Random(42)
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
    if done or not next_q_sets:
        return [r.copy()]

    all_next = [q for qs in next_q_sets for q in qs]
    if not all_next:
        return [r.copy()]

    targets = [
        r + gamma * np.asarray(q, dtype=np.float64) for q in non_dominated(all_next)
    ]
    targets = non_dominated(targets)
    return (
        prune_by_hypervolume(targets, max_size, reference_point)
        if len(targets) > max_size
        else targets
    )


class ParetoArchive:
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._entries: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, obj: np.ndarray, **metadata) -> bool:
        obj = np.asarray(obj, dtype=np.float64)
        if any(dominates(e["obj"], obj) for e in self._entries):
            return False
        self._entries = [e for e in self._entries if not dominates(obj, e["obj"])]
        entry = {"obj": obj.copy(), **metadata}
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
        self.w_accept, self.w_cost = (
            cfg.pareto.accept_weight_init,
            cfg.pareto.cost_weight_init,
        )
        self.utopia_accept, self.utopia_cost = 0.0, 0.0
        self.nadir_accept, self.nadir_cost = -cfg.reward.r_fail, -cfg.reward.r_fail
        self._rho = cfg.pareto.chebyshev_rho

    def sample_weight(self) -> Tuple[float, float]:
        k = random.randint(0, self.cfg.pareto.num_weight_bins - 1)
        w = k / max(1, self.cfg.pareto.num_weight_bins - 1)
        return w, 1.0 - w

    def update_utopia(self, r_accept: float, r_cost: float):
        m = self.cfg.pareto.utopia_momentum
        self.utopia_accept = (
            max(self.utopia_accept, r_accept)
            if r_accept > self.utopia_accept
            else m * self.utopia_accept + (1 - m) * r_accept
        )
        self.utopia_cost = (
            max(self.utopia_cost, r_cost)
            if r_cost > self.utopia_cost
            else m * self.utopia_cost + (1 - m) * r_cost
        )
        self.nadir_accept, self.nadir_cost = min(self.nadir_accept, r_accept), min(
            self.nadir_cost, r_cost
        )

    def scalarize(
        self, r_accept: float, r_cost: float, w_accept: float, w_cost: float
    ) -> float:
        da = (
            w_accept
            * (self.utopia_accept - r_accept)
            / max(1e-6, self.utopia_accept - self.nadir_accept)
        )
        dc = (
            w_cost
            * (self.utopia_cost - r_cost)
            / max(1e-6, self.utopia_cost - self.nadir_cost)
        )
        return -(max(da, dc) + self._rho * (da + dc))

    def adapt_weights(
        self,
        recent_accept_ratio: float,
        recent_cost_norm: float,
        target_accept: float = 0.9,
    ):
        rate = self.cfg.pareto.weight_adapt_rate * (
            1.0 + 5.0 * abs(recent_accept_ratio - target_accept)
        )
        self.w_accept = (
            min(0.95, self.w_accept + rate)
            if recent_accept_ratio < target_accept
            else max(0.05, self.w_accept - rate * 0.5)
        )
        self.w_cost = 1.0 - self.w_accept


def pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    pts = np.array(points, dtype=np.float64)
    return [tuple(pts[i]) for i in non_dominated_indices(pts)]


# ===========================================================================
# Multi-Objective Evaluation Metrics: C-Metric, Delta (Spread), Spacing
# ===========================================================================


def normalize_objectives(
    points: np.ndarray,
    bounds_min: Optional[np.ndarray] = None,
    bounds_max: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize objective vectors to [0, 1]^M range."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    b_min = (
        np.min(pts, axis=0)
        if bounds_min is None
        else np.asarray(bounds_min, dtype=np.float64)
    )
    b_max = (
        np.max(pts, axis=0)
        if bounds_max is None
        else np.asarray(bounds_max, dtype=np.float64)
    )

    diff = b_max - b_min
    diff = np.where(diff < 1e-12, 1.0, diff)
    return (pts - b_min) / diff


def coverage_metric(
    set_a: Union[List[np.ndarray], np.ndarray],
    set_b: Union[List[np.ndarray], np.ndarray],
    maximize: bool = True,
) -> float:
    """
    Compute C-metric (Coverage) C(A, B): Ratio of solutions in B dominated by or equal to
    at least one solution in A.

    C(A, B) = |{b in B | exists a in A : a >= b (dominate or equal)}| / |B|
    """
    pts_a = (
        np.unique(np.asarray(set_a, dtype=np.float64), axis=0)
        if len(set_a) > 0
        else np.empty((0, 2))
    )
    pts_b = (
        np.unique(np.asarray(set_b, dtype=np.float64), axis=0)
        if len(set_b) > 0
        else np.empty((0, 2))
    )

    if len(pts_b) == 0:
        return 0.0
    if len(pts_a) == 0:
        return 0.0

    if not maximize:
        pts_a = -pts_a
        pts_b = -pts_b

    dominated_count = 0
    for b in pts_b:
        # a dominates or equals b in maximization sense
        is_dominated = False
        for a in pts_a:
            if np.all(a >= b):
                is_dominated = True
                break
        if is_dominated:
            dominated_count += 1

    return float(dominated_count / len(pts_b))


def spacing_metric(
    points: Union[List[np.ndarray], np.ndarray], normalize: bool = True
) -> float:
    """
    Compute Schott's Spacing metric (S).
    Measures the uniformity of the spread of points in the Pareto front.
    S = 0 indicates perfectly equidistant solutions.
    """
    pts = (
        np.unique(np.asarray(points, dtype=np.float64), axis=0)
        if len(points) > 0
        else np.empty((0, 2))
    )
    if len(pts) <= 1:
        return 0.0

    if normalize:
        pts = normalize_objectives(pts)

    n = len(pts)
    d = np.zeros(n)
    for i in range(n):
        diffs = pts - pts[i]
        dists = np.sum(
            np.abs(diffs), axis=1
        )  # Manhattan distance per standard Schott metric
        dists[i] = np.inf
        d[i] = np.min(dists)

    d_mean = np.mean(d)
    return float(np.sqrt(np.sum((d - d_mean) ** 2) / (n - 1)))


def spread_metric(
    points: Union[List[np.ndarray], np.ndarray],
    extreme_points: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> float:
    """
    Compute Deb's Generalized Spread / Delta-metric.
    Delta = (sum_{m=1}^M d_m^e + sum_{i=1}^{|P|} |d_i - d_bar|) /
            (sum_{m=1}^M d_m^e + |P| * d_bar)
    """
    pts = (
        np.unique(np.asarray(points, dtype=np.float64), axis=0)
        if len(points) > 0
        else np.empty((0, 2))
    )
    if len(pts) <= 1:
        return 0.0

    if normalize:
        b_min = np.min(pts, axis=0)
        b_max = np.max(pts, axis=0)
        if extreme_points is not None:
            b_min = np.minimum(b_min, np.min(extreme_points, axis=0))
            b_max = np.maximum(b_max, np.max(extreme_points, axis=0))
        pts_norm = normalize_objectives(pts, b_min, b_max)
        ext_norm = (
            normalize_objectives(extreme_points, b_min, b_max)
            if extreme_points is not None
            else None
        )
    else:
        pts_norm = pts
        ext_norm = (
            np.asarray(extreme_points, dtype=np.float64)
            if extreme_points is not None
            else None
        )

    n, m = pts_norm.shape

    # Euclidean nearest neighbor distance
    d = np.zeros(n)
    for i in range(n):
        diffs = pts_norm - pts_norm[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists[i] = np.inf
        d[i] = np.min(dists)

    d_bar = np.mean(d)

    # Extreme points distance
    if ext_norm is not None and len(ext_norm) >= m:
        d_e_sum = 0.0
        for ep in ext_norm:
            d_e = np.min(np.sqrt(np.sum((pts_norm - ep) ** 2, axis=1)))
            d_e_sum += d_e
    else:
        # Empirical extreme points: endpoints on each objective dimension
        d_e_sum = 0.0
        for dim in range(m):
            min_pt = pts_norm[np.argmin(pts_norm[:, dim])]
            max_pt = pts_norm[np.argmax(pts_norm[:, dim])]
            # distance of boundary
            d_e_sum += 0.0  # boundary included within empirical points

    numerator = d_e_sum + np.sum(np.abs(d - d_bar))
    denominator = d_e_sum + n * d_bar

    if denominator < 1e-12:
        return 0.0
    return float(numerator / denominator)
