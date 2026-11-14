import numpy as np
from typing import List, Optional, Tuple, Union


def normalize_pareto_objectives(
    points: np.ndarray,
    cost_min: float,
    cost_max: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    # AR is in [0,1]; f1 = 1 - AR
    f1 = 1.0 - pts[:, 0]
    # f2 = normalized cost (minimization)
    denom = cost_max - cost_min
    if denom < 1e-12:
        f2 = np.zeros(len(pts))
    else:
        f2 = (pts[:, 1] - cost_min) / denom
    return np.stack([f1, f2], axis=1)


def _make_weight_vectors(n: int = 101) -> np.ndarray:
    w = np.linspace(0.0, 1.0, n)
    return np.stack([w, 1.0 - w], axis=1)


def compute_r2(
    pareto_set_norm: np.ndarray,
    ideal_point: np.ndarray,
    n_weights: int = 101,
) -> float:
    if pareto_set_norm.shape[0] == 0:
        return float("inf")
    lambdas = _make_weight_vectors(n_weights)
    z = np.asarray(ideal_point, dtype=np.float64)
    total = 0.0
    for lam in lambdas:
        diffs = np.abs(pareto_set_norm - z)
        g = np.max(lam * diffs, axis=1)
        total += np.min(g)
    return float(total / len(lambdas))


def compute_igd_plus(
    pareto_set_norm: np.ndarray,
    reference_front_norm: np.ndarray,
) -> float:
    if reference_front_norm.shape[0] == 0 or pareto_set_norm.shape[0] == 0:
        return float("inf")
    total = 0.0
    for r in reference_front_norm:
        diff = pareto_set_norm - r
        contrib = np.sqrt(np.sum(np.maximum(diff, 0.0) ** 2, axis=1))
        total += np.min(contrib)
    return float(total / len(reference_front_norm))


def build_pareto_set_from_scan(scan_results: list) -> np.ndarray:
    if not scan_results:
        return np.empty((0, 2))
    pts = np.array([[r["acc_ratio"], r["deploy_cost"]] for r in scan_results], dtype=np.float64)
    pts = np.unique(pts, axis=0)
    # non-dominated: maximize AR, minimize cost -> transform to minimization
    neg_ar = -pts[:, 0]
    cost = pts[:, 1]
    n = len(pts)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if neg_ar[j] <= neg_ar[i] and cost[j] <= cost[i] and (neg_ar[j] < neg_ar[i] or cost[j] < cost[i]):
                keep[i] = False
                break
    return pts[keep]


def build_empirical_reference_front(pareto_sets: List[np.ndarray]) -> np.ndarray:
    if not pareto_sets:
        return np.empty((0, 2))
    combined = np.concatenate([p for p in pareto_sets if len(p) > 0], axis=0)
    if len(combined) == 0:
        return np.empty((0, 2))
    combined = np.unique(combined, axis=0)
    neg_ar = -combined[:, 0]
    cost = combined[:, 1]
    n = len(combined)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if neg_ar[j] <= neg_ar[i] and cost[j] <= cost[i] and (neg_ar[j] < neg_ar[i] or cost[j] < cost[i]):
                keep[i] = False
                break
    return combined[keep]


def compute_hv_normalized(
    pareto_set_norm: np.ndarray,
    ref_point: np.ndarray = None,
) -> float:
    from utils.pareto import compute_hypervolume, non_dominated_2d_fast
    if pareto_set_norm.shape[0] == 0:
        return 0.0
    if ref_point is None:
        ref_point = np.array([1.1, 1.1])
    # For HV in minimization space, convert to maximization: negate
    pts_max = -pareto_set_norm
    ref_max = -np.asarray(ref_point)
    nd = non_dominated_2d_fast(pts_max)
    return compute_hypervolume(nd, ref_max)