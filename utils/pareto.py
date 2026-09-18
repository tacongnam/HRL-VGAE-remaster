import random
import numpy as np
from typing import List, Tuple
from config import Config

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
        self.utopia_accept = m * self.utopia_accept + (1 - m) * r_accept if r_accept <= self.utopia_accept else r_accept
        self.utopia_cost = m * self.utopia_cost + (1 - m) * r_cost if r_cost <= self.utopia_cost else r_cost
        self.nadir_accept = min(self.nadir_accept, r_accept)
        self.nadir_cost = min(self.nadir_cost, r_cost)

    def scalarize(self, r_accept: float, r_cost: float, w_accept: float, w_cost: float) -> float:
        range_accept = max(1e-6, self.utopia_accept - self.nadir_accept)
        range_cost = max(1e-6, self.utopia_cost - self.nadir_cost)
        d_accept = w_accept * (self.utopia_accept - r_accept) / range_accept
        d_cost = w_cost * (self.utopia_cost - r_cost) / range_cost
        max_term = max(d_accept, d_cost)
        sum_term = d_accept + d_cost
        return -(max_term + self._rho * sum_term)

    def adapt_weights(self, recent_accept_ratio: float, recent_cost_norm: float, target_accept: float = 0.9):
        rate = self.cfg.pareto.weight_adapt_rate
        gap = abs(recent_accept_ratio - target_accept)
        adaptive_rate = rate * (1.0 + 5.0 * gap)
        if recent_accept_ratio < target_accept:
            self.w_accept = min(0.95, self.w_accept + adaptive_rate)
        else:
            self.w_accept = max(0.05, self.w_accept - adaptive_rate * 0.5)
        self.w_cost = 1.0 - self.w_accept

def is_dominated(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(b >= a) and np.any(b > a))

def pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if is_dominated(pts[i], pts[j]):
                keep[i] = False
                break
    return [tuple(p) for p in pts[keep]]