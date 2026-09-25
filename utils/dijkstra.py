import heapq
import math
import numpy as np
import networkx as nx
from typing import List, Optional, Dict, Tuple


def custom_dijkstra(
    G: nx.Graph,
    source: int,
    target: int,
    req_bw: float,
    omega_bw: float = 100.0,
    lambda_penalty: float = 1.0,
) -> Optional[List[int]]:
    if source == target:
        return [source]
    dist = {source: 0.0}
    prev = {source: None}
    heap = [(0.0, source)]
    visited = set()
    log_omega = math.log(omega_bw)
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        if d > dist.get(u, math.inf):
            continue
        visited.add(u)
        if u == target:
            break
        for v, e in G[u].items():
            if v in visited:
                continue
            if e["bw_free"] < req_bw:
                continue
            rho = 1.0 - e["bw_free"] / e["bw_total"]
            if rho < 0.0:
                rho = 0.0
            elif rho > 1.0 - 1e-9:
                rho = 1.0 - 1e-9
            w = e["delay"] + lambda_penalty * (math.exp(rho * log_omega) - 1.0)
            new_dist = d + w
            if new_dist < dist.get(v, math.inf):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))
    if target not in dist:
        return None
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def path_edge_arrays(
    G: nx.Graph, path: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bw_free = np.empty(len(path) - 1, dtype=np.float64)
    bw_total = np.empty(len(path) - 1, dtype=np.float64)
    delay = np.empty(len(path) - 1, dtype=np.float64)
    for i in range(len(path) - 1):
        e = G[path[i]][path[i + 1]]
        bw_free[i] = e["bw_free"]
        bw_total[i] = e["bw_total"]
        delay[i] = e["delay"]
    return bw_free, bw_total, delay


def compute_path_cost(G: nx.Graph, path: List[int]) -> float:
    if len(path) < 2:
        return 0.0
    bw_free, bw_total, _ = path_edge_arrays(G, path)
    return float(np.sum(1.0 - bw_free / bw_total))


def compute_path_delay(G: nx.Graph, path: List[int]) -> float:
    if len(path) < 2:
        return 0.0
    _, _, delay = path_edge_arrays(G, path)
    return float(np.sum(delay))


def compute_path_cost_delay(G: nx.Graph, path: List[int]) -> Tuple[float, float]:
    if len(path) < 2:
        return 0.0, 0.0
    bw_free, bw_total, delay = path_edge_arrays(G, path)
    cost = float(np.sum(1.0 - bw_free / bw_total))
    dl = float(np.sum(delay))
    return cost, dl


def compute_load_std(G: nx.Graph) -> float:
    utils = np.array(
        [1.0 - e["bw_free"] / e["bw_total"] for _, _, e in G.edges(data=True)],
        dtype=np.float64,
    )
    if len(utils) < 2:
        return 0.0
    return float(utils.std(ddof=1))


def compute_step_reward(
    G: nx.Graph, path: List[int], alpha: float, beta: float, gamma_load: float
) -> float:
    cost, dl = compute_path_cost_delay(G, path)
    return -(alpha * cost + beta * dl + gamma_load * compute_load_std(G))


class PathCache:
    def __init__(self, capacity: int = 2000):
        self._cache: Dict[Tuple, Optional[List[int]]] = {}
        self._capacity = capacity
        self._order: List[Tuple] = []

    def get(self, src: int, dst: int, bw_key: float) -> Optional[List[int]]:
        return self._cache.get((src, dst, bw_key), None)

    def put(self, src: int, dst: int, bw_key: float, path: Optional[List[int]]):
        key = (src, dst, bw_key)
        if key not in self._cache:
            if len(self._order) >= self._capacity:
                old = self._order.pop(0)
                self._cache.pop(old, None)
            self._order.append(key)
        self._cache[key] = path

    def invalidate(self):
        self._cache.clear()
        self._order.clear()
