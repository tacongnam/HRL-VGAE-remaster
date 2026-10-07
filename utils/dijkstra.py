import heapq
import math
import numpy as np
import networkx as nx
from typing import List, Optional, Tuple


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
        if u in visited or d > dist.get(u, math.inf):
            continue
        visited.add(u)
        if u == target:
            break
        for v, e in G[u].items():
            if v in visited or e["bw_free"] < req_bw:
                continue
            rho = np.clip(1.0 - e["bw_free"] / e["bw_total"], 0.0, 1.0 - 1e-9)
            w = e["delay"] + lambda_penalty * (math.exp(rho * log_omega) - 1.0)
            new_dist = d + w
            if new_dist < dist.get(v, math.inf):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    if target not in dist:
        return None
    path, cur = [], target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def compute_path_cost_delay(G: nx.Graph, path: List[int]) -> Tuple[float, float]:
    if len(path) < 2:
        return 0.0, 0.0
    cost, delay = 0.0, 0.0
    for i in range(len(path) - 1):
        e = G[path[i]][path[i + 1]]
        cost += 1.0 - e["bw_free"] / e["bw_total"]
        delay += e["delay"]
    return float(cost), float(delay)


def compute_load_std(G: nx.Graph) -> float:
    utils = [1.0 - e["bw_free"] / e["bw_total"] for _, _, e in G.edges(data=True)]
    return float(np.std(utils, ddof=1)) if len(utils) >= 2 else 0.0
