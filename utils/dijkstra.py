import heapq
import math

import networkx as nx
import numpy as np


def custom_dijkstra(
    G: nx.Graph,
    source: int,
    target: int,
    req_bw: float,
    omega_bw: float = 100.0,
    lambda_penalty: float = 1.0,
) -> list[int] | None:
    if source == target:
        return [source]
    dist = {source: 0.0}
    prev = {source: None}
    heap = [(0.0, source)]
    visited = set()
    log_omega = math.log(omega_bw)
    adj = G._adj

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited or d > dist.get(u, math.inf):
            continue
        visited.add(u)
        if u == target:
            break
        u_nbrs = adj[u]
        for v, e in u_nbrs.items():
            if v in visited:
                continue
            bw_free = e["bw_free"]
            if bw_free < req_bw:
                continue
            rho = 1.0 - bw_free / e["bw_total"]
            if rho < 0.0:
                rho = 0.0
            elif rho > 0.999999999:
                rho = 0.999999999
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
    path.reverse()
    return path


def compute_path_cost_delay(G: nx.Graph, path: list[int]) -> tuple[float, float]:
    p_len = len(path)
    if p_len < 2:
        return 0.0, 0.0
    cost, delay = 0.0, 0.0
    adj = G._adj
    for i in range(p_len - 1):
        e = adj[path[i]][path[i + 1]]
        cost += 1.0 - e["bw_free"] / e["bw_total"]
        delay += e["delay"]
    return float(cost), float(delay)


def compute_load_std(G: nx.Graph) -> float:
    edges = G.edges(data=True)
    if len(edges) < 2:
        return 0.0
    utils = [1.0 - d["bw_free"] / d["bw_total"] for _, _, d in edges]
    return float(np.std(utils, ddof=1))
