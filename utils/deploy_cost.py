import numpy as np
import networkx as nx
from typing import List, Tuple
from config import Config

def node_deploy_cost(G: nx.Graph, node: int, cpu_req: float, cfg: Config) -> Tuple[float, float, float]:
    nd = G.nodes[node]
    cost_cpu = nd.get('cost_cpu', 1.0) * cpu_req
    ram_req = cpu_req * cfg.deploy_cost.ram_per_cpu_unit
    cost_ram = nd.get('cost_ram', 1.0) * ram_req
    cost_storage = nd.get('cost_stor', 1.0) * cfg.deploy_cost.storage_per_vnf
    return cost_cpu, cost_ram, cost_storage

def link_deploy_cost(bw_req: float, path_len: int, cfg: Config) -> float:
    return bw_req * path_len

def init_cost(path_len: int, cfg: Config) -> float:
    return cfg.deploy_cost.init_cost_per_hop * path_len

def total_deploy_cost(G: nx.Graph, allocations: List[Tuple], cfg: Config) -> float:
    dc = cfg.deploy_cost
    total = 0.0
    for node, path, cpu_req, bw in allocations:
        path_len = max(0, len(path) - 1)
        if cpu_req > 0.0:
            c_cpu, c_ram, c_stor = node_deploy_cost(G, node, cpu_req, cfg)
            total += dc.w_cpu * c_cpu + dc.w_ram * c_ram + dc.w_storage * c_stor
        total += dc.w_bw * link_deploy_cost(bw, path_len, cfg)
        total += dc.w_init * init_cost(path_len, cfg)
    return total

def deploy_cost_breakdown(G: nx.Graph, allocations: List[Tuple], cfg: Config) -> dict:
    dc = cfg.deploy_cost
    parts = {'cpu': 0.0, 'ram': 0.0, 'storage': 0.0, 'bw': 0.0, 'init': 0.0}
    for node, path, cpu_req, bw in allocations:
        path_len = max(0, len(path) - 1)
        if cpu_req > 0.0:
            c_cpu, c_ram, c_stor = node_deploy_cost(G, node, cpu_req, cfg)
            parts['cpu'] += dc.w_cpu * c_cpu
            parts['ram'] += dc.w_ram * c_ram
            parts['storage'] += dc.w_storage * c_stor
        parts['bw'] += dc.w_bw * link_deploy_cost(bw, path_len, cfg)
        parts['init'] += dc.w_init * init_cost(path_len, cfg)
    return parts
