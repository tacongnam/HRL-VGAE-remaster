import random
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List
from config import Config


@dataclass
class VNF:
    vnf_id: int
    cpu_req: float


@dataclass
class SFCRequest:
    sfc_id: int
    source: int
    dest: int
    bandwidth: float
    vnf_sequence: List[VNF]
    deadline: int
    arrival_time: int

    @property
    def F_k(self):
        return len(self.vnf_sequence)

    def urgency(self, t: int) -> float:
        return 1.0 / max(1, self.deadline - t)

    def total_cpu_req(self) -> float:
        return sum(v.cpu_req for v in self.vnf_sequence)

    def is_expired(self, t: int) -> bool:
        return self.deadline <= t

    def to_feature_vector(self, t: int) -> np.ndarray:
        return np.array(
            [
                self.source,
                self.dest,
                self.bandwidth,
                self.F_k,
                self.deadline - t,
                self.urgency(t),
                self.total_cpu_req(),
            ],
            dtype=np.float32,
        )


def build_substrate_network(cfg: Config, rng: random.Random = None) -> nx.Graph:
    if rng is None:
        rng = random.Random()
    nc = cfg.network
    G = nx.barabasi_albert_graph(nc.num_nodes, 3, seed=rng.randint(0, 9999))
    G = nx.Graph(G)
    num_fn = int(nc.num_nodes * nc.function_node_ratio)
    fn_nodes = set(rng.sample(range(nc.num_nodes), num_fn))
    for u in G.nodes():
        G.nodes[u]["cpu_total"] = nc.cpu_capacity_mips
        G.nodes[u]["cpu_free"] = nc.cpu_capacity_mips
        G.nodes[u]["is_function_node"] = u in fn_nodes
        G.nodes[u]["cost_cpu"] = rng.uniform(0.5, 1.5)
        G.nodes[u]["cost_ram"] = rng.uniform(0.5, 1.5)
        G.nodes[u]["cost_stor"] = rng.uniform(0.5, 1.5)
        G.nodes[u]["proc_delay"] = rng.uniform(0.5, 2.0) if u in fn_nodes else 0.0
    for u, v in G.edges():
        bw = rng.uniform(nc.bw_min_mbps, nc.bw_max_mbps)
        delay = rng.uniform(nc.delay_min_ms, nc.delay_max_ms)
        G[u][v]["bw_total"] = bw
        G[u][v]["bw_free"] = bw
        G[u][v]["delay"] = delay
    return G


def generate_sfc_requests(
    cfg: Config, current_t: int, next_id_start: int, rng: random.Random = None
) -> List[SFCRequest]:
    if rng is None:
        rng = random.Random()
    sc = cfg.sfc
    nc = cfg.network
    num_arrivals = np.random.poisson(sc.arrival_rate / sc.arrival_interval)
    requests = []
    for i in range(num_arrivals):
        src, dst = rng.sample(range(nc.num_nodes), 2)
        F_k = rng.randint(sc.vnf_len_min, sc.vnf_len_max)
        vnfs = [VNF(j, rng.uniform(sc.cpu_req_min, sc.cpu_req_max)) for j in range(F_k)]
        deadline = current_t + rng.randint(sc.deadline_min, sc.deadline_max)
        bw = rng.uniform(sc.bw_req_min, sc.bw_req_max)
        requests.append(
            SFCRequest(
                sfc_id=next_id_start + i,
                source=src,
                dest=dst,
                bandwidth=bw,
                vnf_sequence=vnfs,
                deadline=deadline,
                arrival_time=current_t,
            )
        )
    return requests


def get_node_features(G: nx.Graph, cfg: Config) -> np.ndarray:
    nc = cfg.network
    feats = []
    for u in sorted(G.nodes()):
        cpu_util = 1.0 - G.nodes[u]["cpu_free"] / G.nodes[u]["cpu_total"]
        is_fn = float(G.nodes[u]["is_function_node"])
        neighbors = list(G.neighbors(u))
        avg_bw_util = 0.0
        if neighbors:
            utils = []
            for v in neighbors:
                e = G[u][v]
                utils.append(1.0 - e["bw_free"] / e["bw_total"])
            avg_bw_util = float(np.mean(utils))
        degree_norm = G.degree(u) / max(1, nc.num_nodes - 1)
        cpu_free_norm = G.nodes[u]["cpu_free"] / nc.cpu_capacity_mips
        cpu_total_norm = G.nodes[u]["cpu_total"] / nc.cpu_capacity_mips
        proc_delay_norm = G.nodes[u].get("proc_delay", 0.0) / 5.0
        cost_cpu_norm = G.nodes[u].get("cost_cpu", 1.0) / 2.0
        feat = [
            cpu_util,
            cpu_free_norm,
            cpu_total_norm,
            is_fn,
            avg_bw_util,
            degree_norm,
            proc_delay_norm,
            cost_cpu_norm,
        ]
        pad = cfg.vgae.d_in - len(feat)
        if pad > 0:
            feat.extend([0.0] * pad)
        feats.append(feat[: cfg.vgae.d_in])
    return np.array(feats, dtype=np.float32)
