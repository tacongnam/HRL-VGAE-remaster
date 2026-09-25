import json
import re
import math
import os
import glob
import hashlib
from typing import List, Tuple, Dict
import numpy as np
import networkx as nx
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.graph_generator import SFCRequest, VNF


def _node_id(name) -> int:
    if isinstance(name, int):
        return name
    match = re.search(r"\d+", str(name))
    if match:
        return int(match.group())
    try:
        return int(name)
    except ValueError:
        raise ValueError(f"Cannot extract node ID from: '{name}'")


def compute_topology_id(V_raw: dict, E_raw: list) -> str:
    node_sig = sorted(V_raw.keys(), key=lambda x: _node_id(x))
    edge_sig = sorted((_node_id(e["u"]), _node_id(e["v"])) for e in E_raw)
    payload = json.dumps({"n": node_sig, "e": edge_sig}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def parse_episode(path: str) -> Tuple[nx.Graph, List[SFCRequest], List[Dict], str]:
    with open(path, "r") as f:
        data = json.load(f)
    V_raw = data["V"]
    E_raw = data["E"]
    F_raw = data["F"]
    R_raw = data["R"]
    topology_id = compute_topology_id(V_raw, E_raw)
    G = nx.Graph()
    node_names = sorted(V_raw.keys(), key=lambda x: _node_id(x))
    for name in node_names:
        uid = _node_id(name)
        nd = V_raw[name]
        is_server = bool(nd.get("server", False))
        cpu = float(nd["c_v"]) if is_server else 0.0
        ram = float(nd.get("r_v", 0)) if is_server else 0.0
        G.add_node(
            uid,
            cpu_total=cpu,
            cpu_free=cpu,
            ram_total=ram,
            ram_free=ram,
            is_function_node=is_server,
            proc_delay=float(nd.get("d_v", 0.0)) if is_server else 0.0,
            cost_cpu=float(nd.get("cost_c", 1.0)) if is_server else 0.0,
            cost_ram=float(nd.get("cost_r", 1.0)) if is_server else 0.0,
            cost_stor=float(nd.get("cost_h", 1.0)) if is_server else 0.0,
        )
    for e in E_raw:
        u = _node_id(e["u"])
        v = _node_id(e["v"])
        bw = float(e["b_l"])
        dl = float(e["d_l"])
        G.add_edge(u, v, bw_total=bw, bw_free=bw, delay=dl)
    requests: List[SFCRequest] = []
    for idx, r in enumerate(R_raw):
        arrival = int(math.floor(r["T"]))
        d_max = int(r["d_max"])
        deadline = arrival + d_max
        src = _node_id(r["st_r"])
        dst = _node_id(r["d_r"])
        bw = float(r["b_r"])
        vnf_seq: List[VNF] = []
        for fi, f_idx in enumerate(r["F_r"]):
            ftype = F_raw[f_idx]
            vnf_seq.append(VNF(vnf_id=fi, cpu_req=float(ftype["c_f"])))
        requests.append(
            SFCRequest(
                sfc_id=idx,
                source=src,
                dest=dst,
                bandwidth=bw,
                vnf_sequence=vnf_seq,
                deadline=deadline,
                arrival_time=arrival,
            )
        )
    requests.sort(key=lambda r: r.arrival_time)
    return G, requests, F_raw, topology_id


def get_node_features_from_graph(G: nx.Graph, d_in: int = 16) -> np.ndarray:
    nodes = sorted(G.nodes())
    n = len(nodes)
    node_idx = {u: i for i, u in enumerate(nodes)}
    cpu_totals = np.array([G.nodes[u]["cpu_total"] for u in nodes], dtype=np.float32)
    cpu_frees = np.array([G.nodes[u]["cpu_free"] for u in nodes], dtype=np.float32)
    is_fn = np.array(
        [float(G.nodes[u]["is_function_node"]) for u in nodes], dtype=np.float32
    )
    proc_delays = np.array(
        [G.nodes[u].get("proc_delay", 0.0) for u in nodes], dtype=np.float32
    )
    cost_cpus = np.array(
        [G.nodes[u].get("cost_cpu", 0.0) for u in nodes], dtype=np.float32
    )
    cpu_max = cpu_totals.max() if cpu_totals.max() > 0 else 1.0
    cpu_util = np.where(
        cpu_totals > 0, 1.0 - cpu_frees / np.where(cpu_totals > 0, cpu_totals, 1.0), 0.0
    )
    cpu_free_norm = cpu_frees / cpu_max
    cpu_total_norm = cpu_totals / cpu_max
    degrees = np.array([G.degree(u) for u in nodes], dtype=np.float32)
    degree_norm = degrees / max(1, n - 1)
    avg_bw_util = np.zeros(n, dtype=np.float32)
    for u, v, d in G.edges(data=True):
        util = 1.0 - d["bw_free"] / d["bw_total"]
        i, j = node_idx[u], node_idx[v]
        avg_bw_util[i] += util
        avg_bw_util[j] += util
    avg_bw_util = np.where(degrees > 0, avg_bw_util / degrees, 0.0)
    proc_delay_norm = proc_delays / 5.0
    cost_cpu_norm = cost_cpus / 2.0
    feats = np.stack(
        [
            cpu_util,
            cpu_free_norm,
            cpu_total_norm,
            is_fn,
            avg_bw_util,
            degree_norm,
            proc_delay_norm,
            cost_cpu_norm,
        ],
        axis=1,
    )
    if d_in > 8:
        pad = np.zeros((n, d_in - 8), dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=1)
    return feats[:, :d_in]


def discover_episodes(data_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_dir, "episode_*.json")))


def train_test_split(
    paths: List[str], train_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    n_train = max(1, int(len(paths) * train_ratio))
    return paths[:n_train], paths[n_train:]
