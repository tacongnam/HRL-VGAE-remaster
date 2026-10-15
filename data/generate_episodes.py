"""
Generate N episode JSON files from a fixed topology JSON.

Each episode shares the same topology {V, E, F} but has a different
request workload {R}, making them suitable for fair multi-algorithm comparison.

The topology JSON is produced by generate_topology.py and has keys V, E, F
(and optionally an empty R).  This script fills in R and writes episode_NNNN.json.

Usage
-----
python data/generate_episodes.py \\
    --topology data/topologies/nsf_centers.json \\
    --num-episodes 50 \\
    --sim-duration 500 \\
    --seed 42 \\
    --out-dir data/episodes/nsf_centers/train

python data/generate_episodes.py \\
    --topology data/topologies/nsf_centers.json \\
    --num-episodes 20 \\
    --sim-duration 500 \\
    --seed 100 \\
    --out-dir data/episodes/nsf_centers/test

Determinism guarantee
---------------------
Given the same --topology + --seed + --num-episodes + --sim-duration, the
output is always identical.  Each episode gets its own child seed derived
from the master seed, so episodes are independent but reproducible.
"""

import argparse
import heapq
import json
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import islice

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCHEME_RANDOM_RATE = {"easy": 0.0, "normal": 0.5, "hard": 1.0}


def _node_id(name) -> int:
    if isinstance(name, int):
        return name
    match = re.search(r"\d+", str(name))
    if match:
        return int(match.group())
    return int(name)


def _default_request_config(num_nodes: int, num_vnf_types: int):
    """Baseline generate_test() request params (scheme=easy, duration=50)."""
    return {
        "arrival_rate": 50.0,
        "bandwidth_r": (0.0, 10.0),
        "delay_ratio": (1.0, 1.2),
        "segments": min(8, max(1, num_vnf_types)),
        "scheme": "easy",
        "vnf_len_min": 1,
        "vnf_len_max": min(8, num_vnf_types),
        "bw_req_min": 0.0,
        "bw_req_max": 10.0,
    }


def _infer_request_config(R_existing: list, num_nodes: int, num_vnf_types: int):
    """Infer request generation parameters from an existing R list."""
    cfg = _default_request_config(num_nodes, num_vnf_types)
    if not R_existing:
        return cfg
    times = sorted(r["T"] for r in R_existing)
    bws = [r["b_r"] for r in R_existing]
    lens = [len(r["F_r"]) for r in R_existing]
    if len(times) > 1:
        duration = max(times) - min(times)
        cfg["arrival_rate"] = float(len(times) / max(duration, 1.0))
    cfg["bw_req_min"] = float(min(bws))
    cfg["bw_req_max"] = float(max(bws))
    cfg["bandwidth_r"] = (cfg["bw_req_min"], cfg["bw_req_max"])
    cfg["vnf_len_min"] = int(min(lens))
    cfg["vnf_len_max"] = int(max(lens))
    cfg["segments"] = cfg["vnf_len_max"]
    return cfg


def _build_work_graph(V: dict, E: list) -> nx.Graph:
    G = nx.Graph()
    for name, nd in V.items():
        uid = _node_id(name)
        is_server = bool(nd.get("server", False))
        cpu = float(nd["c_v"]) if is_server else 0.0
        ram = float(nd.get("r_v", 0.0)) if is_server else 0.0
        hdd = float(nd.get("h_v", 0.0)) if is_server else 0.0
        G.add_node(
            uid,
            server=is_server,
            cpu_free=cpu,
            ram_free=ram,
            hdd_free=hdd,
            proc_delay=float(nd.get("d_v", 0.0)) if is_server else 0.0,
        )
    for e in E:
        u, v = _node_id(e["u"]), _node_id(e["v"])
        bw = float(e["b_l"])
        G.add_edge(u, v, bw_free=bw, delay=float(e["d_l"]))
    return G


def _vnf_delay_at(ftype: dict, node: int) -> float:
    raw = ftype.get("d_f") or {}
    if node in raw:
        return float(raw[node])
    if str(node) in raw:
        return float(raw[str(node)])
    key = f"v{node}"
    if key in raw:
        return float(raw[key])
    return 0.0


class _GenManager:
    """Minimal current-manager: per-operation CPU/RAM/HDD/BW release."""

    def __init__(self, G: nx.Graph):
        self.G = G
        self.t = 0
        self._op_heap = []
        self._op_counter = 0

    def copy_fresh(self) -> "_GenManager":
        return _GenManager(self.G.copy())

    def advance(self):
        self.t += 1
        self._release()

    def _release(self):
        t_float = float(self.t)
        adj = self.G._adj
        nodes = self.G.nodes
        while self._op_heap and self._op_heap[0][0] < t_float:
            _, _, spec = heapq.heappop(self._op_heap)
            if spec[0] == "link":
                _, path, bw = spec
                for i in range(len(path) - 1):
                    adj[path[i]][path[i + 1]]["bw_free"] += bw
            elif spec[0] == "cpu":
                _, node, cpu_req, ram_req, hdd_req = spec
                nd = nodes[node]
                nd["cpu_free"] += cpu_req
                nd["ram_free"] += ram_req
                nd["hdd_free"] += hdd_req

    def check_deploy_vnf(self, node: int, ftype: dict, tent_cpu, tent_ram, tent_hdd) -> bool:
        nd = self.G.nodes[node]
        if nd["cpu_free"] - tent_cpu[node] < float(ftype["c_f"]):
            return False
        if nd["ram_free"] - tent_ram[node] < float(ftype.get("r_f", 0.0)):
            return False
        hdd_cap = nd["hdd_free"]
        if hdd_cap > 0.0 and hdd_cap - tent_hdd[node] < float(ftype.get("h_f", 0.0)):
            return False
        return True

    def remaining_bw(self, u: int, v: int) -> float:
        return self.G[u][v]["bw_free"]

    def deploy(self, sfc: dict):
        t = float(sfc["start"])
        adj = self.G._adj
        nodes = self.G.nodes
        bw = sfc["br"]
        for node, path, ftype in sfc["placements"]:
            if len(path) > 1:
                link_delay = sum(
                    adj[path[i]][path[i + 1]]["delay"] for i in range(len(path) - 1)
                )
                finish_route = t + link_delay
                for i in range(len(path) - 1):
                    adj[path[i]][path[i + 1]]["bw_free"] -= bw
                heapq.heappush(
                    self._op_heap,
                    (finish_route, self._op_counter, ("link", tuple(path), bw)),
                )
                self._op_counter += 1
                t = finish_route
            if ftype is not None:
                cpu_req = float(ftype["c_f"])
                ram_req = float(ftype.get("r_f", 0.0))
                hdd_req = float(ftype.get("h_f", 0.0))
                nd = nodes[node]
                nd["cpu_free"] -= cpu_req
                nd["ram_free"] -= ram_req
                nd["hdd_free"] -= hdd_req
                proc = nd["proc_delay"] + _vnf_delay_at(ftype, node)
                finish_serve = t + proc
                heapq.heappush(
                    self._op_heap,
                    (
                        finish_serve,
                        self._op_counter,
                        ("cpu", node, cpu_req, ram_req, hdd_req),
                    ),
                )
                self._op_counter += 1
                t = finish_serve
        dest_path = sfc["dest_path"]
        if dest_path is not None and len(dest_path) > 1:
            link_delay = sum(
                adj[dest_path[i]][dest_path[i + 1]]["delay"]
                for i in range(len(dest_path) - 1)
            )
            finish_route = t + link_delay
            for i in range(len(dest_path) - 1):
                adj[dest_path[i]][dest_path[i + 1]]["bw_free"] -= bw
            heapq.heappush(
                self._op_heap,
                (finish_route, self._op_counter, ("link", tuple(dest_path), bw)),
            )
            self._op_counter += 1


def _choose_close_node(G: nx.Graph, current: int, candidates: list, rng: np.random.Generator):
    """Baseline ``choose_close_node``: linear distance weighting.

    The original generator weights a candidate at hop-distance ``d`` by
    ``max(d) - d + 1`` (a linear ramp favouring close nodes) instead of the old
    ``1/d`` inverse weighting.  ``1/d`` collapsed the distribution onto the single
    nearest node, packing VNF chains too tightly so they consumed almost no link
    bandwidth.  The linear ramp spreads chains further, matching baseline load.
    """
    if not candidates:
        return None
    dists = []
    for n in candidates:
        try:
            d = nx.shortest_path_length(G, current, n)
        except nx.NetworkXNoPath:
            d = 1e9
        dists.append(float(d))
    max_d = max(dists)
    weights = np.array([max_d - d + 1.0 for d in dists], dtype=np.float64)
    w = np.clip(weights, 1e-9, None)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return int(candidates[int(rng.integers(0, len(candidates)))])
    return int(candidates[int(rng.choice(len(candidates), p=w / s))])


def _create_path(G: nx.Graph, src: int, dst: int, rng: np.random.Generator, k: int = 5):
    if src == dst:
        return [src]
    try:
        paths = list(islice(nx.shortest_simple_paths(G, src, dst), k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if not paths:
        return None
    return list(paths[int(rng.integers(0, len(paths)))])


def _sfc_min_delay(G: nx.Graph, placements, dest_path) -> float:
    total = 0.0
    for node, path, ftype in placements:
        for i in range(len(path) - 1):
            total += G[path[i]][path[i + 1]]["delay"]
        if ftype is not None:
            total += G.nodes[node]["proc_delay"] + _vnf_delay_at(ftype, node)
    if dest_path is not None:
        for i in range(len(dest_path) - 1):
            total += G[dest_path[i]][dest_path[i + 1]]["delay"]
    return total


def create_feasible_sfc(
    manager: _GenManager,
    F: list,
    rng: np.random.Generator,
    arrival: float,
    start: int,
    segments: int,
    bandwidth: tuple,
    delay_ratio: tuple,
):
    G = manager.G
    server_nodes = [u for u, nd in G.nodes(data=True) if nd["server"]]
    switch_nodes = [u for u, nd in G.nodes(data=True) if not nd["server"]]
    if not server_nodes or not switch_nodes or not F:
        return None
    n_vnf = int(rng.integers(1, max(1, segments) + 1))
    st = int(switch_nodes[int(rng.integers(0, len(switch_nodes)))])
    current = st
    Fr = []
    placements = []
    tent_cpu = defaultdict(float)
    tent_ram = defaultdict(float)
    tent_hdd = defaultdict(float)
    tent_edges = Counter()
    for _ in range(n_vnf):
        type_ids = [i for i in range(len(F)) if not Fr or i != Fr[-1]]
        if not type_ids:
            return None
        placed = False
        for _try in range(20):
            fi = int(type_ids[int(rng.integers(0, len(type_ids)))])
            ftype = F[fi]
            candidates = [
                n
                for n in server_nodes
                if manager.check_deploy_vnf(n, ftype, tent_cpu, tent_ram, tent_hdd)
            ]
            if not candidates:
                continue
            chosen = _choose_close_node(G, current, candidates, rng)
            path = _create_path(G, current, chosen, rng)
            if path is None:
                continue
            Fr.append(fi)
            placements.append((chosen, path, ftype))
            tent_cpu[chosen] += float(ftype["c_f"])
            tent_ram[chosen] += float(ftype.get("r_f", 0.0))
            tent_hdd[chosen] += float(ftype.get("h_f", 0.0))
            for i in range(len(path) - 1):
                tent_edges[tuple(sorted((path[i], path[i + 1])))] += 1
            current = chosen
            placed = True
            break
        if not placed:
            return None
    dest = _choose_close_node(G, current, switch_nodes, rng)
    dest_path = _create_path(G, current, dest, rng)
    if dest_path is None:
        return None
    for i in range(len(dest_path) - 1):
        tent_edges[tuple(sorted((dest_path[i], dest_path[i + 1])))] += 1
    min_delay = _sfc_min_delay(G, placements, dest_path)
    bw_lo, bw_hi = bandwidth
    if tent_edges:
        max_bw = min(
            bw_hi,
            min(
                manager.remaining_bw(u, v) / cnt for (u, v), cnt in tent_edges.items()
            ),
        )
    else:
        max_bw = bw_hi
    if max_bw < bw_lo:
        return None
    ratio = float(rng.uniform(delay_ratio[0], delay_ratio[1]))
    dr = (start - arrival) + min_delay * ratio
    br = float(rng.uniform(bw_lo, max_bw)) if max_bw > bw_lo else float(max_bw)
    return {
        "arrival": float(arrival),
        "start": int(start),
        "st": st,
        "dest": dest,
        "Fr": Fr,
        "br": br,
        "dr": dr,
        "placements": placements,
        "dest_path": dest_path,
    }


def generate_requests(
    V: dict,
    E: list,
    F: list,
    sim_duration: int,
    rng: np.random.Generator,
    arrival_rate: float = 50.0,
    bandwidth: tuple = (0.0, 10.0),
    delay_ratio: tuple = (1.0, 1.2),
    segments: int = None,
    scheme: str = "easy",
    **_ignored,
) -> list:
    """Baseline generate_requests: Poisson arrivals, Exponential start, feasible SFC."""
    if segments is None:
        segments = max(1, len(F))
    random_rate = _SCHEME_RANDOM_RATE.get(scheme, 0.0)
    G0 = _build_work_graph(V, E)
    manager = _GenManager(G0)
    base_manager = _GenManager(G0.copy())

    arrival_counts = rng.poisson(arrival_rate, size=max(1, int(sim_duration)))
    arrivals = [t for t in range(int(sim_duration)) for _ in range(int(arrival_counts[t]))]
    if not arrivals:
        return []
    offsets = rng.exponential(scale=1.0, size=len(arrivals)).astype(int)
    starts = [int(a + o) for a, o in zip(arrivals, offsets)]
    timeline = defaultdict(list)
    for arr, st in zip(arrivals, starts):
        timeline[st].append(arr)
    lifetime = max(starts)

    R = []
    for st in range(lifetime + 1):
        while manager.t < st:
            manager.advance()
            while base_manager.t < manager.t:
                base_manager.advance()
        for arr in timeline[st]:
            use_base = rng.random() < random_rate
            if use_base:
                # Hard-style: ignore already-used resources (may be infeasible).
                sfc = create_feasible_sfc(
                    base_manager, F, rng, arr, st, segments, bandwidth, delay_ratio
                )
                deploy_on_manager = False
            else:
                sfc = create_feasible_sfc(
                    manager, F, rng, arr, st, segments, bandwidth, delay_ratio
                )
                if sfc is None:
                    # Silent-drop fix: the current manager is saturated here. The
                    # original baseline forces the request to exist rather than
                    # dropping it, so retry on the pristine base_manager (which
                    # ignores current load).  The request is recorded but is NOT
                    # deployed on ``manager`` (it is infeasible there), so the
                    # manager's resource accounting stays uncorrupted.
                    sfc = create_feasible_sfc(
                        base_manager, F, rng, arr, st, segments, bandwidth, delay_ratio
                    )
                    deploy_on_manager = False
                else:
                    deploy_on_manager = True
            if sfc is None:
                continue
            if deploy_on_manager:
                manager.deploy(sfc)
            R.append(
                {
                    "T": round(float(sfc["arrival"]), 10),
                    "st_r": f"v{sfc['st']}",
                    "d_r": f"v{sfc['dest']}",
                    "F_r": [int(x) for x in sfc["Fr"]],
                    "b_r": round(float(sfc["br"]), 4),
                    "d_max": round(float(sfc["dr"]), 10),
                }
            )
    R.sort(key=lambda r: r["T"])
    return R


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate episode JSONs from a fixed topology"
    )
    parser.add_argument(
        "--topology",
        type=str,
        required=True,
        help="Path to topology JSON {V,E,F} (from generate_topology.py)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of episode files to generate",
    )
    parser.add_argument(
        "--sim-duration", type=int, default=50, help="Simulation duration (time units)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Master seed — deterministic for same args"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for episode_NNNN.json files",
    )
    parser.add_argument(
        "--infer-cfg",
        action="store_true",
        help="Infer request cfg from existing R in topology file (if any)",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="easy",
        choices=["easy", "normal", "hard"],
        help="Baseline random_rate: easy=0, normal=0.5, hard=1.0",
    )
    parser.add_argument("--arrival-rate", type=float, default=None)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--bw-min", type=float, default=None)
    parser.add_argument("--bw-max", type=float, default=None)
    parser.add_argument("--delay-ratio-min", type=float, default=1.0)
    parser.add_argument("--delay-ratio-max", type=float, default=1.2)
    args = parser.parse_args()

    # Load topology
    with open(args.topology) as f:
        topo = json.load(f)
    V = topo["V"]
    E = topo["E"]
    F = topo.get("F", [])
    R_existing = topo.get("R", [])

    num_nodes = len(V)
    num_vnf_types = len(F)

    if args.infer_cfg and R_existing:
        req_cfg = _infer_request_config(R_existing, num_nodes, num_vnf_types)
        print("Inferred request config from topology R list.")
    else:
        req_cfg = _default_request_config(num_nodes, num_vnf_types)
        print("Using baseline request config.")

    req_cfg["scheme"] = args.scheme
    if args.arrival_rate is not None:
        req_cfg["arrival_rate"] = args.arrival_rate
    if args.segments is not None:
        req_cfg["segments"] = args.segments
    bw_lo = args.bw_min if args.bw_min is not None else req_cfg["bandwidth_r"][0]
    bw_hi = args.bw_max if args.bw_max is not None else req_cfg["bandwidth_r"][1]
    req_cfg["bandwidth_r"] = (bw_lo, bw_hi)
    req_cfg["delay_ratio"] = (args.delay_ratio_min, args.delay_ratio_max)

    print(
        f"Topology  : {args.topology}  ({num_nodes} nodes, {len(E)} edges, {num_vnf_types} VNF types)"
    )
    print(
        f"Config    : arrival_rate={req_cfg['arrival_rate']:.3f}  scheme={req_cfg['scheme']}"
        f"  T={args.sim_duration}  episodes={args.num_episodes}  seed={args.seed}"
    )
    print(f"Output    : {args.out_dir}/\n")

    os.makedirs(args.out_dir, exist_ok=True)
    master_rng = np.random.default_rng(args.seed)

    for ep in range(args.num_episodes):
        ep_seed = int(master_rng.integers(0, 2**31))
        ep_rng = np.random.default_rng(ep_seed)
        R = generate_requests(
            V,
            E,
            F,
            args.sim_duration,
            ep_rng,
            arrival_rate=req_cfg["arrival_rate"],
            bandwidth=req_cfg["bandwidth_r"],
            delay_ratio=req_cfg["delay_ratio"],
            segments=req_cfg["segments"],
            scheme=req_cfg["scheme"],
        )
        episode = {"V": V, "E": E, "F": F, "R": R}
        out_path = os.path.join(args.out_dir, f"episode_{ep:04d}.json")
        with open(out_path, "w") as f:
            json.dump(episode, f, indent=2)
        print(
            f"  Ep {ep + 1:4d}/{args.num_episodes} | requests={len(R):5d} | seed={ep_seed} | {os.path.basename(out_path)}"
        )

    print(f"\nDone. {args.num_episodes} episodes → {os.path.abspath(args.out_dir)}/")
    print(
        "Reproducibility: re-run with same --topology + --seed + --num-episodes to get identical files."
    )


if __name__ == "__main__":
    main()
