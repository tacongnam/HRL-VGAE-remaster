"""
Generate a fixed topology JSON {V, E, F} that can be shared across many
episode files.  Supports two modes:

1.  --mode file   : Load a real topology from CSV nodes + TXT edges
                    (same format as nsf/cogent/conus in the old project).
                    Requires --node-csv and --edge-txt.
2.  --mode random : Generate a Barabasi-Albert random graph (default, no
                    external files needed).

Output: a single JSON file with keys V, E, F, R (R is always empty here).
Episode JSONs later fill in R using generate_episodes.py.

Usage examples
--------------
# Real topology (nsf):
python data/generate_topology.py \\
    --mode file \\
    --node-csv data/topology/nsf_nodes.csv \\
    --edge-txt data/topology/nsf_edges.txt \\
    --server-dist centers \\
    --vnf-types 10 \\
    --seed 42 \\
    --out data/topologies/nsf_centers.json

# Random graph:
python data/generate_topology.py \\
    --mode random \\
    --num-nodes 50 \\
    --seed 42 \\
    --out data/topologies/random_50.json
"""

import argparse
import json
import math
import os
import random
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers shared with old project's network_gen / graph_gen
# ---------------------------------------------------------------------------


def _file_to_graph_nsf(node_csv: str, edge_txt: str) -> nx.Graph:
    """Load NSF-style topology: CSV node positions + space-separated edge list."""
    import csv

    G = nx.Graph()
    # Nodes
    with open(node_csv) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for i, row in enumerate(reader):
            G.add_node(i)
    # Edges — format: "u v weight" or just "u v"
    with open(edge_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            G.add_edge(u, v)
    return G


def _select_server_nodes(G: nx.Graph, distribution: str, rng: random.Random):
    nodes = list(G.nodes())
    nodes.sort(key=lambda n: -G.degree(n))
    n = len(nodes)
    if distribution == "uniform":
        return set(rng.sample(nodes, max(1, int(0.3 * n))))
    elif distribution == "urban":
        return set(nodes[: max(1, int(0.3 * n))])
    elif distribution == "rural":
        return set(nodes[max(0, int(0.7 * n)) :])
    elif distribution == "centers":
        return set(nodes[: max(1, int(0.1 * n))])
    else:
        raise ValueError(f"Unknown server distribution: {distribution}")


def _assign_network_attrs(
    G: nx.Graph,
    server_nodes: set,
    rng: random.Random,
    bw_range=(80, 120),
    cpu_range=(30, 50),
    ram_range=(30, 50),
    cpu_cost_range=(100, 200),
    ram_cost_range=(50, 100),
    hdd_cost_range=(10, 30),
    delay_node_range=(0.1, 0.5),
):
    """Assign resource capacities and costs to nodes and edges."""
    max_dist = 1.0
    # Try to use node positions for link delay (not always available)
    edges = list(G.edges())
    for u, v in edges:
        bw = rng.uniform(*bw_range)
        dl = rng.uniform(0.01, 0.3)
        G[u][v]["bw_total"] = bw
        G[u][v]["bw_free"] = bw
        G[u][v]["delay"] = dl

    for u in G.nodes():
        is_fn = u in server_nodes
        G.nodes[u]["is_function_node"] = is_fn
        G.nodes[u]["cpu_total"] = rng.uniform(*cpu_range) if is_fn else 0.0
        G.nodes[u]["cpu_free"] = G.nodes[u]["cpu_total"]
        G.nodes[u]["ram_total"] = rng.uniform(*ram_range) if is_fn else 0.0
        G.nodes[u]["ram_free"] = G.nodes[u]["ram_total"]
        G.nodes[u]["proc_delay"] = rng.uniform(*delay_node_range) if is_fn else 0.0
        G.nodes[u]["cost_cpu"] = rng.uniform(*cpu_cost_range) if is_fn else 0.0
        G.nodes[u]["cost_ram"] = rng.uniform(*ram_cost_range) if is_fn else 0.0
        G.nodes[u]["cost_stor"] = rng.uniform(*hdd_cost_range) if is_fn else 0.0


def _generate_vnfs(
    G: nx.Graph,
    server_nodes: set,
    vnf_types: int,
    rng: random.Random,
    cpu_range=(1, 5),
    delay_range=(0.1, 1.0),
):
    """Generate VNF type catalog as list of dicts compatible with old/new JSON schema."""
    vnfs = []
    for _ in range(vnf_types):
        cpu = rng.uniform(*cpu_range)
        ram = rng.uniform(*cpu_range)  # simplified: same range
        hdd = rng.uniform(*cpu_range)
        df = {str(u): rng.uniform(*delay_range) for u in server_nodes}
        vnfs.append({"c_f": cpu, "r_f": ram, "h_f": hdd, "d_f": df})
    return vnfs


def _graph_to_json(G: nx.Graph, F_raw: list) -> dict:
    """Convert nx.Graph to the {V,E,F,R} JSON schema."""
    V = {}
    for u in sorted(G.nodes()):
        nd = G.nodes[u]
        if nd.get("is_function_node", False):
            V[f"v{u}"] = {
                "server": True,
                "c_v": nd["cpu_total"],
                "r_v": nd.get("ram_total", 0.0),
                "h_v": 0.0,
                "d_v": nd.get("proc_delay", 0.0),
                "cost_c": nd.get("cost_cpu", 1.0),
                "cost_r": nd.get("cost_ram", 1.0),
                "cost_h": nd.get("cost_stor", 1.0),
            }
        else:
            V[f"v{u}"] = {"server": False}

    E = []
    for u, v, d in G.edges(data=True):
        E.append({"u": f"v{u}", "v": f"v{v}", "b_l": d["bw_total"], "d_l": d["delay"]})

    return {"V": V, "E": E, "F": F_raw, "R": []}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate a fixed topology JSON for NFV experiments"
    )
    parser.add_argument(
        "--mode",
        choices=["file", "random"],
        default="random",
        help="'file': load real topology; 'random': Barabasi-Albert",
    )
    # file mode
    parser.add_argument(
        "--node-csv",
        type=str,
        default="",
        help="[file mode] Path to node CSV (nsf_nodes.csv etc.)",
    )
    parser.add_argument(
        "--edge-txt",
        type=str,
        default="",
        help="[file mode] Path to edge TXT (nsf_edges.txt etc.)",
    )
    parser.add_argument(
        "--server-dist",
        type=str,
        default="centers",
        choices=["uniform", "urban", "rural", "centers"],
        help="[file mode] Server node distribution strategy",
    )
    # random mode
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=50,
        help="[random mode] Number of nodes in Barabasi-Albert graph",
    )
    # shared
    parser.add_argument("--vnf-types", type=int, default=10, help="Number of VNF types")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output topology JSON path (e.g. data/topologies/nsf_centers.json)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if args.mode == "file":
        if not args.node_csv or not args.edge_txt:
            parser.error("--mode file requires --node-csv and --edge-txt")
        print(f"Loading topology from {args.node_csv} + {args.edge_txt}")
        G_base = _file_to_graph_nsf(args.node_csv, args.edge_txt)
        if not nx.is_connected(G_base):
            G_base = G_base.subgraph(
                max(nx.connected_components(G_base), key=len)
            ).copy()
        G = nx.Graph(G_base)
        server_nodes = _select_server_nodes(G, args.server_dist, rng)
    else:
        print(f"Generating random Barabasi-Albert graph: {args.num_nodes} nodes")
        G = nx.barabasi_albert_graph(args.num_nodes, 3, seed=rng.randint(0, 9999))
        G = nx.Graph(G)
        n_fn = max(1, int(args.num_nodes * 0.3))
        nodes_by_degree = sorted(G.nodes(), key=lambda n: -G.degree(n))
        server_nodes = set(nodes_by_degree[:n_fn])

    _assign_network_attrs(G, server_nodes, rng)
    F_raw = _generate_vnfs(G, server_nodes, args.vnf_types, rng)
    data = _graph_to_json(G, F_raw)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    n_servers = sum(1 for u in G.nodes() if G.nodes[u].get("is_function_node", False))
    print(f"Topology saved: {args.out}")
    print(
        f"  Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}"
        f"  Servers={n_servers}  VNF types={args.vnf_types}"
    )


if __name__ == "__main__":
    main()
