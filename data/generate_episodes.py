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
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Request generation  (mirrors generate.py logic, importable)
# ---------------------------------------------------------------------------


def _default_request_config(num_nodes: int, num_vnf_types: int):
    """Sensible defaults when no existing R list to infer from."""
    return {
        "lambda_arr": 1.25,
        "mu_hold": 0.05,
        "deadline_min": 20,
        "deadline_max": 100,
        "vnf_len_min": 2,
        "vnf_len_max": min(8, num_vnf_types),
        "bw_req_min": 0.10,
        "bw_req_max": 1.00,
    }


def _infer_request_config(R_existing: list, num_nodes: int, num_vnf_types: int):
    """Infer request generation parameters from an existing R list."""
    if not R_existing:
        return _default_request_config(num_nodes, num_vnf_types)
    times = sorted(r["T"] for r in R_existing)
    inter = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    lam = 1.0 / np.mean(inter) if inter else 1.25
    d_maxs = [r["d_max"] for r in R_existing]
    bws = [r["b_r"] for r in R_existing]
    lens = [len(r["F_r"]) for r in R_existing]
    return {
        "lambda_arr": float(lam),
        "mu_hold": float(1.0 / np.mean(d_maxs)),
        "deadline_min": int(min(d_maxs)),
        "deadline_max": int(max(d_maxs)),
        "vnf_len_min": int(min(lens)),
        "vnf_len_max": int(max(lens)),
        "bw_req_min": float(min(bws)),
        "bw_req_max": float(max(bws)),
    }


def generate_requests(
    cfg: dict,
    rng: np.random.Generator,
    num_nodes: int,
    num_vnf_types: int,
    sim_duration: int,
) -> list:
    """Generate a Poisson-arrival request list.  Returns list of dicts {T, st_r, d_r, F_r, b_r, d_max}."""
    R = []
    t = 0.0
    lam = cfg["lambda_arr"]
    mu = cfg["mu_hold"]
    dl_min, dl_max = cfg["deadline_min"], cfg["deadline_max"]
    vl_min, vl_max = cfg["vnf_len_min"], cfg["vnf_len_max"]
    bw_min, bw_max = cfg["bw_req_min"], cfg["bw_req_max"]

    while t < sim_duration:
        t += rng.exponential(1.0 / lam)
        if t >= sim_duration:
            break
        hold = rng.exponential(1.0 / mu)
        d_max = int(np.clip(math.ceil(hold), dl_min, dl_max))
        src, dst = rng.choice(num_nodes, size=2, replace=False).tolist()
        F_k = int(rng.integers(vl_min, vl_max + 1))
        F_r = rng.choice(
            num_vnf_types, size=min(F_k, num_vnf_types), replace=False
        ).tolist()
        bw = float(rng.uniform(bw_min, bw_max))
        R.append(
            {
                "T": round(float(t), 10),
                "st_r": f"v{src}",
                "d_r": f"v{dst}",
                "F_r": [int(x) for x in F_r],
                "b_r": round(bw, 4),
                "d_max": d_max,
            }
        )
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
        print("Using default request config.")

    print(
        f"Topology  : {args.topology}  ({num_nodes} nodes, {len(E)} edges, {num_vnf_types} VNF types)"
    )
    print(
        f"Config    : λ={req_cfg['lambda_arr']:.3f}  μ={req_cfg['mu_hold']:.4f}"
        f"  T={args.sim_duration}  episodes={args.num_episodes}  seed={args.seed}"
    )
    print(f"Output    : {args.out_dir}/\n")

    os.makedirs(args.out_dir, exist_ok=True)
    master_rng = np.random.default_rng(args.seed)

    for ep in range(args.num_episodes):
        ep_seed = int(master_rng.integers(0, 2**31))
        ep_rng = np.random.default_rng(ep_seed)
        R = generate_requests(
            req_cfg, ep_rng, num_nodes, num_vnf_types, args.sim_duration
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
