import sys
import os
import json
import math
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RequestConfig:
    lambda_arr: float = 1.25
    mu_hold: float = 0.05
    deadline_min: int = 20
    deadline_max: int = 100
    vnf_len_min: int = 2
    vnf_len_max: int = 8
    bw_req_min: float = 10.0
    bw_req_max: float = 100.0
    sim_duration: int = 1000


def _infer_config_from_train(R: List[Dict], V: Dict, sim_duration: int) -> RequestConfig:
    if not R:
        return RequestConfig(sim_duration=sim_duration)
    times = sorted(r["T"] for r in R)
    inter_arrivals = [times[i+1] - times[i] for i in range(len(times)-1)]
    lambda_arr = 1.0 / np.mean(inter_arrivals) if inter_arrivals else 1.25
    d_maxs = [r["d_max"] for r in R]
    deadline_min = int(min(d_maxs))
    deadline_max = int(max(d_maxs))
    mu_hold = 1.0 / np.mean(d_maxs)
    bw_reqs = [r["b_r"] * 100.0 for r in R]
    bw_req_min = float(min(bw_reqs))
    bw_req_max = float(max(bw_reqs))
    chain_lens = [len(r["F_r"]) for r in R]
    vnf_len_min = int(min(chain_lens))
    vnf_len_max = int(max(chain_lens))
    return RequestConfig(
        lambda_arr=lambda_arr,
        mu_hold=mu_hold,
        deadline_min=deadline_min,
        deadline_max=deadline_max,
        vnf_len_min=vnf_len_min,
        vnf_len_max=vnf_len_max,
        bw_req_min=bw_req_min,
        bw_req_max=bw_req_max,
        sim_duration=sim_duration,
    )


def generate_requests(cfg: RequestConfig, rng: np.random.Generator, num_nodes: int, num_distinct_vnfs: int) -> List[Dict]:
    R = []
    current_time = 0.0
    while current_time < cfg.sim_duration:
        inter_arrival = rng.exponential(scale=1.0 / cfg.lambda_arr)
        current_time += inter_arrival
        if current_time >= cfg.sim_duration:
            break
        holding_time = rng.exponential(scale=1.0 / cfg.mu_hold)
        raw_deadline = int(math.ceil(holding_time))
        d_max = int(np.clip(raw_deadline, cfg.deadline_min, cfg.deadline_max))
        src, dst = rng.choice(num_nodes, size=2, replace=False).tolist()
        F_k = int(rng.integers(cfg.vnf_len_min, cfg.vnf_len_max + 1))
        F_r = rng.choice(num_distinct_vnfs, size=min(F_k, num_distinct_vnfs), replace=False).tolist()
        bw_req = float(rng.uniform(cfg.bw_req_min, cfg.bw_req_max))
        R.append({
            "T": round(float(current_time), 14),
            "st_r": f"v{src}",
            "d_r": f"v{dst}",
            "F_r": F_r,
            "b_r": round(bw_req / 100.0, 2),
            "d_max": d_max
        })
    return R


def main():
    parser = argparse.ArgumentParser(description="Generate test episodes from a train JSON file")
    parser.add_argument("--train-file", type=str, required=True, help="Path to train JSON file (e.g. abc.json)")
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data/test", help="Output dir (default: same folder as train file)")
    parser.add_argument("--sim-duration", type=int, default=0, help="Override sim duration (0 = infer from train file)")
    args = parser.parse_args()

    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Train file not found: {args.train_file}")

    with open(args.train_file, "r") as f:
        train_data = json.load(f)

    V = train_data["V"]
    E = train_data["E"]
    F = train_data["F"]
    R_train = train_data["R"]
    num_nodes = len(V)
    num_distinct_vnfs = len(F)

    sim_duration = args.sim_duration
    if sim_duration == 0:
        sim_duration = int(math.ceil(max(r["T"] for r in R_train))) + 1 if R_train else 1000

    cfg = _infer_config_from_train(R_train, V, sim_duration)

    out_dir = args.out_dir if args.out_dir else os.path.join(os.path.dirname(os.path.abspath(args.train_file)), "test")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Train file   : {args.train_file}")
    print(f"Topology     : {num_nodes} nodes, {len(E)} edges, {num_distinct_vnfs} VNF types")
    print(f"Inferred cfg : λ={cfg.lambda_arr:.4f}, μ={cfg.mu_hold:.4f}, T={sim_duration}")
    print(f"               deadline=[{cfg.deadline_min},{cfg.deadline_max}], bw=[{cfg.bw_req_min:.1f},{cfg.bw_req_max:.1f}]")
    print(f"               chain_len=[{cfg.vnf_len_min},{cfg.vnf_len_max}]")
    print(f"Generating   : {args.test_episodes} test episodes → {out_dir}/\n")

    master_rng = np.random.default_rng(args.seed)
    for ep in range(args.test_episodes):
        ep_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        R = generate_requests(cfg, ep_rng, num_nodes, num_distinct_vnfs)
        data = {"V": V, "E": E, "F": F, "R": R}
        fpath = os.path.join(out_dir, f"episode_{ep:04d}.json")
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Ep {ep+1:4d}/{args.test_episodes} | Requests: {len(R):5d} | {os.path.basename(fpath)}")

    print(f"\nDone. Saved {args.test_episodes} test episodes to: {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()