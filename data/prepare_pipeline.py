"""
Prepare train/test pipeline from 36 original testcases.

Naming convention: data/testcase/{topology}_{allocation}_{difficulty}.json
  topology:   cogent, conus, nsf
  allocation: urban, center, rural, uniform
  difficulty: easy, normal, hard

Pipeline:
  TRAIN: 12 hard originals (paths written to train_manifest.txt, files unchanged)
  TEST:  generate N episodes per difficulty per group using
         topology V/E/F from hard original + request config from each difficulty original

Usage:
    python data/prepare_pipeline.py \
        --source-dir data/testcase \
        --output-dir data/all_tests \
        --num-per-difficulty 5 \
        --seed 42
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse existing generator logic
from data.generate_episodes import (
    _infer_request_config,
    _default_request_config,
    generate_requests,
)

TOPOLOGIES = ["cogent", "conus", "nsf"]
ALLOCATIONS = ["urban", "center", "rural", "uniform"]
DIFFICULTIES = ["easy", "normal", "hard"]


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _sim_duration_for_count(target_count: int, lambda_arr: float) -> int:
    """Compute sim_duration so that E[requests] = target_count.
    E[count] = lambda_arr * sim_duration  →  sim_duration = target / lambda_arr
    """
    if lambda_arr <= 0 or target_count <= 0:
        return 500
    return max(int(math.ceil(target_count / lambda_arr)), 10)


def _generate_with_target(
    cfg_d: dict,
    rng_seed: int,
    num_nodes: int,
    num_vnf_types: int,
    target_count: int,
    max_retries: int = 5,
) -> list:
    """Generate requests, retrying with adjusted sim_duration until count is close to target.
    Tolerance: within 20% of target_count, or best attempt after max_retries.
    """
    lam = cfg_d["lambda_arr"]
    base_dur = _sim_duration_for_count(target_count, lam)
    best_R = None
    best_delta = float("inf")

    for attempt in range(max_retries):
        # Slightly scale duration each retry to compensate Poisson variance
        scale = 1.0 + 0.05 * attempt  # 1.0, 1.05, 1.10, 1.15, 1.20
        sim_dur = max(10, int(math.ceil(base_dur * scale)))
        seed = (rng_seed + attempt * 7919) % (2**31)  # deterministic per attempt
        rng = np.random.default_rng(seed)
        R = generate_requests(cfg_d, rng, num_nodes, num_vnf_types, sim_dur)
        delta = abs(len(R) - target_count)
        if delta < best_delta:
            best_delta = delta
            best_R = R
        if target_count == 0 or delta / max(1, target_count) <= 0.20:
            break  # within 20% tolerance

    return best_R


def _derive_seed(
    base_seed: int, topology: str, allocation: str, difficulty: str, idx: int
) -> int:
    """Deterministic seed from (base, topology, allocation, difficulty, idx)."""
    key = f"{topology}_{allocation}_{difficulty}_{idx}"
    # Use a stable hash: sum of (char * position) mod large prime, XOR with base
    h = sum(ord(c) * (i + 1) for i, c in enumerate(key)) % (2**31 - 1)
    return (base_seed ^ h) % (2**31)


def _write_episode(path: str, V: dict, E: list, F: list, R: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"V": V, "E": E, "F": F, "R": R}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare train/test pipeline")
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing 36 original testcases",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for generated test episodes",
    )
    parser.add_argument(
        "--num-per-difficulty",
        type=int,
        default=5,
        help="Number of test episodes per difficulty per group (recommended 5-10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base seed for deterministic generation"
    )
    args = parser.parse_args()

    if not (5 <= args.num_per_difficulty <= 10):
        print(
            f"WARNING: --num-per-difficulty={args.num_per_difficulty} outside recommended 5-10"
        )

    # Validate all 36 source files exist
    missing = []
    for topo in TOPOLOGIES:
        for alloc in ALLOCATIONS:
            for diff in DIFFICULTIES:
                p = os.path.join(args.source_dir, f"{topo}_{alloc}_{diff}.json")
                if not os.path.exists(p):
                    missing.append(p)
    if missing:
        print("ERROR: missing source files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    train_paths = []
    total_generated = 0

    for topo in TOPOLOGIES:
        for alloc in ALLOCATIONS:
            # --- TRAIN: hard original (unchanged) ---
            hard_path = os.path.join(args.source_dir, f"{topo}_{alloc}_hard.json")
            train_paths.append(os.path.abspath(hard_path))

            # Load topology from hard original
            hard_data = _load_json(hard_path)
            V = hard_data["V"]
            E = hard_data["E"]
            F = hard_data.get("F", [])
            num_nodes = len(V)
            num_vnf_types = len(F)

            # --- TEST: infer request config from each difficulty original ---
            req_configs = {}
            target_counts = {}
            for diff in DIFFICULTIES:
                src_path = os.path.join(args.source_dir, f"{topo}_{alloc}_{diff}.json")
                src_data = _load_json(src_path)
                R_orig = src_data.get("R", [])
                if R_orig:
                    cfg_d = _infer_request_config(R_orig, num_nodes, num_vnf_types)
                else:
                    cfg_d = _default_request_config(num_nodes, num_vnf_types)
                req_configs[diff] = cfg_d
                target_counts[diff] = len(R_orig)

            # Generate episodes for each difficulty
            for diff in DIFFICULTIES:
                cfg_d = req_configs[diff]
                target = target_counts[diff]
                out_dir = os.path.join(args.output_dir, topo, alloc, diff)

                counts = []
                for idx in range(args.num_per_difficulty):
                    ep_seed = _derive_seed(args.seed, topo, alloc, diff, idx)
                    R = _generate_with_target(
                        cfg_d, ep_seed, num_nodes, num_vnf_types, target
                    )
                    out_path = os.path.join(out_dir, f"episode_{idx:04d}.json")
                    _write_episode(out_path, V, E, F, R)
                    total_generated += 1
                    counts.append(len(R))

                avg_count = sum(counts) / len(counts) if counts else 0
                print(
                    f"  {topo}/{alloc}/{diff}: {args.num_per_difficulty} eps  "
                    f"target={target}  avg_generated={avg_count:.0f}  "
                    f"λ={cfg_d['lambda_arr']:.3f}"
                )

    # Write train manifest
    manifest_path = os.path.join(args.output_dir, "train_manifest.txt")
    with open(manifest_path, "w") as f:
        for p in train_paths:
            f.write(p + "\n")

    print(f"\nDone.")
    print(f"  TRAIN : {len(train_paths)} hard originals (unchanged) → {manifest_path}")
    print(f"  TEST  : {total_generated} generated episodes → {args.output_dir}/")
    print(f"  Seed  : {args.seed} (deterministic — re-run with same args = same files)")


if __name__ == "__main__":
    main()
