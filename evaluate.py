import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import numpy as np


def _load_scan(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    valid = [
        r
        for r in data
        if isinstance(r.get("acc_ratio"), (int, float))
        and isinstance(r.get("deploy_cost"), (int, float))
        and np.isfinite(r["acc_ratio"])
        and np.isfinite(r["deploy_cost"])
    ]
    if not valid:
        raise ValueError(f"No valid entries in {path}")
    return valid


def _to_hv_space(results: list):
    return np.array(
        [[r["acc_ratio"], -r["deploy_cost"]] for r in results], dtype=np.float64
    )


def _ref_point(pts: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    return np.array([pts[:, 0].min() - eps, pts[:, 1].min() - eps])


def _compute_hv(scan_json: str, out: str, eps: float):
    from utils.pareto import pareto_front, compute_hypervolume

    results = _load_scan(scan_json)
    pts = _to_hv_space(results)
    pts = np.unique(pts, axis=0)
    front_pts = pareto_front([tuple(p) for p in pts])
    if not front_pts:
        print("WARNING: empty Pareto front")
        hv = 0.0
        ref = _ref_point(pts, eps).tolist()
    else:
        front_arr = np.array(front_pts, dtype=np.float64)
        ref = _ref_point(pts, eps)
        hv = compute_hypervolume(front_arr, ref)
        ref = ref.tolist()
    output = {
        "hypervolume": float(hv),
        "reference_point": {"acc_ratio": ref[0], "neg_deploy_cost": ref[1]},
        "pareto_front": [
            {
                "acc_ratio": float(p[0]),
                "neg_deploy_cost": float(p[1]),
                "deploy_cost": float(-p[1]),
            }
            for p in (front_pts if front_pts else [])
        ],
        "all_points": [
            {
                "acc_ratio": float(r["acc_ratio"]),
                "deploy_cost": float(r["deploy_cost"]),
                "w_accept": float(r.get("w_accept", 0.0)),
                "w_cost": float(r.get("w_cost", 0.0)),
            }
            for r in results
        ],
    }
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"HV={hv:.6f}  ref={ref}  front_size={len(front_pts)}  saved→{out}")


def _migrate_hl_state_dict(sd: dict, n_obj: int = 4) -> dict:
    """Migrate legacy HLSharedQScorer checkpoint to current architecture.

    Handles legacy formats:
      1. net.* keys         → rename to trunk.*
      2. trunk.4.weight shape [N, H]:
            N == n_obj  → split row-i into q_heads.i  (full multi-obj checkpoint)
            N < n_obj   → copy available rows; zero-init remaining heads
                           (single/partial-obj checkpoint — heads beyond N are
                            zero-initialised, not random; trained signal is preserved)

    Current format (trunk.0/2 + q_heads.0-3, no trunk.4) passes through unchanged.
    Raises ValueError if shapes are inconsistent in an unexpected way.
    """
    import torch

    # Step 1: rename net.* → trunk.*
    if any(k.startswith("net.") for k in sd):
        sd = {
            ("trunk." + k[len("net.") :] if k.startswith("net.") else k): v
            for k, v in sd.items()
        }

    # Step 2: split old combined final linear into per-objective heads
    if "trunk.4.weight" in sd:
        w = sd.pop("trunk.4.weight")  # shape [N, H]
        b = sd.pop("trunk.4.bias")  # shape [N]
        n_ckpt, H = int(w.shape[0]), int(w.shape[1])
        if n_ckpt > n_obj:
            raise ValueError(
                f"Checkpoint trunk.4.weight has {n_ckpt} rows but model has {n_obj} heads."
            )
        for i in range(n_obj):
            if i < n_ckpt:
                sd[f"q_heads.{i}.weight"] = w[i : i + 1].clone()
                sd[f"q_heads.{i}.bias"] = b[i : i + 1].clone()
            else:
                # Head was never trained — zero-init (neutral, not random)
                sd[f"q_heads.{i}.weight"] = torch.zeros(1, H)
                sd[f"q_heads.{i}.bias"] = torch.zeros(1)

    return sd


def _migrate_ll_state_dict(sd: dict, n_obj: int = 4) -> dict:
    """Migrate legacy LLNodeScorer checkpoint to current architecture.

    LLNodeScorer uses 'shared' backbone — no rename needed.
    If the checkpoint has fewer than n_obj q_heads, zero-init the missing ones.
    Existing heads are preserved exactly. Current-format passes through unchanged.
    """
    import torch

    existing = sorted(
        int(k.split(".")[1])
        for k in sd
        if k.startswith("q_heads.") and k.endswith(".weight")
    )
    if not existing:
        return sd
    n_ckpt = max(existing) + 1
    if n_ckpt >= n_obj:
        return sd
    H = int(sd[f"q_heads.{existing[-1]}.weight"].shape[1])
    sd = dict(sd)
    for i in range(n_ckpt, n_obj):
        sd[f"q_heads.{i}.weight"] = torch.zeros(1, H)
        sd[f"q_heads.{i}.bias"] = torch.zeros(1)
    return sd


def _load_checkpoint(checkpoint: str, cfg, device):
    import torch
    from models.vgae import MNVGAE
    from agents.hl_agent import HLAgent
    from agents.ll_agent import LLAgent
    from utils.pareto import ParetoScalarizer

    ckpt = torch.load(checkpoint, map_location=device)
    vgae = MNVGAE(cfg).to(device)
    vgae.load_state_dict(ckpt["vgae"])
    vgae.eval()

    hl_agent = HLAgent(cfg, device)
    hl_q_sd = _migrate_hl_state_dict(ckpt["hl_q"])
    hl_agent.q_net.load_state_dict(hl_q_sd)
    hl_agent.target_net.load_state_dict(hl_q_sd)
    hl_agent.epsilon = 0.0

    ll_agent = LLAgent(cfg, device)
    ll_q_sd = _migrate_ll_state_dict(ckpt["ll_q"])
    ll_agent.q_net.load_state_dict(ll_q_sd)
    ll_agent.target_net.load_state_dict(ll_q_sd)
    ll_agent.epsilon = 0.0

    scalarizer = ParetoScalarizer(cfg)
    scalarizer.w_accept = float(ckpt.get("w_accept", 0.5))
    scalarizer.w_cost = float(ckpt.get("w_cost", 0.5))

    return vgae, hl_agent, ll_agent, scalarizer


def _discover_recursive(root: str):
    """Recursively find all episode_*.json files under root.

    Returns list of (path, meta) where meta = {topology, allocation, difficulty, test_index}.
    Expected directory structure: root/{topology}/{allocation}/{difficulty}/episode_NNNN.json
    Falls back to flat discover_episodes(root) for backward compatibility.
    """
    import glob as _glob

    structured = []
    for path in sorted(_glob.glob(os.path.join(root, "*", "*", "*", "episode_*.json"))):
        parts = path.replace("\\", "/").split("/")
        # Find the segment that matches known difficulties
        try:
            diff_idx = next(
                i for i, p in enumerate(parts) if p in ("easy", "normal", "hard")
            )
            difficulty = parts[diff_idx]
            allocation = parts[diff_idx - 1]
            topology = parts[diff_idx - 2]
            fname = os.path.splitext(os.path.basename(path))[0]
            test_index = (
                int(fname.split("_")[-1]) if fname.split("_")[-1].isdigit() else 0
            )
            structured.append(
                (
                    path,
                    {
                        "topology": topology,
                        "allocation": allocation,
                        "difficulty": difficulty,
                        "test_index": test_index,
                    },
                )
            )
        except StopIteration:
            structured.append(
                (
                    path,
                    {
                        "topology": "unknown",
                        "allocation": "unknown",
                        "difficulty": "unknown",
                        "test_index": 0,
                    },
                )
            )

    if structured:
        return structured

    # Fallback: flat directory
    from data.loader import discover_episodes

    flat = discover_episodes(root)
    return [
        (
            p,
            {
                "topology": "unknown",
                "allocation": "unknown",
                "difficulty": "unknown",
                "test_index": 0,
            },
        )
        for p in flat
    ]


def _scan_pareto(args):
    import torch
    from config import Config
    from env.nfv_env import NFVEnvironment
    from data.loader import parse_episode
    from train import run_episode

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgae, hl_agent, ll_agent, scalarizer = _load_checkpoint(
        args.checkpoint, cfg, device
    )

    if args.recursive:
        path_meta_list = _discover_recursive(args.data_dir)
    else:
        from data.loader import discover_episodes

        flat = discover_episodes(args.data_dir)
        path_meta_list = [
            (
                p,
                {
                    "topology": "unknown",
                    "allocation": "unknown",
                    "difficulty": "unknown",
                    "test_index": 0,
                },
            )
            for p in flat
        ]

    if not path_meta_list:
        raise FileNotFoundError(f"No episodes found in {args.data_dir}")

    n_pts = args.pareto_points
    weights = [k / max(1, n_pts - 1) for k in range(n_pts)]
    results = []

    with torch.no_grad():
        for w_accept in weights:
            w_cost = 1.0 - w_accept
            for path, meta in path_meta_list:
                G, reqs, _, topo_id = parse_episode(path)
                env = NFVEnvironment(cfg)
                env.reset(G, reqs, topology_id=topo_id)
                stats = run_episode(
                    env,
                    vgae,
                    hl_agent,
                    ll_agent,
                    vgae_optimizer=None,
                    scalarizer=scalarizer,
                    cfg=cfg,
                    device=device,
                    train=False,
                    fixed_weight=(w_accept, w_cost),
                )
                entry = {
                    "w_accept": w_accept,
                    "w_cost": w_cost,
                    "acc_ratio": float(stats["acceptance_ratio"]),
                    "deploy_cost": float(stats["total_deploy_cost"]),
                    "topology": meta["topology"],
                    "allocation": meta["allocation"],
                    "difficulty": meta["difficulty"],
                    "test_index": meta["test_index"],
                    "path": path,
                }
                results.append(entry)

            # Per-weight summary
            w_entries = [e for e in results if abs(e["w_accept"] - w_accept) < 1e-9]
            mean_acc = float(np.mean([e["acc_ratio"] for e in w_entries]))
            mean_cost = float(np.mean([e["deploy_cost"] for e in w_entries]))
            print(
                f"  w_accept={w_accept:.2f}  acc={mean_acc:.4f}  cost={mean_cost:.2f}"
                f"  (n={len(w_entries)})"
            )

    with open(args.pareto_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pareto scan saved → {args.pareto_out}")

    if args.compute_hv:
        # _compute_hv expects {acc_ratio, deploy_cost} — compatible subset
        _compute_hv(args.pareto_out, args.hv_out, args.hv_eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-json", type=str, default="pareto_front.json")
    parser.add_argument("--out", type=str, default="hv_result.json")
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--pareto-scan", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="vgae_hrl_ql_checkpoint.pt")
    parser.add_argument("--data-dir", type=str, default="data/dataset_01/test")
    parser.add_argument("--pareto-points", type=int, default=11)
    parser.add_argument("--pareto-out", type=str, default="pareto_front.json")
    parser.add_argument("--compute-hv", action="store_true")
    parser.add_argument("--hv-out", type=str, default="hv_result.json")
    parser.add_argument("--hv-eps", type=float, default=1e-2)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover episodes under --data-dir with "
        "{topology}/{allocation}/{difficulty}/ structure (for data/all_tests/).",
    )
    args = parser.parse_args()

    if args.pareto_scan:
        _scan_pareto(args)
    else:
        _compute_hv(args.scan_json, args.out, args.eps)


if __name__ == "__main__":
    main()
