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
    hl_agent.q_net.load_state_dict(ckpt["hl_q"])
    hl_agent.target_net.load_state_dict(ckpt["hl_q"])
    hl_agent.epsilon = 0.0

    ll_agent = LLAgent(cfg, device)
    ll_agent.q_net.load_state_dict(ckpt["ll_q"])
    ll_agent.target_net.load_state_dict(ckpt["ll_q"])
    ll_agent.epsilon = 0.0

    scalarizer = ParetoScalarizer(cfg)
    scalarizer.w_accept = float(ckpt.get("w_accept", 0.5))
    scalarizer.w_cost = float(ckpt.get("w_cost", 0.5))

    return vgae, hl_agent, ll_agent, scalarizer


def _scan_pareto(args):
    import torch
    from config import Config
    from env.nfv_env import NFVEnvironment
    from data.loader import discover_episodes, parse_episode
    from train import run_episode

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgae, hl_agent, ll_agent, scalarizer = _load_checkpoint(
        args.checkpoint, cfg, device
    )

    test_paths = discover_episodes(args.data_dir)
    if not test_paths:
        raise FileNotFoundError(f"No episodes found in {args.data_dir}")

    n_pts = args.pareto_points
    weights = [k / max(1, n_pts - 1) for k in range(n_pts)]
    results = []

    for w_accept in weights:
        w_cost = 1.0 - w_accept
        acc_ratios, deploy_costs = [], []
        for path in test_paths:
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
            acc_ratios.append(stats["acceptance_ratio"])
            deploy_costs.append(stats["total_deploy_cost"])
        entry = {
            "w_accept": w_accept,
            "w_cost": w_cost,
            "acc_ratio": float(np.mean(acc_ratios)),
            "deploy_cost": float(np.mean(deploy_costs)),
        }
        results.append(entry)
        print(
            f"  w_accept={w_accept:.2f}  acc={entry['acc_ratio']:.4f}  cost={entry['deploy_cost']:.2f}"
        )

    with open(args.pareto_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pareto scan saved → {args.pareto_out}")

    if args.compute_hv:
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
    args = parser.parse_args()

    if args.pareto_scan:
        _scan_pareto(args)
    else:
        _compute_hv(args.scan_json, args.out, args.eps)


if __name__ == "__main__":
    main()
