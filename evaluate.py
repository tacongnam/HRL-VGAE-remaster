import argparse
import json
import os
import glob
import numpy as np
import torch

from config import Config
from env import NFVEnvironment
from models import MNVGAE
from agents import HLAgent, LLAgent
from utils import (
    ParetoScalarizer,
    pareto_front,
    compute_hypervolume,
    coverage_metric,
    spacing_metric,
    spread_metric,
)
from data import parse_episode, discover_episodes


def _compute_hv(scan_json: str, out: str, eps: float, baseline_json: str = ""):
    with open(scan_json) as f:
        data = json.load(f)
    pts = np.unique(
        np.array([[r["acc_ratio"], -r["deploy_cost"]] for r in data], dtype=np.float64),
        axis=0,
    )
    front_pts = pareto_front([tuple(p) for p in pts])
    ref = np.array([pts[:, 0].min() - eps, pts[:, 1].min() - eps])
    hv = (
        compute_hypervolume(np.array(front_pts, dtype=np.float64), ref)
        if front_pts
        else 0.0
    )

    front_arr = np.array(front_pts, dtype=np.float64) if front_pts else np.empty((0, 2))
    spacing = spacing_metric(front_arr, normalize=True)
    spread = spread_metric(front_arr, normalize=True)

    metrics_dict = {
        "hypervolume": float(hv),
        "spacing": float(spacing),
        "delta": float(spread),
        "spread": float(spread),
        "reference_point": {
            "acc_ratio": float(ref[0]),
            "neg_deploy_cost": float(ref[1]),
        },
        "pareto_front": [
            {"acc_ratio": float(p[0]), "deploy_cost": float(-p[1])} for p in front_pts
        ],
        "all_points": data,
    }

    if baseline_json and os.path.exists(baseline_json):
        with open(baseline_json) as f:
            base_data = json.load(f)
        base_pts = np.unique(
            np.array(
                [[r["acc_ratio"], -r["deploy_cost"]] for r in base_data],
                dtype=np.float64,
            ),
            axis=0,
        )
        base_front = np.array(
            pareto_front([tuple(p) for p in base_pts]), dtype=np.float64
        )
        c_ab = coverage_metric(front_arr, base_front, maximize=True)
        c_ba = coverage_metric(base_front, front_arr, maximize=True)
        metrics_dict["coverage_ab"] = float(c_ab)
        metrics_dict["coverage_ba"] = float(c_ba)
        metrics_dict["coverage"] = float(c_ab)

    with open(out, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(
        f"HV={hv:.6f}  Spacing={spacing:.4f}  Delta={spread:.4f}  "
        f"ref={ref.tolist()}  front_size={len(front_pts)}  saved→{out}"
    )


def _load_checkpoint(checkpoint: str, cfg: Config, device: torch.device):
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
    from train import run_episode

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgae, hl_agent, ll_agent, scalarizer = _load_checkpoint(
        args.checkpoint, cfg, device
    )

    if args.single_episode:
        paths = [args.single_episode]
    elif args.recursive:
        paths = sorted(
            glob.glob(os.path.join(args.data_dir, "*", "*", "*", "episode_*.json"))
        )
    else:
        paths = discover_episodes(args.data_dir)

    weights = [k / max(1, args.pareto_points - 1) for k in range(args.pareto_points)]
    results = []

    with torch.no_grad():
        for w_acc in weights:
            w_cost = 1.0 - w_acc
            for path in paths:
                G, reqs, _, topo_id = parse_episode(path)
                env = NFVEnvironment(cfg)
                env.reset(G, reqs, topology_id=topo_id)
                stats = run_episode(
                    env,
                    vgae,
                    hl_agent,
                    ll_agent,
                    None,
                    scalarizer,
                    cfg,
                    device,
                    train=False,
                    fixed_weight=(w_acc, w_cost),
                )
                results.append(
                    {
                        "w_accept": w_acc,
                        "w_cost": w_cost,
                        "acc_ratio": float(stats["acceptance_ratio"]),
                        "deploy_cost": float(stats["total_deploy_cost"]),
                        "path": path,
                    }
                )

            w_entries = [e for e in results if abs(e["w_accept"] - w_acc) < 1e-9]
            print(
                f"  w_accept={w_acc:.2f}  acc={np.mean([e['acc_ratio'] for e in w_entries]):.4f}  cost={np.mean([e['deploy_cost'] for e in w_entries]):.2f}"
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
    parser.add_argument("--single-episode", type=str, default="")
    parser.add_argument("--recursive", action="store_true")
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
