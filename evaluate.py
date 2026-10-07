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


def _ensure_parent(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _resolve_episode_paths(args) -> list:
    if args.single_episode:
        return [args.single_episode]
    if args.recursive:
        return sorted(
            glob.glob(os.path.join(args.data_dir, "*", "*", "*", "episode_*.json"))
        )
    return discover_episodes(args.data_dir)


def _compute_hv(scan_json: str, out: str, eps: float, baseline_json: str = ""):
    with open(scan_json) as f:
        data = json.load(f)

    raw = np.array(
        [[r["acc_ratio"], -r["deploy_cost"]] for r in data],
        dtype=np.float64,
    )
    if raw.size == 0:
        pts = np.empty((0, 2), dtype=np.float64)
    else:
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        pts = np.unique(raw, axis=0)

    front_pts = pareto_front([tuple(p) for p in pts]) if len(pts) else []

    if len(pts) == 0:
        ref = np.array([-eps, -eps], dtype=np.float64)
        hv = 0.0
    else:
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
        base_raw = np.array(
            [[r["acc_ratio"], -r["deploy_cost"]] for r in base_data],
            dtype=np.float64,
        )
        if base_raw.size == 0:
            base_pts = np.empty((0, 2), dtype=np.float64)
        else:
            if base_raw.ndim == 1:
                base_raw = base_raw.reshape(1, -1)
            base_pts = np.unique(base_raw, axis=0)
        base_front_list = (
            pareto_front([tuple(p) for p in base_pts]) if len(base_pts) else []
        )
        base_front = (
            np.array(base_front_list, dtype=np.float64)
            if base_front_list
            else np.empty((0, 2), dtype=np.float64)
        )
        c_ab = coverage_metric(front_arr, base_front, maximize=True)
        c_ba = coverage_metric(base_front, front_arr, maximize=True)
        metrics_dict["coverage_ab"] = float(c_ab)
        metrics_dict["coverage_ba"] = float(c_ba)
        metrics_dict["coverage"] = float(c_ab)

    _ensure_parent(out)
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


def _eval_ar_cost(args):
    """
    Evaluate AR + deploy_cost only, using weights loaded from the checkpoint.
    No Pareto weight grid — one run per episode with (w_accept, w_cost) from ckpt.
    """
    from train import run_episode

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgae, hl_agent, ll_agent, scalarizer = _load_checkpoint(
        args.checkpoint, cfg, device
    )

    w_acc = float(scalarizer.w_accept)
    w_cost = float(scalarizer.w_cost)
    print(
        f"[ar-cost] using checkpoint weights: w_accept={w_acc:.4f}  w_cost={w_cost:.4f}",
        flush=True,
    )

    paths = _resolve_episode_paths(args)
    if not paths:
        print(f"[ar-cost] WARNING: no episodes found under data-dir={args.data_dir}")

    results = []
    with torch.no_grad():
        for path in paths:
            G, reqs, _, topo_id = parse_episode(path)
            env = NFVEnvironment(cfg)
            env.reset(G, reqs, topology_id=topo_id)
            # fixed_weight=None + train=False → run_episode uses scalarizer.w_*
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
                fixed_weight=None,
            )
            entry = {
                "w_accept": w_acc,
                "w_cost": w_cost,
                "acc_ratio": float(stats["acceptance_ratio"]),
                "deploy_cost": float(stats["total_deploy_cost"]),
                "accepted": int(stats.get("accepted", 0)),
                "rejected": int(stats.get("rejected", 0)),
                "path": path,
            }
            results.append(entry)
            print(
                f"  {os.path.basename(path)}  AR={entry['acc_ratio']:.4f}  "
                f"cost={entry['deploy_cost']:.2f}  "
                f"acc={entry['accepted']} rej={entry['rejected']}",
                flush=True,
            )

    out_path = args.ar_cost_out or args.pareto_out
    _ensure_parent(out_path)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    if results:
        mean_ar = float(np.mean([e["acc_ratio"] for e in results]))
        mean_cost = float(np.mean([e["deploy_cost"] for e in results]))
        print(
            f"[ar-cost] n={len(results)}  mean_AR={mean_ar:.4f}  "
            f"mean_cost={mean_cost:.2f}  saved→{out_path}",
            flush=True,
        )
    else:
        print(f"[ar-cost] n=0  saved→{out_path}", flush=True)


def _scan_pareto(args):
    from train import run_episode

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgae, hl_agent, ll_agent, scalarizer = _load_checkpoint(
        args.checkpoint, cfg, device
    )

    paths = _resolve_episode_paths(args)
    if not paths:
        print(f"[pareto] WARNING: no episodes found under data-dir={args.data_dir}")

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
                f"  w_accept={w_acc:.2f}  "
                f"acc={np.mean([e['acc_ratio'] for e in w_entries]):.4f}  "
                f"cost={np.mean([e['deploy_cost'] for e in w_entries]):.2f}"
            )

    _ensure_parent(args.pareto_out)
    with open(args.pareto_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pareto scan saved → {args.pareto_out}  ({len(results)} entries)")
    if args.compute_hv:
        _compute_hv(args.pareto_out, args.hv_out, args.hv_eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-json", type=str, default="pareto_front.json")
    parser.add_argument("--out", type=str, default="hv_result.json")
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument(
        "--pareto-scan",
        action="store_true",
        help="Full Pareto weight-grid scan",
    )
    parser.add_argument(
        "--ar-cost-only",
        action="store_true",
        help="Evaluate AR + deploy_cost only, using weights from checkpoint (no Pareto grid)",
    )
    parser.add_argument("--checkpoint", type=str, default="vgae_hrl_ql_checkpoint.pt")
    parser.add_argument("--data-dir", type=str, default="data/dataset_01/test")
    parser.add_argument("--single-episode", type=str, default="")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--pareto-points", type=int, default=11)
    parser.add_argument("--pareto-out", type=str, default="pareto_front.json")
    parser.add_argument(
        "--ar-cost-out",
        type=str,
        default="",
        help="Output path for --ar-cost-only (default: same as --pareto-out)",
    )
    parser.add_argument("--compute-hv", action="store_true")
    parser.add_argument("--hv-out", type=str, default="hv_result.json")
    parser.add_argument("--hv-eps", type=float, default=1e-2)
    args = parser.parse_args()

    if args.ar_cost_only:
        _eval_ar_cost(args)
    elif args.pareto_scan:
        _scan_pareto(args)
    else:
        _compute_hv(args.scan_json, args.out, args.eps)


if __name__ == "__main__":
    main()