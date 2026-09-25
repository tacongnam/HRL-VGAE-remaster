import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_csv(path: str):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {k: float(v) if k != "episode" else int(v) for k, v in row.items()}
            )
    return rows


def smooth(values, window: int = 20):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_fig(x, y_raw, ylabel, title, out_path, window, color="steelblue"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y_raw, alpha=0.25, color=color, linewidth=0.8, label="raw")
    if len(y_raw) >= window:
        ax.plot(
            x[window - 1 :],
            smooth(y_raw, window),
            color=color,
            linewidth=1.8,
            label=f"smooth(w={window})",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training(rows, out_dir: str, window: int = 20):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    episodes = [r["episode"] for r in rows]

    def get(key):
        return [r[key] for r in rows]

    save_fig(
        episodes,
        get("acceptance_ratio"),
        "Acceptance Ratio",
        "SFC Acceptance Ratio",
        out / "acceptance_ratio.png",
        window,
        "seagreen",
    )
    save_fig(
        episodes,
        get("total_revenue"),
        "Revenue",
        "Total Revenue per Episode",
        out / "revenue.png",
        window,
        "darkorange",
    )
    save_fig(
        episodes,
        get("total_deploy_cost"),
        "Deploy Cost",
        "Total Deployment Cost per Episode",
        out / "deploy_cost.png",
        window,
        "firebrick",
    )
    save_fig(
        episodes,
        get("hl_reward"),
        "HL Reward",
        "High-Level Agent Reward",
        out / "hl_reward.png",
        window,
        "steelblue",
    )
    save_fig(
        episodes,
        get("ll_reward"),
        "LL Reward",
        "Low-Level Agent Reward",
        out / "ll_reward.png",
        window,
        "mediumpurple",
    )
    save_fig(
        episodes,
        get("vgae_loss"),
        "VGAE Loss",
        "MN-VGAE Training Loss",
        out / "vgae_loss.png",
        window,
        "tomato",
    )
    save_fig(
        episodes,
        get("hl_loss"),
        "HL DQN Loss",
        "High-Level DQN Loss",
        out / "hl_loss.png",
        window,
        "royalblue",
    )
    save_fig(
        episodes,
        get("ll_loss"),
        "LL DQN Loss",
        "Low-Level DQN Loss",
        out / "ll_loss.png",
        window,
        "orchid",
    )
    save_fig(
        episodes,
        get("avg_link_util"),
        "Link Utilization",
        "Average Link Utilization",
        out / "link_util.png",
        window,
        "goldenrod",
    )
    save_fig(
        episodes,
        get("avg_cpu_util"),
        "CPU Utilization",
        "Average CPU Utilization",
        out / "cpu_util.png",
        window,
        "cadetblue",
    )
    save_fig(
        episodes,
        get("epsilon"),
        "Epsilon",
        "ε-Greedy Exploration Decay",
        out / "epsilon.png",
        window,
        "slategray",
    )
    save_fig(
        episodes,
        get("w_accept"),
        "Pareto Weight (accept)",
        "Adaptive Pareto Weight over Training",
        out / "pareto_weight.png",
        window,
        "darkviolet",
    )
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    pairs = [
        ("acceptance_ratio", "Acceptance Ratio", "seagreen"),
        ("total_revenue", "Revenue", "darkorange"),
        ("total_deploy_cost", "Deploy Cost", "firebrick"),
        ("hl_reward", "HL Reward", "steelblue"),
        ("avg_link_util", "Link Util", "goldenrod"),
        ("avg_cpu_util", "CPU Util", "cadetblue"),
        ("vgae_loss", "VGAE Loss", "tomato"),
        ("w_accept", "Pareto W", "darkviolet"),
    ]
    for ax, (key, label, color) in zip(axes.flat, pairs):
        y_raw = get(key)
        ax.plot(episodes, y_raw, alpha=0.2, color=color, linewidth=0.7)
        if len(y_raw) >= window:
            ax.plot(
                episodes[window - 1 :],
                smooth(y_raw, window),
                color=color,
                linewidth=1.6,
            )
        ax.set_title(label)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
    fig.suptitle("VGAE-HRL-QL Multi-Objective Training Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=150)
    plt.close(fig)
    print(f"Training plots saved to: {out.resolve()}")


def plot_pareto_front(pareto_json: str, out_dir: str, hv_json: str = ""):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(pareto_json) as f:
        scan = json.load(f)
    acc = np.array([r["acc_ratio"] for r in scan])
    cost = np.array([r["deploy_cost"] for r in scan])
    w_accept = np.array([r["w_accept"] for r in scan])

    hv_data = None
    if hv_json and os.path.exists(hv_json):
        with open(hv_json) as f:
            hv_data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        cost, acc, c=w_accept, cmap="viridis", s=80, edgecolor="black", zorder=3
    )
    fig.colorbar(sc, ax=ax, label="w_accept")

    if hv_data and hv_data.get("pareto_front"):
        front = hv_data["pareto_front"]
        fx = np.array([p["deploy_cost"] for p in front])
        fy = np.array([p["acc_ratio"] for p in front])
        order = np.argsort(fx)
        ax.plot(
            fx[order],
            fy[order],
            color="crimson",
            linewidth=2,
            linestyle="--",
            zorder=2,
            label="Pareto Front",
        )

        hv = hv_data.get("hypervolume", None)
        rp = hv_data.get("reference_point", {})
        ref_acc = rp.get("acc_ratio", None)
        ref_cost = -rp.get("neg_deploy_cost", 0.0)
        parts = []
        if hv is not None:
            parts.append(f"HV = {hv:.4f}")
        if ref_acc is not None:
            parts.append(f"ref = ({ref_acc:.3f}, {ref_cost:.1f})")
        if parts:
            ax.text(
                0.02,
                0.02,
                "\n".join(parts),
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
    else:
        # fallback: compute front from scan data (no HV available)
        order = np.argsort(cost)
        acc_s, cost_s = acc[order], cost[order]
        is_front = np.zeros(len(acc_s), dtype=bool)
        best_acc = -np.inf
        for i in range(len(acc_s) - 1, -1, -1):
            if acc_s[i] > best_acc:
                is_front[i] = True
                best_acc = acc_s[i]
        ax.plot(
            cost_s[is_front],
            acc_s[is_front],
            color="crimson",
            linewidth=2,
            linestyle="--",
            zorder=2,
            label="Pareto Front",
        )

    ax.set_xlabel("Total Deployment Cost")
    ax.set_ylabel("Acceptance Ratio")
    ax.set_title("Pareto Front: Acceptance Ratio vs Deployment Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out / "pareto_front.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Pareto front plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="training_log.csv")
    parser.add_argument("--pareto-json", type=str, default="")
    parser.add_argument("--hv-json", type=str, default="")
    parser.add_argument("--out", type=str, default="plots")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    if os.path.exists(args.csv):
        rows = load_csv(args.csv)
        plot_training(rows, args.out, args.window)
    if args.pareto_json and os.path.exists(args.pareto_json):
        plot_pareto_front(args.pareto_json, args.out, hv_json=args.hv_json)
