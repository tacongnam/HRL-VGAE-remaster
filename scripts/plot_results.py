"""
Plot training curves, AR/cost screening, Pareto front, and comparative analyses for HRL-VGAE.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_csv(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k == "episode":
                    parsed[k] = int(v)
                else:
                    try:
                        parsed[k] = float(v)
                    except (TypeError, ValueError):
                        parsed[k] = v
            rows.append(parsed)
    return rows


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def smooth(values, window: int = 20):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _safe_get(rows: List[dict], key: str) -> Optional[List[float]]:
    if not rows or key not in rows[0]:
        return None
    out = []
    for r in rows:
        v = r.get(key)
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return None
    return out


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


def plot_training(rows: List[dict], out_dir: str, window: int = 20):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    episodes = [r["episode"] for r in rows]

    specs = [
        ("acceptance_ratio", "Acceptance Ratio", "SFC Acceptance Ratio", "seagreen"),
        ("total_revenue", "Revenue", "Total Revenue per Episode", "darkorange"),
        ("total_deploy_cost", "Deploy Cost", "Total Deployment Cost per Episode", "firebrick"),
        ("hl_reward", "HL Reward", "High-Level Agent Reward", "steelblue"),
        ("ll_reward", "LL Reward", "Low-Level Agent Reward", "mediumpurple"),
        ("vgae_loss", "VGAE Loss", "MN-VGAE Training Loss", "tomato"),
        ("hl_loss", "HL DQN Loss", "High-Level DQN Loss", "royalblue"),
        ("ll_loss", "LL DQN Loss", "Low-Level DQN Loss", "orchid"),
        ("avg_link_util", "Link Utilization", "Average Link Utilization", "goldenrod"),
        ("avg_cpu_util", "CPU Utilization", "Average CPU Utilization", "cadetblue"),
        ("epsilon", "Epsilon", "ε-Greedy Exploration Decay", "slategray"),
        ("w_accept", "Pareto Weight (accept)", "Adaptive Pareto Weight over Training", "darkviolet"),
    ]

    plotted = []
    for key, ylabel, title, color in specs:
        y = _safe_get(rows, key)
        if y is None:
            continue
        fname = key if key != "w_accept" else "pareto_weight"
        save_fig(episodes, y, ylabel, title, out / f"{fname}.png", window, color)
        plotted.append(key)

    overview_keys = [
        ("acceptance_ratio", "Acceptance Ratio", "seagreen"),
        ("total_revenue", "Revenue", "darkorange"),
        ("total_deploy_cost", "Deploy Cost", "firebrick"),
        ("hl_reward", "HL Reward", "steelblue"),
        ("avg_link_util", "Link Util", "goldenrod"),
        ("avg_cpu_util", "CPU Util", "cadetblue"),
        ("vgae_loss", "VGAE Loss", "tomato"),
        ("w_accept", "Pareto W", "darkviolet"),
    ]
    available = [(k, lab, c) for k, lab, c in overview_keys if _safe_get(rows, k) is not None]
    if available:
        n = len(available)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows))
        axes_flat = np.atleast_1d(axes).ravel()
        for ax, (key, label, color) in zip(axes_flat, available):
            y_raw = _safe_get(rows, key)
            ax.plot(episodes, y_raw, alpha=0.2, color=color, linewidth=0.7)
            if y_raw is not None and len(y_raw) >= window:
                ax.plot(
                    episodes[window - 1 :],
                    smooth(y_raw, window),
                    color=color,
                    linewidth=1.6,
                )
            ax.set_title(label)
            ax.set_xlabel("Episode")
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[len(available) :]:
            ax.set_visible(False)
        fig.suptitle("VGAE-HRL-QL Multi-Objective Training Overview", fontsize=14)
        fig.tight_layout()
        fig.savefig(out / "overview.png", dpi=150)
        plt.close(fig)

    print(f"Training plots saved to: {out.resolve()}  ({len(plotted)} series)")


def plot_screening(
    screen_json: str,
    out_dir: str,
    summary_json: str = "",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_json(screen_json)
    if not data:
        print(f"[screen] empty: {screen_json}")
        return

    by_path: Dict[str, dict] = {}
    for row in data:
        path = row.get("path", "")
        ar = float(row.get("acc_ratio", 0.0))
        cost = float(row.get("deploy_cost", 0.0))
        prev = by_path.get(path)
        if prev is None or ar > prev["acc_ratio"] or (
            ar == prev["acc_ratio"] and cost < prev["deploy_cost"]
        ):
            by_path[path] = {
                "path": path,
                "name": Path(path).name,
                "acc_ratio": ar,
                "deploy_cost": cost,
                "w_accept": row.get("w_accept"),
                "w_cost": row.get("w_cost"),
            }

    ranked = sorted(
        by_path.values(),
        key=lambda r: (-r["acc_ratio"], r["deploy_cost"], r["name"]),
    )
    names = [r["name"] for r in ranked]
    ars = [r["acc_ratio"] for r in ranked]
    costs = [r["deploy_cost"] for r in ranked]

    selected_name = None
    if summary_json and os.path.exists(summary_json):
        summary = load_json(summary_json)
        ep_tests = summary.get("episode_tests") or {}
        sel = summary.get("selected_for_pareto") or ep_tests.get("selected_for_pareto") or {}
        if sel.get("path"):
            selected_name = Path(sel["path"]).name
    if selected_name is None and ranked:
        selected_name = ranked[0]["name"]

    colors = [
        "crimson" if n == selected_name else "steelblue" for n in names
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), 5))
    bars = ax.bar(range(len(names)), ars, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Acceptance Ratio")
    ax.set_xlabel("Test episode")
    w_note = ""
    if ranked and ranked[0].get("w_accept") is not None:
        w_note = f"  (w_accept={ranked[0]['w_accept']:.3f} from checkpoint)"
    ax.set_title(f"Screening: Acceptance Ratio per Test Episode{w_note}")
    ax.grid(True, axis="y", alpha=0.3)
    if selected_name:
        ax.legend(
            [plt.Rectangle((0, 0), 1, 1, fc="crimson"),
             plt.Rectangle((0, 0), 1, 1, fc="steelblue")],
            ["Selected for Pareto", "Other tests"],
            loc="best",
        )
    for i, (bar, v) in enumerate(zip(bars, ars)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(out / "screen_ar.png", dpi=150)
    plt.close(fig)

    cost_colors = [
        "crimson" if n == selected_name else "darkorange" for n in names
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), 5))
    bars = ax.bar(
        range(len(names)), costs, color=cost_colors, edgecolor="black", linewidth=0.6
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Deployment Cost")
    ax.set_xlabel("Test episode")
    ax.set_title("Screening: Deployment Cost per Test Episode")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, costs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(out / "screen_cost.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in ranked:
        is_sel = r["name"] == selected_name
        ax.scatter(
            r["deploy_cost"],
            r["acc_ratio"],
            s=120 if is_sel else 70,
            c="crimson" if is_sel else "steelblue",
            edgecolor="black",
            zorder=3 if is_sel else 2,
            label="Selected for Pareto" if is_sel else None,
        )
        ax.annotate(
            r["name"].replace("episode_", "e"),
            (r["deploy_cost"], r["acc_ratio"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )
    ax.set_xlabel("Deployment Cost")
    ax.set_ylabel("Acceptance Ratio")
    ax.set_title("Screening: AR vs Cost (all test episodes)")
    if selected_name:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "screen_scatter.png", dpi=150)
    plt.close(fig)

    print(
        f"Screening plots saved to: {out.resolve()}  "
        f"(n={len(ranked)}, selected={selected_name})"
    )


def plot_pareto_front(pareto_json: str, out_dir: str, hv_json: str = ""):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scan = load_json(pareto_json)
    if not scan:
        print(f"[pareto] empty: {pareto_json}")
        return

    acc = np.array([float(r["acc_ratio"]) for r in scan])
    cost = np.array([float(r["deploy_cost"]) for r in scan])
    w_accept = np.array([float(r["w_accept"]) for r in scan])

    hv_data = None
    if hv_json and os.path.exists(hv_json):
        hv_data = load_json(hv_json)

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
        ref_cost = -rp.get("neg_deploy_cost", 0.0) if rp else None
        parts = []
        if hv is not None:
            parts.append(f"HV = {hv:.4f}")
        spacing = hv_data.get("spacing")
        spread = hv_data.get("spread")
        if spacing is not None:
            parts.append(f"Spacing = {spacing:.4f}")
        if spread is not None:
            parts.append(f"Spread = {spread:.4f}")
        if ref_acc is not None and ref_cost is not None:
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

    paths = sorted({r.get("path", "") for r in scan if r.get("path")})
    subtitle = ""
    if len(paths) == 1:
        subtitle = f"\n(episode: {Path(paths[0]).name})"
    elif len(paths) > 1:
        subtitle = f"\n({len(paths)} episodes)"

    ax.set_xlabel("Total Deployment Cost")
    ax.set_ylabel("Acceptance Ratio")
    ax.set_title(f"Pareto Front: Acceptance Ratio vs Deployment Cost{subtitle}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out / "pareto_front.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Pareto front plot saved to: {out_path.resolve()}")


def plot_metrics_card(hv_json: str = "", summary_json: str = "", out_dir: str = "plots"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Any] = {}
    title = "Pareto Quality Metrics"

    if hv_json and os.path.exists(hv_json):
        hv = load_json(hv_json)
        metrics = {
            "Hypervolume": hv.get("hypervolume"),
            "Spacing": hv.get("spacing"),
            "Spread (Δ)": hv.get("spread", hv.get("delta")),
            "Front size": len(hv.get("pareto_front") or []),
        }
        if hv.get("hv_min") is not None:
            metrics["HV (norm)"] = hv.get("hv_min")
        if hv.get("r2") is not None:
            metrics["R2"] = hv.get("r2")
        if hv.get("igd_plus") is not None and hv.get("ref_front_used", True):
            metrics["IGD+"] = hv.get("igd_plus")
        else:
            metrics["IGD+"] = "N/A"
        if "coverage_ab" in hv:
            metrics["Coverage C(A,B)"] = hv.get("coverage_ab")
            metrics["Coverage C(B,A)"] = hv.get("coverage_ba")

    selected = None
    if summary_json and os.path.exists(summary_json):
        summary = load_json(summary_json)
        title = f"Metrics — {summary.get('experiment', '')}".strip(" —")
        ep_tests = summary.get("episode_tests") or {}
        sel = summary.get("selected_for_pareto") or ep_tests.get("selected_for_pareto") or {}
        if sel:
            selected = (
                f"Pareto episode: {Path(sel.get('path', '')).name}\n"
                f"AR={sel.get('acc_ratio', 0):.4f}  cost={sel.get('deploy_cost', 0):.2f}"
            )
        m = summary.get("metrics") or ep_tests.get("metrics") or {}
        if m and not metrics:
            metrics = {
                "Hypervolume": m.get("hypervolume"),
                "Spacing": m.get("spacing"),
                "Spread (Δ)": m.get("spread", m.get("delta")),
                "Front size": m.get("front_size"),
                "HV (norm)": m.get("hv_normalized"),
                "R2": m.get("r2"),
                "IGD+": m.get("igd_plus") if m.get("igd_plus") is not None else "N/A",
            }
            metrics = {k: v for k, v in metrics.items() if v is not None}

    if not metrics:
        return

    labels = list(metrics.keys())
    values = [metrics[k] for k in labels]
    display = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in values]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")
    table = ax.table(
        cellText=[[lab, val] for lab, val in zip(labels, display)],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    ax.set_title(title, fontsize=13, pad=12)
    if selected:
        fig.text(0.5, 0.02, selected, ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(out / "metrics_card.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Metrics card saved to: {(out / 'metrics_card.png').resolve()}")


def plot_results_tree(results_root: str, out_root: str, logs_root: str = ""):
    root = Path(results_root)
    out_root_p = Path(out_root)
    if not root.is_dir():
        print(f"[batch] results root not found: {root}")
        return

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not subdirs:
        print(f"[batch] no experiment folders under {root}")
        return

    for exp_dir in subdirs:
        name = exp_dir.name
        out_dir = out_root_p / name
        print(f"\n=== {name} ===", flush=True)

        screen = exp_dir / "screen_ar_cost.json"
        pareto = exp_dir / "pareto_front.json"
        hv = exp_dir / "hv_result.json"
        summary = exp_dir / "summary.json"

        if screen.is_file():
            plot_screening(
                str(screen),
                str(out_dir),
                summary_json=str(summary) if summary.is_file() else "",
            )
        if pareto.is_file():
            plot_pareto_front(
                str(pareto),
                str(out_dir),
                hv_json=str(hv) if hv.is_file() else "",
            )
        if hv.is_file() or summary.is_file():
            plot_metrics_card(
                hv_json=str(hv) if hv.is_file() else "",
                summary_json=str(summary) if summary.is_file() else "",
                out_dir=str(out_dir),
            )

        if logs_root:
            csv_path = Path(logs_root) / f"{name}.csv"
            if csv_path.is_file():
                rows = load_csv(str(csv_path))
                plot_training(rows, str(out_dir))


METHOD_COLORS = {
    "HRL-VGAE": "#d62728",
    "HRL-noVGAE": "#1f77b4",
    "GreedyLowLevel": "#2ca02c",
    "GreedyFIFO": "#ff7f0e",
    "Random-Fit": "#9467bd",
    "DefaultBaseline": "#7f7f7f",
}
METHOD_MARKERS = {
    "HRL-VGAE": "o",
    "HRL-noVGAE": "s",
    "GreedyLowLevel": "^",
    "GreedyFIFO": "D",
    "Random-Fit": "v",
    "DefaultBaseline": "X",
}


def _save_fig_formats(fig, out_base: Path, formats: List[str]):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt.lstrip('.')}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _nd_mask_max_ar_min_cost(ars: np.ndarray, costs: np.ndarray) -> np.ndarray:
    n = len(ars)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and ars[j] >= ars[i] and costs[j] <= costs[i] and (ars[j] > ars[i] or costs[j] < costs[i]):
                keep[i] = False
                break
    return keep


def plot_comparative_pareto(
    base_instance: str,
    instance_records: List[Dict[str, Any]],
    out_dir: str,
    x_mode: str = "cost",
    mo_json_path: str = "",
    formats: Optional[List[str]] = None,
):
    formats = formats or ["png"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    by_abl: Dict[str, List[Dict[str, Any]]] = {}
    for r in instance_records:
        if r.get("status") == "completed" and r.get("pareto_front_path"):
            by_abl.setdefault(r["ablation"], []).append(r)
    if not by_abl:
        plt.close(fig)
        return

    for abl, runs in by_abl.items():
        all_pts = []
        for run in runs:
            scan = load_json(run["pareto_front_path"])
            for pt in scan:
                if x_mode == "cost_per_acc" and int(pt.get("accepted") or 0) <= 0:
                    continue
                x_val = float(pt["avg_cost_per_accepted"]) if x_mode == "cost_per_acc" else float(pt["deploy_cost"])
                all_pts.append((x_val, float(pt["acc_ratio"])))
        if not all_pts:
            continue
        arr = np.unique(np.array(all_pts, dtype=np.float64), axis=0)
        nd = _nd_mask_max_ar_min_cost(arr[:, 1], arr[:, 0])
        color = METHOD_COLORS.get(abl, "black")
        marker = METHOD_MARKERS.get(abl, "o")
        if np.any(~nd):
            ax.scatter(arr[~nd, 0], arr[~nd, 1], color=color, marker=marker, alpha=0.25, s=35)
        nd_pts = arr[nd]
        order = np.argsort(nd_pts[:, 0])
        nd_pts = nd_pts[order]
        ax.plot(nd_pts[:, 0], nd_pts[:, 1], color=color, marker=marker, markersize=7, linewidth=1.8, label=abl, zorder=3)

    if x_mode == "cost" and mo_json_path and os.path.isfile(mo_json_path):
        mo = load_json(mo_json_path)
        rp = mo.get("raw_ref_point")
        if rp and len(rp) == 2:
            ax.scatter([-float(rp[1])], [float(rp[0])], marker="*", s=140, c="black", zorder=4, label="Reference point")

    ax.set_xlabel("Avg Cost per Accepted SFC (↓)" if x_mode == "cost_per_acc" else "Total Deployment Cost (↓)")
    ax.set_ylabel("Acceptance Ratio (↑)")
    ax.set_title(f"Pareto Front Comparison — {base_instance}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    _save_fig_formats(fig, Path(out_dir) / "pareto_comparison", formats)


def plot_metric_comparison(
    comparative: Dict[str, Any],
    metric_key: str,
    ylabel: str,
    out_dir: str,
    formats: Optional[List[str]] = None,
):
    formats = formats or ["png"]
    by_abl = comparative.get("across_testcases_by_ablation") or {}
    methods = [m for m in (comparative.get("methods") or []) if m in by_abl]
    if not methods:
        return
    means, errs, labels = [], [], []
    for m in methods:
        st = (by_abl[m]["overall"].get(metric_key) or {})
        means.append(st.get("mean"))
        errs.append(st.get("std") or 0.0)
        labels.append(m)
    if all(v is None for v in means):
        return
    fig, ax = plt.subplots(figsize=(max(6.5, len(methods) * 1.1), 4.5))
    x = np.arange(len(methods))
    plot_vals = [v if v is not None else 0.0 for v in means]
    colors = [METHOD_COLORS.get(m, "steelblue") for m in methods]
    bars = ax.bar(x, plot_vals, yerr=[e if v is not None else 0.0 for v, e in zip(means, errs)], capsize=4, color=colors, edgecolor="black", alpha=0.88)
    for idx, (bar, v) in enumerate(zip(bars, means)):
        txt = "N/A" if v is None else f"{v:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), txt, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} across Methods")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig_formats(fig, Path(out_dir) / f"metric_{metric_key}", formats)


def plot_training_convergence(
    base_instance: str,
    instance_records: List[Dict[str, Any]],
    out_dir: str,
    window: int = 5,
    formats: Optional[List[str]] = None,
):
    formats = formats or ["png"]
    by_abl: Dict[str, List[List[dict]]] = {}
    for r in instance_records:
        if r.get("log_csv_path") and os.path.isfile(r["log_csv_path"]):
            by_abl.setdefault(r["ablation"], []).append(load_csv(r["log_csv_path"]))
    if not by_abl:
        return
    panels = [
        ("acceptance_ratio", "Acceptance Ratio"),
        ("hl_loss", "HL DQN Loss"),
        ("ll_loss", "LL DQN Loss"),
        ("vgae_loss", "VGAE Loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.ravel(), panels):
        has_data = False
        for abl, seed_runs in by_abl.items():
            series = []
            min_len = None
            for rows in seed_runs:
                y = _safe_get(rows, key)
                if y:
                    y_s = smooth(np.array(y, dtype=np.float64), min(window, len(y)))
                    series.append(y_s)
                    min_len = len(y_s) if min_len is None else min(min_len, len(y_s))
            if not series or not min_len:
                continue
            has_data = True
            mat = np.stack([s[:min_len] for s in series], axis=0)
            mean_y = np.mean(mat, axis=0)
            std_y = np.std(mat, axis=0) if mat.shape[0] > 1 else np.zeros_like(mean_y)
            xs = np.arange(1, min_len + 1)
            c = METHOD_COLORS.get(abl, "black")
            ax.plot(xs, mean_y, label=abl, color=c, linewidth=1.6)
            if mat.shape[0] > 1:
                ax.fill_between(xs, mean_y - std_y, mean_y + std_y, color=c, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
        if has_data:
            ax.legend(fontsize=7)
    fig.suptitle(f"Training Convergence — {base_instance}", fontsize=13)
    fig.tight_layout()
    _save_fig_formats(fig, Path(out_dir) / "convergence_comparison", formats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot HRL-VGAE training / screening / Pareto results"
    )
    parser.add_argument("--csv", type=str, default="", help="Training log CSV")
    parser.add_argument(
        "--screen-json",
        type=str,
        default="",
        help="screen_ar_cost.json from --ar-cost-only",
    )
    parser.add_argument(
        "--pareto-json", type=str, default="", help="pareto_front.json from --pareto-scan"
    )
    parser.add_argument("--hv-json", type=str, default="", help="hv_result.json")
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="summary.json from run_kaggle_experiments",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="",
        help="Plot every experiment folder under this directory",
    )
    parser.add_argument(
        "--logs-root",
        type=str,
        default="",
        help="With --results-root: look for {name}.csv here for training curves",
    )
    parser.add_argument("--out", type=str, default="plots")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.results_root:
        plot_results_tree(args.results_root, args.out, logs_root=args.logs_root)
    else:
        if args.csv and os.path.exists(args.csv):
            rows = load_csv(args.csv)
            plot_training(rows, args.out, args.window)
        elif args.csv:
            print(f"[warn] CSV not found: {args.csv}")

        if args.screen_json and os.path.exists(args.screen_json):
            plot_screening(
                args.screen_json,
                args.out,
                summary_json=args.summary_json,
            )
        elif args.screen_json:
            print(f"[warn] screen JSON not found: {args.screen_json}")

        if args.pareto_json and os.path.exists(args.pareto_json):
            plot_pareto_front(args.pareto_json, args.out, hv_json=args.hv_json)
        elif args.pareto_json:
            print(f"[warn] pareto JSON not found: {args.pareto_json}")

        if (args.hv_json and os.path.exists(args.hv_json)) or (
            args.summary_json and os.path.exists(args.summary_json)
        ):
            plot_metrics_card(
                hv_json=args.hv_json,
                summary_json=args.summary_json,
                out_dir=args.out,
            )