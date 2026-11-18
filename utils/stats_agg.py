import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

_T_CRIT_95 = {
    2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060,
    9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
    16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860, 21: 2.0796, 22: 2.0739,
    23: 2.0687, 24: 2.0639, 25: 2.0595, 26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452,
    30: 2.0423,
}


def _t_crit_95(df: int) -> Optional[float]:
    if df < 2:
        return None
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df=df))
    except Exception:
        return _T_CRIT_95.get(df, 1.96)


def summarize_samples(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(clean)
    if n == 0:
        return {
            "n": 0, "mean": None, "std": None, "median": None,
            "iqr": None, "ci95_low": None, "ci95_high": None,
            "status": "no_valid_samples",
        }
    arr = np.asarray(clean, dtype=np.float64)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    iqr = float(q75 - q25)
    if n == 1:
        return {
            "n": 1, "mean": mean, "std": None, "median": median,
            "iqr": iqr, "ci95_low": None, "ci95_high": None,
            "status": "single_sample",
        }
    std = float(np.std(arr, ddof=1))
    ci_low, ci_high = None, None
    if n >= 3:
        t_val = _t_crit_95(n - 1)
        if t_val is not None:
            margin = t_val * (std / math.sqrt(n))
            ci_low = float(mean - margin)
            ci_high = float(mean + margin)
    return {
        "n": n, "mean": mean, "std": std, "median": median,
        "iqr": iqr, "ci95_low": ci_low, "ci95_high": ci_high,
        "status": "ok" if n >= 3 else "insufficient_samples_for_ci",
    }


def paired_significance_test(
    samples_a: Sequence[Optional[float]],
    samples_b: Sequence[Optional[float]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    pairs: List[Tuple[float, float]] = [
        (float(a), float(b))
        for a, b in zip(samples_a, samples_b)
        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    n = len(pairs)
    if n < 5:
        return {
            "n_pairs": n, "test": None, "statistic": None,
            "p_value": None, "significant": None,
            "status": "insufficient_samples",
        }
    a_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    b_arr = np.array([p[1] for p in pairs], dtype=np.float64)
    diff = a_arr - b_arr
    if np.allclose(diff, 0.0):
        return {
            "n_pairs": n, "test": "wilcoxon", "statistic": 0.0,
            "p_value": 1.0, "significant": False, "status": "identical_samples",
        }
    try:
        from scipy import stats
        stat, p_val = stats.wilcoxon(a_arr, b_arr)
        return {
            "n_pairs": n, "test": "wilcoxon",
            "statistic": float(stat), "p_value": float(p_val),
            "significant": bool(p_val < alpha), "status": "ok",
        }
    except Exception:
        pos = int(np.sum(diff > 0.0))
        neg = int(np.sum(diff < 0.0))
        m = pos + neg
        if m == 0:
            p_val = 1.0
        else:
            k = min(pos, neg)
            p_val = min(1.0, 2.0 * sum(math.comb(m, i) for i in range(k + 1)) * (0.5 ** m))
        return {
            "n_pairs": n, "test": "sign_test",
            "statistic": float(pos - neg), "p_value": float(p_val),
            "significant": bool(p_val < alpha), "status": "ok",
        }


def aggregate_experiment_records(records: List[Dict[str, Any]], metrics: Sequence[str]) -> Dict[str, Any]:
    by_inst_abl: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in records:
        if r.get("status") != "completed":
            continue
        key = (str(r["base_instance"]), str(r["ablation"]))
        by_inst_abl.setdefault(key, []).append(r)

    across_seeds_per_instance: Dict[str, Dict[str, Any]] = {}
    per_inst_abl_means: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    inst_meta: Dict[str, Dict[str, str]] = {}
    for (inst, abl), rows in sorted(by_inst_abl.items()):
        inst_meta[inst] = {
            "topology": rows[0].get("topology", ""),
            "allocation": rows[0].get("allocation", ""),
            "difficulty": rows[0].get("difficulty", ""),
        }
        m_stats = {}
        m_means = {}
        for m in metrics:
            vals = [row.get("metrics", {}).get(m) for row in rows]
            s = summarize_samples(vals)
            m_stats[m] = s
            m_means[m] = s["mean"]
        across_seeds_per_instance[f"{inst}::{abl}"] = {
            "base_instance": inst, "ablation": abl,
            "seeds": [row.get("seed") for row in rows],
            "metrics": m_stats,
        }
        per_inst_abl_means[(inst, abl)] = m_means

    across_testcases_by_ablation: Dict[str, Dict[str, Any]] = {}
    ablations = sorted({abl for _, abl in per_inst_abl_means})
    for abl in ablations:
        inst_rows = [(inst, per_inst_abl_means[(inst, abl)]) for inst, a in per_inst_abl_means if a == abl]
        overall = {m: summarize_samples([vals.get(m) for _, vals in inst_rows]) for m in metrics}
        by_topo: Dict[str, Any] = {}
        for topo in sorted({inst_meta[inst]["topology"] for inst, _ in inst_rows if inst in inst_meta}):
            t_rows = [vals for inst, vals in inst_rows if inst_meta.get(inst, {}).get("topology") == topo]
            by_topo[topo] = {m: summarize_samples([v.get(m) for v in t_rows]) for m in metrics}
        by_diff: Dict[str, Any] = {}
        for diff in sorted({inst_meta[inst]["difficulty"] for inst, _ in inst_rows if inst in inst_meta}):
            d_rows = [vals for inst, vals in inst_rows if inst_meta.get(inst, {}).get("difficulty") == diff]
            by_diff[diff] = {m: summarize_samples([v.get(m) for v in d_rows]) for m in metrics}
        across_testcases_by_ablation[abl] = {
            "num_testcases": len(inst_rows),
            "overall": overall,
            "by_topology": by_topo,
            "by_difficulty": by_diff,
        }

    return {
        "across_seeds_per_instance": across_seeds_per_instance,
        "across_testcases_by_ablation": across_testcases_by_ablation,
    }


METHOD_ORDER = ("HRL-VGAE", "HRL-noVGAE", "GreedyLowLevel", "GreedyFIFO", "Random-Fit", "DefaultBaseline")
METRIC_META = {
    "mean_ar": {"label": "Acceptance Ratio", "higher_better": True},
    "mean_cost_per_acc": {"label": "Cost / Acc", "higher_better": False},
    "hv_normalized": {"label": "HV (norm)", "higher_better": True},
    "r2": {"label": "R2", "higher_better": False},
    "igd_plus": {"label": "IGD+", "higher_better": False},
    "duration_sec": {"label": "Runtime (s)", "higher_better": False},
}
ABLATION_COMPONENT_MAP = {
    "HRL-noVGAE": "MNVGAE graph representation",
    "GreedyLowLevel": "Low-Level RL node placement agent",
    "Random-Fit": "High-Level RL SFC scheduler (vs Random)",
    "GreedyFIFO": "High-Level RL SFC scheduler (vs FIFO)",
    "DefaultBaseline": "Full HRL-VGAE framework (vs EDF + MaxCPU heuristic)",
}


def _ordered_methods(methods: Sequence[str]) -> List[str]:
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(set(methods), key=lambda m: (rank.get(m, 999), m))


def _finite_or_none(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_read_json(path: Path, default: Any):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"[WARN] Unreadable JSON {path}: {e}")
        return default


def load_experiment_records(results_root: Path, logs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not results_root.is_dir():
        return records
    for exp_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        m_path = exp_dir / "manifest.json"
        s_path = exp_dir / "summary.json"
        scr_path = exp_dir / "screen_ar_cost.json"
        hv_path = exp_dir / "hv_result.json"
        pf_path = exp_dir / "pareto_front.json"
        if not m_path.is_file() and not s_path.is_file():
            continue
        mdata = _safe_read_json(m_path, {}) if m_path.is_file() else {}
        sdata = _safe_read_json(s_path, {}) if s_path.is_file() else {}
        ec = mdata.get("effective_config") or {}
        ep_tests = sdata.get("episode_tests") or {}
        sel = sdata.get("selected_for_pareto") or ep_tests.get("selected_for_pareto") or {}
        s_metrics = sdata.get("metrics") or ep_tests.get("metrics") or {}
        m_metrics = mdata.get("metrics") or {}

        scr_rows = _safe_read_json(scr_path, []) if scr_path.is_file() else []
        ep_ars = [_finite_or_none(r.get("acc_ratio")) for r in scr_rows]
        ep_costs = [_finite_or_none(r.get("deploy_cost")) for r in scr_rows]
        ep_cpa: List[Optional[float]] = []
        zero_acc_count = 0
        for r in scr_rows:
            acc = int(r.get("accepted") or 0)
            dc = _finite_or_none(r.get("deploy_cost"))
            if acc > 0 and dc is not None:
                ep_cpa.append(dc / acc)
            else:
                zero_acc_count += 1
                ep_cpa.append(None)

        ar_stat = summarize_samples(ep_ars)
        cpa_stat = summarize_samples(ep_cpa)
        cost_stat = summarize_samples(ep_costs)

        exp_id = mdata.get("experiment_id") or sdata.get("experiment") or exp_dir.name
        ablation = ec.get("ablation") or sdata.get("ablation") or "HRL-VGAE"
        base_instance = ec.get("base_instance") or "_".join(exp_dir.name.split("_")[:3])
        parts = base_instance.split("_")
        topo = ec.get("topology") or (parts[0] if len(parts) >= 1 else "")
        alloc = ec.get("allocation") or (parts[1] if len(parts) >= 2 else "")
        diff = ec.get("difficulty") or (parts[2] if len(parts) >= 3 else "")
        seed = ec.get("seed")
        if seed is None:
            seed = 42

        hv_data = _safe_read_json(hv_path, {}) if hv_path.is_file() else {}
        ref_used = hv_data.get("ref_front_used", False)
        igd_val = _finite_or_none(s_metrics.get("igd_plus"))
        if igd_val is None and ref_used:
            igd_val = _finite_or_none(hv_data.get("igd_plus"))
        if igd_val is None:
            igd_val = _finite_or_none(m_metrics.get("igd_plus"))

        csv_path = (logs_root / f"{exp_id}.csv") if logs_root is not None else None
        metrics = {
            "mean_ar": ar_stat["mean"] if ar_stat["mean"] is not None else _finite_or_none(m_metrics.get("mean_ar")),
            "mean_cost_per_acc": cpa_stat["mean"],
            "mean_deploy_cost": cost_stat["mean"] if cost_stat["mean"] is not None else _finite_or_none(m_metrics.get("mean_deploy_cost")),
            "hypervolume": _finite_or_none(s_metrics.get("hypervolume")) if s_metrics.get("hypervolume") is not None else _finite_or_none(hv_data.get("hypervolume")),
            "hv_normalized": _finite_or_none(s_metrics.get("hv_normalized")) if s_metrics.get("hv_normalized") is not None else _finite_or_none(m_metrics.get("hv_normalized")),
            "r2": _finite_or_none(s_metrics.get("r2")) if s_metrics.get("r2") is not None else _finite_or_none(m_metrics.get("r2")),
            "igd_plus": igd_val,
            "spacing": _finite_or_none(s_metrics.get("spacing")),
            "spread": _finite_or_none(s_metrics.get("spread")),
            "duration_sec": _finite_or_none(mdata.get("duration_sec")),
        }
        records.append({
            "experiment_id": exp_id,
            "exp_dir": str(exp_dir),
            "base_instance": base_instance,
            "topology": topo,
            "allocation": alloc,
            "difficulty": diff,
            "ablation": ablation,
            "seed": int(seed),
            "test_seed": ec.get("test_seed"),
            "status": mdata.get("status") or ("completed" if s_path.is_file() and pf_path.is_file() else "incomplete"),
            "has_manifest": m_path.is_file(),
            "config_fingerprint": mdata.get("config_fingerprint"),
            "checkpoint": mdata.get("eval_checkpoint") or mdata.get("checkpoint"),
            "pareto_episode": ec.get("pareto_episode") or (Path(sel["path"]).name if sel.get("path") else ""),
            "test_episodes": [Path(p).name for p in (mdata.get("test_episodes") or [r.get("path", "") for r in scr_rows if r.get("path")])],
            "normalization_scope": s_metrics.get("normalization_scope"),
            "cost_min_used": _finite_or_none(s_metrics.get("cost_min_used")),
            "cost_max_used": _finite_or_none(s_metrics.get("cost_max_used")),
            "zero_accepted_episodes": zero_acc_count,
            "episode_variance": {"ar": ar_stat, "cost_per_acc": cpa_stat, "deploy_cost": cost_stat},
            "pareto_front_path": str(pf_path) if pf_path.is_file() else None,
            "hv_result_path": str(hv_path) if hv_path.is_file() else None,
            "log_csv_path": str(csv_path) if (csv_path is not None and csv_path.is_file()) else None,
            "metrics": metrics,
        })
    return records


def check_comparability_group(instance_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [r for r in instance_records if r.get("status") == "completed"]
    warnings: List[str] = []
    if not completed:
        return {"comparable": False, "warnings": ["no_completed_runs"]}
    pareto_eps = {r.get("pareto_episode") for r in completed if r.get("pareto_episode")}
    if len(pareto_eps) > 1:
        warnings.append(f"pareto_episode_mismatch: {sorted(pareto_eps)}")
    ep_lists = {tuple(r.get("test_episodes") or []) for r in completed if r.get("test_episodes")}
    if len(ep_lists) > 1:
        warnings.append("test_episodes_mismatch")
    tseeds = {r.get("test_seed") for r in completed if r.get("test_seed") is not None}
    if len(tseeds) > 1:
        warnings.append(f"test_seed_mismatch: {sorted(tseeds)}")
    if len({r.get("ablation") for r in completed}) > 1:
        scopes = {r.get("normalization_scope") for r in completed}
        cmins = {round(r["cost_min_used"], 6) for r in completed if r.get("cost_min_used") is not None}
        cmaxs = {round(r["cost_max_used"], 6) for r in completed if r.get("cost_max_used") is not None}
        if "cross_ablation" not in scopes or len(cmins) != 1 or len(cmaxs) != 1:
            warnings.append("cross_ablation_normalization_missing_or_mismatched")
    return {"comparable": len(warnings) == 0, "warnings": warnings}


def build_comparative_analysis(
    records: List[Dict[str, Any]],
    metrics: Sequence[str] = ("mean_ar", "mean_cost_per_acc", "hv_normalized", "r2", "igd_plus", "duration_sec"),
) -> Dict[str, Any]:
    completed = [r for r in records if r.get("status") == "completed"]
    incomplete = [
        {"experiment_id": r["experiment_id"], "ablation": r["ablation"], "seed": r["seed"], "status": r["status"]}
        for r in records if r.get("status") != "completed"
    ]
    by_inst: Dict[str, List[Dict[str, Any]]] = {}
    for r in completed:
        by_inst.setdefault(r["base_instance"], []).append(r)

    comparability = {inst: check_comparability_group(rows) for inst, rows in sorted(by_inst.items())}
    core_agg = aggregate_experiment_records(completed, metrics=metrics)

    methods = _ordered_methods([r["ablation"] for r in completed])
    by_key = {(r["base_instance"], r["ablation"], r["seed"]): r["metrics"] for r in completed}
    pairwise: Dict[str, Any] = {}
    for ref_m in ("HRL-VGAE", "DefaultBaseline"):
        if ref_m not in methods:
            continue
        pairwise[ref_m] = {}
        for other_m in methods:
            if other_m == ref_m:
                continue
            matched_keys = sorted(
                (inst, seed)
                for (inst, abl, seed) in by_key
                if abl == other_m and (inst, ref_m, seed) in by_key and comparability.get(inst, {}).get("comparable", False)
            )
            m_res: Dict[str, Any] = {"n_matched_runs": len(matched_keys), "metrics": {}}
            for m in metrics:
                a_vals = [by_key[(inst, other_m, s)].get(m) for inst, s in matched_keys]
                b_vals = [by_key[(inst, ref_m, s)].get(m) for inst, s in matched_keys]
                diffs = [a - b for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
                m_res["metrics"][m] = {
                    "delta_summary": summarize_samples(diffs),
                    "test": paired_significance_test(a_vals, b_vals),
                }
            pairwise[ref_m][other_m] = m_res

    return {
        "methods": methods,
        "num_completed": len(completed),
        "incomplete_runs": incomplete,
        "comparability_by_instance": comparability,
        "across_seeds_per_instance": core_agg["across_seeds_per_instance"],
        "across_testcases_by_ablation": core_agg["across_testcases_by_ablation"],
        "pairwise_vs_reference": pairwise,
    }


def build_ablation_contributions(comparative: Dict[str, Any], reference: str = "HRL-VGAE") -> List[Dict[str, Any]]:
    pw = (comparative.get("pairwise_vs_reference") or {}).get(reference) or {}
    rows: List[Dict[str, Any]] = []
    for abl, comp_desc in ABLATION_COMPONENT_MAP.items():
        if abl not in pw:
            continue
        entry = {"ablation": abl, "ablated_component": comp_desc, "n_matched_runs": pw[abl]["n_matched_runs"], "deltas": {}}
        for m, m_info in pw[abl]["metrics"].items():
            ds = m_info["delta_summary"]
            ts = m_info["test"]
            entry["deltas"][m] = {
                "mean_delta": ds["mean"], "std_delta": ds["std"],
                "p_value": ts["p_value"], "significant": ts["significant"], "test_status": ts["status"],
            }
        rows.append(entry)
    return rows


def export_summary_csv(comparative: Dict[str, Any], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scope", "group", "ablation", "metric", "n", "mean", "std", "median", "iqr", "ci95_low", "ci95_high", "comparable", "status"]
    rows: List[Dict[str, Any]] = []
    comp_map = comparative.get("comparability_by_instance") or {}
    for _, entry in sorted((comparative.get("across_seeds_per_instance") or {}).items()):
        inst = entry["base_instance"]
        is_comp = comp_map.get(inst, {}).get("comparable", False)
        for m, st in entry["metrics"].items():
            rows.append({"scope": "per_instance_seeds", "group": inst, "ablation": entry["ablation"], "metric": m, "comparable": is_comp, **st})
    for abl, adata in (comparative.get("across_testcases_by_ablation") or {}).items():
        for m, st in adata["overall"].items():
            rows.append({"scope": "overall_testcases", "group": "all", "ablation": abl, "metric": m, "comparable": "", **st})
        for topo, tm in adata["by_topology"].items():
            for m, st in tm.items():
                rows.append({"scope": "by_topology", "group": topo, "ablation": abl, "metric": m, "comparable": "", **st})
        for diff, dm in adata["by_difficulty"].items():
            for m, st in dm.items():
                rows.append({"scope": "by_difficulty", "group": diff, "ablation": abl, "metric": m, "comparable": "", **st})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _fmt_mean_std(st: Dict[str, Any], digits: int = 3) -> str:
    if st.get("mean") is None:
        return "N/A"
    if st.get("std") is None:
        return f"{st['mean']:.{digits}f}"
    return f"{st['mean']:.{digits}f} $\\pm$ {st['std']:.{digits}f}"


def export_latex_tables(comparative: Dict[str, Any], out_tex: Path):
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    cols = ("mean_ar", "mean_cost_per_acc", "hv_normalized", "r2", "igd_plus")
    lines = ["% Auto-generated HRL-VGAE Comparative Tables (requires booktabs)", ""]
    by_abl = comparative.get("across_testcases_by_ablation") or {}
    methods = _ordered_methods(list(by_abl.keys()))
    lines.extend([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Overall comparison across testcases (mean $\\pm$ std over testcase means).}",
        "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
        "Method & AR ($\\uparrow$) & Cost/Acc ($\\downarrow$) & HV ($\\uparrow$) & R2 ($\\downarrow$) & IGD+ ($\\downarrow$) \\\\",
        "\\midrule",
    ])
    for m_name in methods:
        ov = by_abl[m_name]["overall"]
        vals = [_fmt_mean_std(ov[c], 4 if c != "mean_cost_per_acc" else 2) for c in cols]
        lines.append(f"{m_name} & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    out_tex.write_text("\n".join(lines))


def export_markdown_report(
    comparative: Dict[str, Any],
    ablation_rows: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    out_md: Path,
):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    by_abl = comparative.get("across_testcases_by_ablation") or {}
    methods = _ordered_methods(list(by_abl.keys()))
    lines = ["# HRL-VGAE Experimental Research Summary", ""]
    lines.append(f"- Completed runs: {comparative.get('num_completed', 0)}")
    lines.append(f"- Incomplete/failed runs: {len(comparative.get('incomplete_runs') or [])}")
    non_comp = [k for k, v in (comparative.get("comparability_by_instance") or {}).items() if not v.get("comparable")]
    if non_comp:
        lines.append(f"- Comparability warnings on instances: {', '.join(non_comp)}")
    lines.extend(["", "## 1. Overall Method Comparison (Testcase-Level Means)", ""])
    lines.append("| Method | Testcases | AR (↑) | Cost/Acc (↓) | HV norm (↑) | R2 (↓) | IGD+ (↓) | Runtime s (↓) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m_name in methods:
        ov = by_abl[m_name]["overall"]

        def _md(k, d=4):
            st = ov.get(k) or {}
            if st.get("mean") is None:
                return "N/A"
            return f"{st['mean']:.{d}f}" + (f" ± {st['std']:.{d}f}" if st.get("std") is not None else "")

        lines.append(
            f"| {m_name} | {by_abl[m_name]['num_testcases']} | {_md('mean_ar')} | {_md('mean_cost_per_acc', 2)} | "
            f"{_md('hv_normalized')} | {_md('r2')} | {_md('igd_plus')} | {_md('duration_sec', 1)} |"
        )
    if ablation_rows:
        lines.extend(["", "## 2. Ablation Analysis vs HRL-VGAE (Matched Runs)", ""])
        lines.append("| Ablation | Component Tested | Matched Pairs | ΔAR | ΔCost/Acc | ΔHV | ΔIGD+ | AR Test Status |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for row in ablation_rows:
            def _d(k, fmt=".4f"):
                v = row["deltas"].get(k, {}).get("mean_delta")
                return "N/A" if v is None else format(v, fmt)
            ar_t = row["deltas"].get("mean_ar", {})
            t_str = f"p={ar_t['p_value']:.4f}" if ar_t.get("p_value") is not None else str(ar_t.get("test_status"))
            lines.append(
                f"| {row['ablation']} | {row['ablated_component']} | {row['n_matched_runs']} | "
                f"{_d('mean_ar', '+.4f')} | {_d('mean_cost_per_acc', '+.2f')} | {_d('hv_normalized', '+.4f')} | {_d('igd_plus', '+.4f')} | {t_str} |"
            )
    if comparative.get("incomplete_runs"):
        lines.extend(["", "## 3. Incomplete / Failed Runs", ""])
        for r in comparative["incomplete_runs"]:
            lines.append(f"- `{r['experiment_id']}` ({r['ablation']}, seed={r['seed']}): `{r['status']}`")
    out_md.write_text("\n".join(lines) + "\n")