"""
Kaggle experiment runner for HRL-VGAE.

Topology : nsf, cogent, conus
Allocation: centers, uniform, rural, urban
Difficulty: easy, normal, hard
"""

from __future__ import annotations

import numpy as np
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

TOPOLOGIES = ("nsf", "cogent", "conus")
ALLOCATIONS = ("centers", "uniform", "rural", "urban")
DIFFICULTIES = ("easy", "normal", "hard")

ABLATIONS = ("DefaultBaseline", "Random-Fit", "GreedyFIFO", "GreedyLowLevel", "HRL-VGAE", "HRL-noVGAE")
DEFAULT_ABLATION = "HRL-VGAE"

NUM_TEST_EPISODES = 10
DEFAULT_SEED = 42


def ablation_suffix(ablation: str) -> str:
    return "" if ablation == DEFAULT_ABLATION else f"_{ablation}"


def experiment_id_for(
    topology: str,
    allocation: str,
    difficulty: str,
    ablation: str,
    seed: int,
    multi_seed: bool = False,
) -> str:
    base = f"{topology}_{allocation}_{difficulty}{ablation_suffix(ablation)}"
    if multi_seed or seed != DEFAULT_SEED:
        return f"{base}_seed{seed}"
    return base


def _atomic_write_json(path: Path, data: Any):
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _compute_config_fingerprint(cfg_dict: Dict[str, Any]) -> str:
    payload = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _collect_env_info() -> Dict[str, Any]:
    has_cuda = torch.cuda.is_available()
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": has_cuda,
        "gpu_count": torch.cuda.device_count() if has_cuda else 0,
        "gpu_name": torch.cuda.get_device_name(0) if has_cuda else None,
    }


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def shared_test_dir(test_root: Path, topology: str, allocation: str, difficulty: str) -> Path:
    return test_root / topology / allocation / difficulty


def _list_episode_files(d: Path) -> List[Path]:
    return sorted(d.glob("episode_*.json")) if d.is_dir() else []


def _episode_id(p: Path) -> int:
    stem = p.stem
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def ensure_shared_test_episodes(
    topology: str,
    allocation: str,
    difficulty: str,
    testcase_dir: Path,
    test_root: Path,
    num_episodes: int,
    seed: int,
    dry_run: bool,
) -> List[Path]:
    out_dir = shared_test_dir(test_root, topology, allocation, difficulty)
    existing = _list_episode_files(out_dir)

    if len(existing) >= num_episodes:
        selected = existing[:num_episodes]
        print(
            f"[EPISODES] reusing {len(selected)} existing episodes: {out_dir}",
            flush=True,
        )
        return selected

    tc = testcase_path(testcase_dir, topology, allocation, difficulty)
    if not tc.is_file():
        print(f"[ERROR] testcase missing, cannot generate episodes: {tc}", flush=True)
        return existing

    if len(existing) == 0:
        rc = generate_tests(tc, out_dir, num_episodes, seed, dry_run)
        if rc != 0:
            print(f"[FAIL] generate episodes: {topology}_{allocation}_{difficulty}", flush=True)
        existing = _list_episode_files(out_dir)
        return existing[:num_episodes]

    present_ids = {_episode_id(p) for p in existing}
    missing_count = num_episodes - len(existing)
    missing_ids = [i for i in range(num_episodes) if i not in present_ids][:missing_count]
    if missing_ids:
        print(
            f"[EPISODES] {len(existing)}/{num_episodes} present; generating missing IDs {missing_ids}",
            flush=True,
        )
        for ep_id in missing_ids:
            ep_seed = seed + ep_id
            _generate_single_episode(tc, out_dir, ep_id, ep_seed, dry_run)
    existing = _list_episode_files(out_dir)
    selected = existing[:num_episodes]
    print(
        f"[EPISODES] using {len(selected)} episodes: {out_dir}",
        flush=True,
    )
    return selected


def _generate_single_episode(
    topology_json: Path,
    out_dir: Path,
    ep_id: int,
    seed: int,
    dry_run: bool,
) -> int:
    ensure_dir(out_dir)
    tmp_dir = out_dir / f"_tmp_ep{ep_id}"
    ensure_dir(tmp_dir)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "data" / "generate_episodes.py"),
        "--topology", str(topology_json),
        "--num-episodes", "1",
        "--infer-cfg",
        "--seed", str(seed),
        "--out-dir", str(tmp_dir),
    ]
    rc = run_cmd(cmd, dry_run=dry_run)
    if not dry_run and rc == 0:
        tmp_files = sorted(tmp_dir.glob("episode_*.json"))
        if tmp_files:
            dest = out_dir / f"episode_{ep_id:04d}.json"
            tmp_files[0].rename(dest)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return rc


def _load_pareto_indicators():
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from utils.pareto_indicator import (
        build_pareto_set_from_scan,
        normalize_pareto_objectives,
        compute_hv_normalized,
        compute_r2,
        compute_igd_plus,
        build_empirical_reference_front,
    )
    return (
        build_pareto_set_from_scan,
        normalize_pareto_objectives,
        compute_hv_normalized,
        compute_r2,
        compute_igd_plus,
        build_empirical_reference_front,
    )


def compute_mo_indicators(
    pareto_scan_path: Path,
    cost_bounds: Tuple[float, float],
    ideal_point: np.ndarray,
    reference_front_norm: Optional[np.ndarray] = None,
    hv_ref_point: np.ndarray = None,
    n_weights: int = 101,
) -> Dict[str, Any]:
    (
        build_pareto_set_from_scan,
        normalize_pareto_objectives,
        compute_hv_normalized,
        compute_r2,
        compute_igd_plus,
        _,
    ) = _load_pareto_indicators()

    if not pareto_scan_path.is_file():
        return {}
    with open(pareto_scan_path) as f:
        scan = json.load(f)
    if not scan:
        return {}

    pareto_set = build_pareto_set_from_scan(scan)
    if pareto_set.shape[0] == 0:
        return {}

    cost_min, cost_max = cost_bounds
    pareto_norm = normalize_pareto_objectives(pareto_set, cost_min, cost_max)

    if hv_ref_point is None:
        hv_ref_point = np.array([1.1, 1.1])

    hv = compute_hv_normalized(pareto_norm, hv_ref_point)
    r2 = compute_r2(pareto_norm, ideal_point, n_weights)
    igd_plus = (
        compute_igd_plus(pareto_norm, reference_front_norm)
        if reference_front_norm is not None and len(reference_front_norm) > 0
        else None
    )

    result = {
        "hv_normalized": _safe_float(hv),
        "r2": _safe_float(r2),
        "pareto_set_size": int(pareto_set.shape[0]),
        "cost_min_used": float(cost_min),
        "cost_max_used": float(cost_max),
    }
    if igd_plus is not None:
        result["igd_plus"] = _safe_float(igd_plus)
    return result


def _collect_all_pareto_sets_for_instance(
    experiment_name: str,
    ablations: List[str],
    results_root: Path,
    seeds: Optional[List[int]] = None,
    multi_seed: bool = False,
) -> Dict[str, np.ndarray]:
    (build_pareto_set_from_scan, *_) = _load_pareto_indicators()
    sets = {}
    seed_list = seeds or [DEFAULT_SEED]
    for abl in ablations:
        for s in seed_list:
            exp_id = f"{experiment_name}{ablation_suffix(abl)}" + (f"_seed{s}" if (multi_seed or s != DEFAULT_SEED) else "")
            pf = results_root / exp_id / "pareto_front.json"
            if pf.is_file():
                with open(pf) as f:
                    scan = json.load(f)
                ps = build_pareto_set_from_scan(scan)
                if ps.shape[0] > 0:
                    sets[exp_id] = ps
    return sets


def _collect_all_raw_scans_for_instance(
    experiment_name: str,
    ablations: List[str],
    results_root: Path,
    seeds: Optional[List[int]] = None,
    multi_seed: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    scans: Dict[str, List[Dict[str, Any]]] = {}
    seed_list = seeds or [DEFAULT_SEED]
    for abl in ablations:
        for s in seed_list:
            exp_id = f"{experiment_name}{ablation_suffix(abl)}" + (f"_seed{s}" if (multi_seed or s != DEFAULT_SEED) else "")
            pf = results_root / exp_id / "pareto_front.json"
            if pf.is_file():
                with open(pf) as f:
                    data = json.load(f)
                if data:
                    scans[exp_id] = data
    return scans


def _infer_cost_bounds(
    results_root: Path,
    experiment_name: str,
    ablations: List[str],
    seeds: Optional[List[int]] = None,
    multi_seed: bool = False,
) -> Tuple[float, float]:
    scans = _collect_all_raw_scans_for_instance(experiment_name, ablations, results_root, seeds=seeds, multi_seed=multi_seed)
    all_costs = []
    for rows in scans.values():
        all_costs.extend(float(r["deploy_cost"]) for r in rows if "deploy_cost" in r)
    if not all_costs:
        return 0.0, 1.0
    c_min, c_max = min(all_costs), max(all_costs)
    if abs(c_max - c_min) < 1e-12:
        c_max = c_min + 1.0
    return c_min, c_max


def compute_and_save_mo_indicators_for_experiment(
    experiment_name: str,
    ablations: List[str],
    results_root: Path,
    n_weights: int = 101,
    c_min_override: Optional[float] = None,
    c_max_override: Optional[float] = None,
    ref_acc_override: Optional[float] = None,
    ref_cost_override: Optional[float] = None,
    seeds: Optional[List[int]] = None,
    multi_seed: bool = False,
):
    (
        build_pareto_set_from_scan,
        normalize_pareto_objectives,
        compute_hv_normalized,
        compute_r2,
        compute_igd_plus,
        build_empirical_reference_front,
    ) = _load_pareto_indicators()
    from evaluate import _compute_hv

    scans = _collect_all_raw_scans_for_instance(experiment_name, ablations, results_root, seeds=seeds, multi_seed=multi_seed)
    if c_min_override is not None and c_max_override is not None:
        cost_min, cost_max = float(c_min_override), float(c_max_override)
        if abs(cost_max - cost_min) < 1e-12:
            cost_max = cost_min + 1.0
    else:
        cost_min, cost_max = _infer_cost_bounds(results_root, experiment_name, ablations, seeds=seeds, multi_seed=multi_seed)
    hv_ref = np.array([1.1, 1.1])
    ideal = np.array([0.0, 0.0])

    all_sets = _collect_all_pareto_sets_for_instance(experiment_name, ablations, results_root, seeds=seeds, multi_seed=multi_seed)
    if not all_sets:
        print(f"[CROSS-MO WARN] {experiment_name}: no valid pareto_front.json found across {ablations}", flush=True)
        return

    ep_paths = {
        exp_id: sorted({r.get("path", "") for r in rows if r.get("path")})
        for exp_id, rows in scans.items()
    }
    unique_ep_sets = {tuple(v) for v in ep_paths.values() if v}
    if len(unique_ep_sets) > 1:
        print(f"[WARN] {experiment_name}: runs used different Pareto episodes: {ep_paths}", flush=True)

    if ref_acc_override is not None and ref_cost_override is not None:
        raw_ref = np.array([float(ref_acc_override), -float(ref_cost_override)], dtype=np.float64)
    else:
        all_ars = [float(r["acc_ratio"]) for rows in scans.values() for r in rows if "acc_ratio" in r]
        all_raw_costs = [float(r["deploy_cost"]) for rows in scans.values() for r in rows if "deploy_cost" in r]
        min_ar = min(all_ars) if all_ars else 0.0
        max_c = max(all_raw_costs) if all_raw_costs else cost_max
        raw_ref = np.array([min_ar - 1e-2, -max_c - 1e-2], dtype=np.float64)

    ref_front_raw = build_empirical_reference_front(list(all_sets.values()))
    ref_front_norm = (
        normalize_pareto_objectives(ref_front_raw, cost_min, cost_max)
        if ref_front_raw.shape[0] > 0
        else np.empty((0, 2))
    )
    ref_front_path = results_root / f"{experiment_name}_empirical_ref_front.json"
    _atomic_write_json(ref_front_path, ref_front_norm.tolist())

    single_default_seed = (not multi_seed) and (seeds is None or seeds == [DEFAULT_SEED])
    mo_results = {}
    summary_table_csv = results_root / "ablation_summary.csv"
    updated_csv_rows = []
    for exp_name, ps in all_sets.items():
        res_dir = results_root / exp_name
        pareto_out = res_dir / "pareto_front.json"
        hv_out = res_dir / "hv_result.json"
        summary_out = res_dir / "summary.json"
        screen_out = res_dir / "screen_ar_cost.json"
        manifest_out = res_dir / "manifest.json"

        _compute_hv(
            str(pareto_out),
            str(hv_out),
            1e-2,
            ref_point=raw_ref,
            ref_front_json=str(ref_front_path),
            c_min_override=cost_min,
            c_max_override=cost_max,
        )
        ps_norm = normalize_pareto_objectives(ps, cost_min, cost_max)
        hv = compute_hv_normalized(ps_norm, hv_ref)
        r2 = compute_r2(ps_norm, ideal, n_weights)
        igd = compute_igd_plus(ps_norm, ref_front_norm) if ref_front_norm.shape[0] > 0 else None

        abl_name = ""
        if summary_out.is_file():
            with open(summary_out) as sf:
                abl_name = json.load(sf).get("ablation", "")
        if not abl_name:
            for cand_abl in ablations:
                if exp_name == f"{experiment_name}{ablation_suffix(cand_abl)}":
                    abl_name = cand_abl
                    break
        mo_key = abl_name if (single_default_seed and abl_name) else exp_name

        mo_results[mo_key] = {
            "hv_normalized": _safe_float(hv),
            "r2": _safe_float(r2),
            "pareto_set_size": int(ps.shape[0]),
            "cost_min_used": float(cost_min),
            "cost_max_used": float(cost_max),
        }
        if igd is not None:
            mo_results[mo_key]["igd_plus"] = _safe_float(igd)

        hv_raw_val = sp_val = spread_val = ""
        if hv_out.is_file():
            with open(hv_out) as f:
                hd = json.load(f)
            hv_raw_val = hd.get("hypervolume", "")
            sp_val = hd.get("spacing", "")
            spread_val = hd.get("spread", hd.get("delta", ""))
            mo_results[mo_key]["hypervolume"] = hv_raw_val

        if summary_out.is_file():
            with open(summary_out) as f:
                sdata = json.load(f)
            abl = sdata.get("ablation", abl_name)
            ep_metrics = (sdata.get("episode_tests") or {}).get("metrics") or {}
            ep_metrics.update({
                "hypervolume": hv_raw_val,
                "spacing": sp_val,
                "spread": spread_val,
                "hv_normalized": _safe_float(hv),
                "r2": _safe_float(r2),
                "cost_min_used": float(cost_min),
                "cost_max_used": float(cost_max),
                "normalization_scope": "cross_ablation",
            })
            if igd is not None:
                ep_metrics["igd_plus"] = _safe_float(igd)
            sdata.setdefault("episode_tests", {})["metrics"] = ep_metrics
            sdata["metrics"] = dict(ep_metrics)
            _atomic_write_json(summary_out, sdata)
            if manifest_out.is_file():
                with open(manifest_out) as mf:
                    mdata = json.load(mf)
                mdata.setdefault("metrics", {}).update(ep_metrics)
                _atomic_write_json(manifest_out, mdata)

            sel_path = ((sdata.get("episode_tests") or {}).get("selected_for_pareto") or {}).get("path", "")
            sel_name = Path(sel_path).name if sel_path else ""
            if screen_out.is_file():
                with open(screen_out) as f:
                    ep_data = json.load(f)
                for r in ep_data:
                    rp = r.get("path", "")
                    is_sel = bool(sel_path) and (rp == sel_path or (rp and Path(rp).name == sel_name))
                    updated_csv_rows.append({
                        "experiment": exp_name, "ablation": abl, "test_type": "episode_test",
                        "path": rp,
                        "acc_ratio": r.get("acc_ratio", ""), "deploy_cost": r.get("deploy_cost", ""),
                        "avg_cost_per_accepted": r.get("avg_cost_per_accepted", ""),
                        "accepted": r.get("accepted", ""), "rejected": r.get("rejected", ""),
                        "hypervolume": hv_raw_val if is_sel else "",
                        "spacing": sp_val if is_sel else "",
                        "spread": spread_val if is_sel else "",
                        "hv_normalized": _safe_float(hv) if is_sel else "",
                        "r2": _safe_float(r2) if is_sel else "",
                        "igd_plus": _safe_float(igd) if (is_sel and igd is not None) else "",
                    })
                if ep_data:
                    ars = [float(r.get("acc_ratio", 0)) for r in ep_data]
                    costs = [float(r.get("deploy_cost", 0)) for r in ep_data]
                    avg_costs = [float(r.get("avg_cost_per_accepted", 0)) for r in ep_data]
                    updated_csv_rows.append({
                        "experiment": exp_name, "ablation": abl, "test_type": "episode_test_mean",
                        "path": f"{len(ep_data)}_episodes_mean",
                        "acc_ratio": f"{np.mean(ars):.4f}", "deploy_cost": f"{np.mean(costs):.2f}",
                        "avg_cost_per_accepted": f"{np.mean(avg_costs):.2f}",
                        "hypervolume": hv_raw_val, "spacing": sp_val, "spread": spread_val,
                        "hv_normalized": _safe_float(hv), "r2": _safe_float(r2),
                        "igd_plus": _safe_float(igd) if igd is not None else "",
                    })

    if updated_csv_rows:
        update_ablation_summary_table(summary_table_csv, updated_csv_rows)

    out_path = results_root / f"{experiment_name}_mo_indicators.json"
    _atomic_write_json(out_path, {
        "experiment": experiment_name,
        "cost_bounds": {"min": float(cost_min), "max": float(cost_max)},
        "raw_ref_point": raw_ref.tolist(),
        "hv_ref_point_norm": hv_ref.tolist(),
        "empirical_ref_front_size": int(ref_front_raw.shape[0]),
        "empirical_ref_front_source": sorted(all_sets.keys()),
        "indicators": mo_results,
    })
    print(f"[MO] indicators saved -> {out_path}", flush=True)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def testcase_path(testcase_dir: Path, topology: str, allocation: str, difficulty: str) -> Path:
    return testcase_dir / f"{topology}_{allocation}_{difficulty}.json"


def test_out_dir(test_root: Path, topology: str, allocation: str, difficulty: str) -> Path:
    return test_root / topology / allocation / difficulty


def result_dir(results_root, topology, allocation, difficulty, suffix="") -> Path:
    return Path(results_root) / f"{topology}_{allocation}_{difficulty}{suffix}"


def checkpoint_path(ckpt_root, topology, allocation, difficulty, suffix="") -> Path:
    return Path(ckpt_root) / f"{topology}_{allocation}_{difficulty}{suffix}.pt"


def log_csv_path(log_root, topology, allocation, difficulty, suffix="") -> Path:
    return Path(log_root) / f"{topology}_{allocation}_{difficulty}{suffix}.csv"


def run_cmd(cmd: Sequence[str], dry_run: bool = False) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(list(cmd), check=False)
    if proc.returncode != 0:
        print(f"[ERROR] exit={proc.returncode}: {' '.join(cmd)}", flush=True)
    return proc.returncode


def train_one(train_file, checkpoint, csv_path, epochs, passes_per_file, warmup_epochs,
              pretrained_vgae, ablation, dry_run, *, seed: Optional[int] = None,
              deterministic: bool = False, val_dir: Optional[Path] = None,
              best_checkpoint: Optional[Path] = None, resume_train: bool = False,
              device: str = "auto") -> int:
    ensure_dir(checkpoint.parent)
    ensure_dir(csv_path.parent)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--train-file", str(train_file),
        "--checkpoint", str(checkpoint),
        "--csv", str(csv_path),
        "--epochs", str(epochs),
        "--passes-per-file", str(passes_per_file),
        "--warmup-epochs", str(warmup_epochs),
        "--ablation", ablation,
        "--device", device,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if deterministic:
        cmd.append("--deterministic")
    if val_dir is not None:
        cmd.extend(["--val-dir", str(val_dir)])
    if best_checkpoint is not None:
        cmd.extend(["--best-checkpoint", str(best_checkpoint)])
    if resume_train:
        cmd.append("--resume-train")
    if pretrained_vgae:
        cmd.extend(["--pretrained-vgae", pretrained_vgae])
    return run_cmd(cmd, dry_run=dry_run)


def resolve_testcase_dir(path: Path) -> Path:
    if path.is_dir():
        return path
    rel_candidate = PROJECT_ROOT / path
    if rel_candidate.is_dir():
        return rel_candidate
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for candidate in kaggle_input.glob("**/testcase"):
            if candidate.is_dir():
                return candidate
    return path


def generate_tests(topology_json, out_dir, num_episodes, seed, dry_run) -> int:
    ensure_dir(out_dir)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "data" / "generate_episodes.py"),
        "--topology", str(topology_json),
        "--num-episodes", str(num_episodes),
        "--infer-cfg",
        "--seed", str(seed),
        "--out-dir", str(out_dir),
    ]
    return run_cmd(cmd, dry_run=dry_run)


def evaluate_screen_single(checkpoint, episode_path, screen_out, ablation, dry_run, seed: Optional[int] = None, device: str = "auto") -> int:
    ensure_dir(screen_out.parent)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "evaluate.py"),
        "--ar-cost-only",
        "--checkpoint", str(checkpoint),
        "--single-episode", str(episode_path),
        "--ar-cost-out", str(screen_out),
        "--ablation", ablation,
        "--device", device,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    return run_cmd(cmd, dry_run=dry_run)


def evaluate_screen_all(checkpoint, data_dir, screen_out, ablation, dry_run, max_episodes: Optional[int] = None, seed: Optional[int] = None, device: str = "auto") -> int:
    ensure_dir(screen_out.parent)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "evaluate.py"),
        "--ar-cost-only",
        "--checkpoint", str(checkpoint),
        "--data-dir", str(data_dir),
        "--ar-cost-out", str(screen_out),
        "--ablation", ablation,
        "--device", device,
    ]
    if max_episodes is not None and max_episodes > 0:
        cmd.extend(["--max-episodes", str(max_episodes)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    return run_cmd(cmd, dry_run=dry_run)


def evaluate_screen_filelist(checkpoint, episode_files, screen_out, ablation, dry_run, seed: Optional[int] = None, device: str = "auto") -> int:
    if not episode_files:
        if not dry_run:
            ensure_dir(screen_out.parent)
            with open(screen_out, "w") as f:
                json.dump([], f)
        return 0
    ensure_dir(screen_out.parent)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "evaluate.py"),
        "--ar-cost-only",
        "--checkpoint", str(checkpoint),
        "--episodes", *[str(ep) for ep in episode_files],
        "--max-episodes", str(len(episode_files)),
        "--ar-cost-out", str(screen_out),
        "--ablation", ablation,
        "--device", device,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    return run_cmd(cmd, dry_run=dry_run)


def evaluate_pareto_single(checkpoint, episode_path, pareto_out, hv_out, pareto_points,
                            ablation, dry_run, ref_acc=None, ref_cost=None,
                            c_min=None, c_max=None, seed: Optional[int] = None, device: str = "auto") -> int:
    ensure_dir(pareto_out.parent)
    ensure_dir(hv_out.parent)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "evaluate.py"),
        "--pareto-scan",
        "--checkpoint", str(checkpoint),
        "--single-episode", str(episode_path),
        "--pareto-points", str(pareto_points),
        "--pareto-out", str(pareto_out),
        "--compute-hv",
        "--hv-out", str(hv_out),
        "--ablation", ablation,
        "--device", device,
    ]
    if ref_acc is not None and ref_cost is not None:
        cmd.extend(["--ref-acc", str(ref_acc), "--ref-cost", str(ref_cost)])
    if c_min is not None and c_max is not None:
        cmd.extend(["--c-min", str(c_min), "--c-max", str(c_max)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    return run_cmd(cmd, dry_run=dry_run)


def select_shared_pareto_episode(
    episode_files: List[Path],
    screen_json: Path,
    pareto_episode: str = "",
    pareto_episode_idx: int = 0,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    if not episode_files:
        print("[ERROR] no shared episode files available", flush=True)
        return None
    target_path: Optional[Path] = None
    if pareto_episode:
        for p in episode_files:
            if p.name == pareto_episode or str(p) == pareto_episode:
                target_path = p
                break
        if target_path is None:
            print(f"[ERROR] pareto episode '{pareto_episode}' not found in shared episodes", flush=True)
            return None
    else:
        idx = max(0, min(pareto_episode_idx, len(episode_files) - 1))
        target_path = episode_files[idx]

    if dry_run or not screen_json.is_file():
        return {
            "path": str(target_path),
            "acc_ratio": 0.0,
            "deploy_cost": 0.0,
            "avg_cost_per_accepted": 0.0,
        }

    with open(screen_json) as f:
        data = json.load(f)
    matched = None
    target_resolved = str(target_path.resolve())
    for row in data:
        p = row.get("path", "")
        if p == str(target_path) or Path(p).name == target_path.name or (p and str(Path(p).resolve()) == target_resolved):
            matched = row
            break
    if matched is None:
        matched = {"path": str(target_path), "acc_ratio": 0.0, "deploy_cost": 0.0, "avg_cost_per_accepted": 0.0}

    print(
        f"[SELECT] shared fixed Pareto episode: {target_path}  "
        f"AR={float(matched.get('acc_ratio', 0.0)):.4f}  cost={float(matched.get('deploy_cost', 0.0)):.2f}",
        flush=True,
    )
    return {
        "path": str(target_path),
        "acc_ratio": float(matched.get("acc_ratio", 0.0)),
        "deploy_cost": float(matched.get("deploy_cost", 0.0)),
        "avg_cost_per_accepted": float(matched.get("avg_cost_per_accepted", 0.0)),
        "accepted": matched.get("accepted"),
        "rejected": matched.get("rejected"),
        "w_accept": matched.get("w_accept"),
        "w_cost": matched.get("w_cost"),
    }


def select_best_ar_episode(screen_json: Path) -> Optional[Dict[str, Any]]:
    if not screen_json.is_file():
        print(f"[ERROR] screen file missing: {screen_json}", flush=True)
        return None
    with open(screen_json) as f:
        data = json.load(f)
    if not data:
        print(f"[ERROR] empty screen results: {screen_json}", flush=True)
        return None

    best_by_path: Dict[str, Dict[str, Any]] = {}
    for row in data:
        path = row.get("path", "")
        ar = float(row.get("acc_ratio", 0.0))
        cost = float(row.get("deploy_cost", float("inf")))
        avg_cost = float(row.get("avg_cost_per_accepted", 0.0))
        prev = best_by_path.get(path)
        if prev is None or ar > prev["acc_ratio"] or (
            ar == prev["acc_ratio"] and cost < prev["deploy_cost"]
        ):
            best_by_path[path] = {
                "path": path, "acc_ratio": ar, "deploy_cost": cost,
                "avg_cost_per_accepted": avg_cost,
                "accepted": row.get("accepted"), "rejected": row.get("rejected"),
                "w_accept": row.get("w_accept"), "w_cost": row.get("w_cost"),
            }

    ranked = sorted(best_by_path.values(), key=lambda r: (-r["acc_ratio"], r["deploy_cost"], r["path"]))
    return ranked[0]


def write_summary(summary_path, *, name, ablation, orig_screen_json, orig_hv_json,
                  screen_json, best, pareto_out, hv_out, mo_indicators=None):
    ensure_dir(summary_path.parent)

    orig_metrics = {}
    if orig_hv_json.is_file():
        with open(orig_hv_json) as f:
            orig_hv_data = json.load(f)
        orig_metrics = {
            "hypervolume": orig_hv_data.get("hypervolume"),
            "spacing": orig_hv_data.get("spacing"),
            "spread": orig_hv_data.get("spread"),
            "front_size": len(orig_hv_data.get("pareto_front", [])),
        }

    orig_screen_data = []
    if orig_screen_json.is_file():
        with open(orig_screen_json) as f:
            orig_screen_data = json.load(f)
    orig_entry = orig_screen_data[0] if orig_screen_data else {}

    hv_metrics = {}
    if hv_out.is_file():
        with open(hv_out) as f:
            hv = json.load(f)
        hv_metrics = {
            "hypervolume": hv.get("hypervolume"),
            "spacing": hv.get("spacing"),
            "spread": hv.get("spread"),
            "front_size": len(hv.get("pareto_front", [])),
        }
    if mo_indicators:
        hv_metrics.update(mo_indicators)

    selected_block = {
        "path": best.get("path"),
        "acc_ratio": best.get("acc_ratio"),
        "deploy_cost": best.get("deploy_cost"),
        "avg_cost_per_accepted": best.get("avg_cost_per_accepted"),
        "selection_policy": "fixed_shared_episode",
    }
    all_screen = []
    if screen_json.is_file():
        with open(screen_json) as f:
            all_screen = json.load(f)

    summary = {
        "experiment": name, "ablation": ablation,
        "selected_for_pareto": selected_block,
        "metrics": hv_metrics,
        "original_test": {
            "file": str(orig_screen_json),
            "path": orig_entry.get("path"),
            "acc_ratio": orig_entry.get("acc_ratio"),
            "deploy_cost": orig_entry.get("deploy_cost"),
            "metrics": orig_metrics,
        },
        "episode_tests": {
            "screening_file": str(screen_json),
            "num_entries": len(all_screen),
            "selected_for_pareto": dict(selected_block),
            "pareto_scan": str(pareto_out),
            "hv_result": str(hv_out),
            "metrics": hv_metrics,
        },
    }
    _atomic_write_json(summary_path, summary)
    print(f"[SUMMARY] saved -> {summary_path}", flush=True)


def update_ablation_summary_table(csv_path: Path, rows: List[Dict[str, Any]]):
    ensure_dir(csv_path.parent)
    fieldnames = [
        "experiment", "ablation", "test_type", "path",
        "acc_ratio", "deploy_cost", "avg_cost_per_accepted", "accepted", "rejected",
        "hypervolume", "spacing", "spread", "hv_normalized", "r2", "igd_plus",
    ]
    existing_rows = []
    if csv_path.is_file():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(r)

    def row_key(d):
        return (str(d.get("experiment", "")), str(d.get("ablation", "")),
                str(d.get("test_type", "")), str(d.get("path", "")))

    updated_keys = {row_key(r) for r in rows}
    merged = [r for r in existing_rows if row_key(r) not in updated_keys]
    merged.extend(rows)

    tmp_csv = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with open(tmp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in merged:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp_csv, csv_path)
    print(f"[TABLE] updated -> {csv_path}", flush=True)


def iter_experiments(topologies, allocations, difficulties):
    return [(t, a, d) for t in topologies for a in allocations for d in difficulties]


def run_experiment(
    topology, allocation, difficulty, *,
    testcase_dir, test_root, results_root, ckpt_root, log_root,
    epochs, passes_per_file, warmup_epochs, num_test_episodes, seed,
    pareto_points, pretrained_vgae, ablation,
    skip_train, skip_generate, skip_evaluate, dry_run,
    resume=True, ref_acc=None, ref_cost=None,
    c_min=None, c_max=None, pareto_episode="", pareto_episode_idx=0,
    test_seed: int = DEFAULT_SEED, multi_seed: bool = False,
    deterministic: bool = False, val_dir: Optional[Path] = None,
    use_best_ckpt: bool = False, resume_train: bool = False,
    device: str = "auto",
    shared_episode_files: Optional[List[Path]] = None,
) -> Tuple[bool, str, float, Dict[str, Any]]:
    base_instance = f"{topology}_{allocation}_{difficulty}"
    name = experiment_id_for(topology, allocation, difficulty, ablation, seed, multi_seed=multi_seed)
    tc = testcase_path(testcase_dir, topology, allocation, difficulty)

    if not tc.is_file():
        print(f"[ERROR] missing original testcase: {tc}", flush=True)
        return False, "failed", 0.0, {}

    ckpt = Path(ckpt_root) / f"{name}.pt"
    best_ckpt = Path(ckpt_root) / f"{name}_best.pt"
    csv_path = Path(log_root) / f"{name}.csv"
    out_test = shared_test_dir(test_root, topology, allocation, difficulty)
    res_dir = Path(results_root) / name

    orig_screen_out = res_dir / "original_test_ar_cost.json"
    orig_pareto_out = res_dir / "original_test_pareto_front.json"
    orig_hv_out = res_dir / "original_test_hv.json"
    screen_out = res_dir / "screen_ar_cost.json"
    pareto_out = res_dir / "pareto_front.json"
    hv_out = res_dir / "hv_result.json"
    summary_out = res_dir / "summary.json"
    manifest_out = res_dir / "manifest.json"
    summary_table_csv = Path(results_root) / "ablation_summary.csv"

    expected_ep_name = (
        Path(pareto_episode).name
        if pareto_episode
        else (
            shared_episode_files[max(0, min(pareto_episode_idx, len(shared_episode_files) - 1))].name
            if shared_episode_files
            else f"episode_{pareto_episode_idx:04d}.json"
        )
    )
    effective_cfg = {
        "protocol_version": "batch5_v1",
        "base_instance": base_instance, "topology": topology, "allocation": allocation,
        "difficulty": difficulty, "ablation": ablation, "seed": int(seed), "test_seed": int(test_seed),
        "epochs": int(epochs), "passes_per_file": int(passes_per_file), "warmup_epochs": int(warmup_epochs),
        "num_test_episodes": int(num_test_episodes), "pareto_points": int(pareto_points),
        "pareto_episode": expected_ep_name, "pretrained_vgae": pretrained_vgae,
        "deterministic": bool(deterministic), "use_best_ckpt": bool(use_best_ckpt and val_dir is not None),
        "val_enabled": bool(val_dir is not None),
    }
    cfg_fingerprint = _compute_config_fingerprint(effective_cfg)

    if resume and manifest_out.is_file() and summary_out.is_file() and hv_out.is_file() and pareto_out.is_file():
        resume_ok = False
        try:
            with open(manifest_out) as mf:
                mdata = json.load(mf)
            resume_ok = (
                mdata.get("status") == "completed"
                and mdata.get("config_fingerprint") == cfg_fingerprint
                and screen_out.is_file()
            )
        except (OSError, json.JSONDecodeError):
            resume_ok = False
        if resume_ok:
            print(f"\n[RESUME SKIP] {name} completed (fingerprint={cfg_fingerprint}).", flush=True)
            with open(manifest_out) as mf:
                mdata = json.load(mf)
            return True, "completed_resumed", float(mdata.get("duration_sec", 0.0)), mdata.get("metrics", {})
        print(f"\n[RESUME INVALIDATED] {name} missing/mismatched manifest (expected fp={cfg_fingerprint}); re-running without deleting checkpoints.", flush=True)

    print(f"\n{'='*60}\n EXPERIMENT: {name}  [ablation={ablation} seed={seed}]\n{'='*60}", flush=True)

    for d in (ckpt_root, log_root, res_dir):
        ensure_dir(d)

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest_data = {
        "experiment_id": name, "config_fingerprint": cfg_fingerprint,
        "effective_config": effective_cfg, "environment": _collect_env_info(),
        "checkpoint": str(ckpt), "best_checkpoint": str(best_ckpt) if val_dir is not None else None,
        "train_file": str(tc), "started_at": started_at, "status": "running",
    }
    if not dry_run:
        _atomic_write_json(manifest_out, manifest_data)

    t0 = time.perf_counter()
    ok = True

    should_train = not skip_train
    if resume and ckpt.is_file() and not skip_train:
        try:
            _ck = torch.load(ckpt, map_location="cpu")
            ck_epoch = int(_ck.get("epoch", epochs))
            ck_seed = (_ck.get("env_meta") or {}).get("seed")
            if _ck.get("ablation_mode") in (None, ablation) and ck_seed in (None, seed) and ck_epoch >= epochs:
                print(f"[RESUME train] reusing completed checkpoint: {ckpt}", flush=True)
                should_train = False
        except Exception:
            pass
    if should_train:
        rc = train_one(tc, ckpt, csv_path, epochs, passes_per_file, warmup_epochs,
                       pretrained_vgae, ablation, dry_run, seed=seed,
                       deterministic=deterministic, val_dir=val_dir,
                       best_checkpoint=best_ckpt if val_dir is not None else None,
                       resume_train=(resume and resume_train), device=device)
        if rc != 0:
            print(f"[FAIL] train: {name}", flush=True)
            ok = False
    elif skip_train:
        print(f"[SKIP train] {name}", flush=True)

    if use_best_ckpt and val_dir is not None:
        if dry_run or best_ckpt.is_file():
            eval_ckpt = best_ckpt
        else:
            print(f"[WARN] {name}: best checkpoint {best_ckpt} missing; falling back to {ckpt}", flush=True)
            eval_ckpt = ckpt
    else:
        eval_ckpt = ckpt

    episode_files = shared_episode_files or _list_episode_files(out_test)[:num_test_episodes]
    dry_run_virtual_eps = dry_run and not episode_files
    if dry_run_virtual_eps:
        episode_files = [out_test / f"episode_{i:04d}.json" for i in range(num_test_episodes)]

    if ok and not skip_evaluate:
        if not dry_run and not eval_ckpt.is_file():
            print(f"[FAIL] no checkpoint: {eval_ckpt}", flush=True)
            ok = False
        else:
            rc_os = evaluate_screen_single(eval_ckpt, tc, orig_screen_out, ablation, dry_run, seed=seed, device=device)
            if rc_os != 0:
                ok = False
            if ok:
                rc_op = evaluate_pareto_single(eval_ckpt, tc, orig_pareto_out, orig_hv_out,
                                               pareto_points, ablation, dry_run, ref_acc, ref_cost, seed=seed, device=device)
                if rc_op != 0:
                    ok = False

            if ok:
                print(f"\n[EPISODE TESTS] screening {len(episode_files)} shared episodes", flush=True)
                use_dir_screen = bool(episode_files) and (
                    dry_run_virtual_eps
                    or episode_files == _list_episode_files(out_test)[:len(episode_files)]
                )
                if use_dir_screen:
                    rc = evaluate_screen_all(eval_ckpt, out_test, screen_out, ablation, dry_run,
                                             max_episodes=len(episode_files), seed=seed, device=device)
                else:
                    rc = evaluate_screen_filelist(eval_ckpt, episode_files, screen_out, ablation, dry_run, seed=seed, device=device)
                if rc != 0:
                    ok = False
                else:
                    best = select_shared_pareto_episode(
                        episode_files,
                        screen_out,
                        pareto_episode=pareto_episode,
                        pareto_episode_idx=pareto_episode_idx,
                        dry_run=dry_run,
                    )

                    if best is None:
                        ok = False
                    else:
                        episode_path = Path(best["path"])
                        rc = evaluate_pareto_single(eval_ckpt, episode_path, pareto_out, hv_out,
                                                    pareto_points, ablation, dry_run, ref_acc, ref_cost,
                                                    c_min=c_min, c_max=c_max, seed=seed, device=device)
                        if rc != 0:
                            ok = False
                        elif not dry_run:
                            mo_ind = {}
                            if pareto_out.is_file() and c_min is not None and c_max is not None:
                                mo_ind = compute_mo_indicators(
                                    pareto_out, (float(c_min), float(c_max)), np.array([0.0, 0.0])
                                )
                            elif pareto_out.is_file():
                                mo_ind = {"normalization_scope": "pending_cross_mo"}

                            write_summary(summary_out, name=name, ablation=ablation,
                                          orig_screen_json=orig_screen_out, orig_hv_json=orig_hv_out,
                                          screen_json=screen_out, best=best,
                                          pareto_out=pareto_out, hv_out=hv_out,
                                          mo_indicators=mo_ind)

                            rows_to_record = _build_csv_rows(
                                name, ablation, tc, orig_screen_out, orig_hv_out,
                                screen_out, hv_out, best, ep_s_mo=mo_ind)
                            update_ablation_summary_table(summary_table_csv, rows_to_record)
    elif skip_evaluate:
        print(f"[SKIP evaluate] {name}", flush=True)

    elapsed = time.perf_counter() - t0
    status = "OK" if ok else "FAILED"
    final_state = "completed" if ok else "failed"
    exp_metrics = {}
    if ok and not dry_run and summary_out.is_file():
        with open(summary_out) as sf:
            s_loaded = json.load(sf)
        ep_data = []
        if screen_out.is_file():
            with open(screen_out) as scf:
                ep_data = json.load(scf)
        ep_m = s_loaded.get("metrics") or {}
        exp_metrics = {
            "mean_ar": float(np.mean([float(r["acc_ratio"]) for r in ep_data])) if ep_data else None,
            "mean_deploy_cost": float(np.mean([float(r["deploy_cost"]) for r in ep_data])) if ep_data else None,
            "hypervolume": _safe_float(ep_m.get("hypervolume")),
            "hv_normalized": _safe_float(ep_m.get("hv_normalized")),
            "r2": _safe_float(ep_m.get("r2")),
            "igd_plus": _safe_float(ep_m.get("igd_plus")),
        }
    if not dry_run:
        manifest_data.update({
            "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "duration_sec": round(elapsed, 2),
            "status": final_state,
            "eval_checkpoint": str(eval_ckpt),
            "test_episodes": [str(p) for p in episode_files],
            "pareto_episode": expected_ep_name,
            "metrics": exp_metrics,
        })
        _atomic_write_json(manifest_out, manifest_data)
    print(f"[{status}] {name}  ({elapsed:.1f}s)", flush=True)
    return ok, final_state, elapsed, exp_metrics


def _build_csv_rows(name, ablation, tc, orig_screen_out, orig_hv_out,
                    screen_out, hv_out, best, ep_s_mo=None):
    rows = []

    orig_hv_val = orig_sp_val = orig_spread_val = ""
    if orig_hv_out.is_file():
        with open(orig_hv_out) as f:
            d = json.load(f)
        orig_hv_val = d.get("hypervolume", "")
        orig_sp_val = d.get("spacing", "")
        orig_spread_val = d.get("spread", d.get("delta", ""))

    if orig_screen_out.is_file():
        with open(orig_screen_out) as f:
            od = json.load(f)
        orig_row = od[0] if od else {}
        rows.append({
            "experiment": name, "ablation": ablation, "test_type": "original_test",
            "path": orig_row.get("path", str(tc)),
            "acc_ratio": orig_row.get("acc_ratio", ""),
            "deploy_cost": orig_row.get("deploy_cost", ""),
            "avg_cost_per_accepted": orig_row.get("avg_cost_per_accepted", ""),
            "accepted": orig_row.get("accepted", ""),
            "rejected": orig_row.get("rejected", ""),
            "hypervolume": orig_hv_val, "spacing": orig_sp_val, "spread": orig_spread_val,
        })

    best_hv_val = best_sp_val = best_spread_val = ""
    best_hv_norm = best_r2 = best_igd = ""
    if hv_out.is_file():
        with open(hv_out) as f:
            hd = json.load(f)
        best_hv_val = hd.get("hypervolume", "")
        best_sp_val = hd.get("spacing", "")
        best_spread_val = hd.get("spread", hd.get("delta", ""))
    if ep_s_mo:
        best_hv_norm = ep_s_mo.get("hv_normalized", "")
        best_r2 = ep_s_mo.get("r2", "")
        best_igd = ep_s_mo.get("igd_plus", "")

    if screen_out.is_file():
        with open(screen_out) as f:
            ep_data = json.load(f)
        best_path = best.get("path", "")
        best_name = Path(best_path).name if best_path else ""
        for r in ep_data:
            rp = r.get("path", "")
            is_best = bool(best_path) and (rp == best_path or (rp and Path(rp).name == best_name))
            rows.append({
                "experiment": name, "ablation": ablation, "test_type": "episode_test",
                "path": rp,
                "acc_ratio": r.get("acc_ratio", ""), "deploy_cost": r.get("deploy_cost", ""),
                "avg_cost_per_accepted": r.get("avg_cost_per_accepted", ""),
                "accepted": r.get("accepted", ""), "rejected": r.get("rejected", ""),
                "hypervolume": best_hv_val if is_best else "",
                "spacing": best_sp_val if is_best else "",
                "spread": best_spread_val if is_best else "",
                "hv_normalized": best_hv_norm if is_best else "",
                "r2": best_r2 if is_best else "",
                "igd_plus": best_igd if is_best else "",
            })
        if ep_data:
            ars = [float(r.get("acc_ratio", 0)) for r in ep_data]
            costs = [float(r.get("deploy_cost", 0)) for r in ep_data]
            avg_costs = [float(r.get("avg_cost_per_accepted", 0)) for r in ep_data]
            rows.append({
                "experiment": name, "ablation": ablation, "test_type": "episode_test_mean",
                "path": f"{len(ep_data)}_episodes_mean",
                "acc_ratio": f"{np.mean(ars):.4f}", "deploy_cost": f"{np.mean(costs):.2f}",
                "avg_cost_per_accepted": f"{np.mean(avg_costs):.2f}",
                "hypervolume": best_hv_val, "spacing": best_sp_val, "spread": best_spread_val,
                "hv_normalized": best_hv_norm, "r2": best_r2, "igd_plus": best_igd,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    is_kaggle = Path("/kaggle/working").is_dir() and os.access("/kaggle/working", os.W_OK)
    default_root = Path("/kaggle/working") if is_kaggle else Path(".")
    default_test_root = default_root / "data" / "test" if is_kaggle else Path("data/test")
    default_val_root = default_root / "data" / "val" if is_kaggle else Path("data/val")

    parser.add_argument("--testcase-dir", type=str, default="data/testcase")
    parser.add_argument("--test-root", type=str, default=str(default_test_root))
    parser.add_argument("--val-root", type=str, default=str(default_val_root))
    parser.add_argument("--results-root", type=str, default=str(default_root / "results"))
    parser.add_argument("--ckpt-root", type=str, default=str(default_root / "checkpoints"))
    parser.add_argument("--log-root", type=str, default=str(default_root / "logs"))

    parser.add_argument("--topology", type=str, nargs="*", default=None, choices=list(TOPOLOGIES))
    parser.add_argument("--allocation", type=str, nargs="*", default=None, choices=list(ALLOCATIONS))
    parser.add_argument("--difficulty", type=str, nargs="*", default=None, choices=list(DIFFICULTIES))

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--passes-per-file", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--num-test-episodes", type=int, default=NUM_TEST_EPISODES)
    parser.add_argument("--num-val-episodes", type=int, default=0)
    parser.add_argument("--use-best-ckpt", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--test-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--pareto-points", type=int, default=11)
    parser.add_argument("--pretrained-vgae", type=str, default="")
    parser.add_argument("--ablation", type=str, default=DEFAULT_ABLATION, choices=list(ABLATIONS))

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-train", action="store_true")
    parser.add_argument("--ref-acc", type=float, default=None)
    parser.add_argument("--ref-cost", type=float, default=None)
    parser.add_argument("--c-min", type=float, default=None)
    parser.add_argument("--c-max", type=float, default=None)
    parser.add_argument("--pareto-episode", type=str, default="",
                        help="Fixed shared episode filename for Pareto scan (overrides --pareto-episode-idx)")
    parser.add_argument("--pareto-episode-idx", type=int, default=0,
                        help="Fixed shared episode index for Pareto scan across all ablations")
    parser.add_argument("--compute-cross-mo", action="store_true",
                        help="After all experiments, compute cross-ablation MO indicators (HV, R2, IGD+)")
    parser.add_argument("--cross-mo-ablations", type=str, nargs="*", default=None,
                        help="Ablations to include in cross-MO (default: all ABLATIONS)")
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--check-env", action="store_true")

    args = parser.parse_args()
    if args.check_env:
        env_info = _collect_env_info()
        deps = {}
        for mod in ("torch", "torch_geometric", "numpy", "networkx", "matplotlib", "scipy"):
            try:
                m = __import__(mod)
                deps[mod] = getattr(m, "__version__", "installed")
            except ImportError:
                deps[mod] = "missing"
        print(json.dumps({"environment": env_info, "dependencies": deps}, indent=2))
        return

    seeds = list(args.seeds) if args.seeds else [int(args.seed)]
    multi_seed = len(seeds) > 1

    topologies = tuple(args.topology) if args.topology else TOPOLOGIES
    allocations = tuple(args.allocation) if args.allocation else ALLOCATIONS
    difficulties = tuple(args.difficulty) if args.difficulty else DIFFICULTIES

    testcase_dir = resolve_testcase_dir(Path(args.testcase_dir))
    test_root = Path(args.test_root)
    val_root = Path(args.val_root)
    results_root = Path(args.results_root)
    ckpt_root = Path(args.ckpt_root)
    log_root = Path(args.log_root)

    for d in (test_root, results_root, ckpt_root, log_root):
        ensure_dir(d)

    experiments = iter_experiments(topologies, allocations, difficulties)
    if args.aggregate_only:
        args.aggregate = True
    else:
        print(f"Scheduled experiments: {len(experiments) * len(seeds)} ({len(experiments)} instances x {len(seeds)} seeds)", flush=True)
        print(f"Ablation: {args.ablation}  Seeds: {seeds}  TestSeed: {args.test_seed}  Resume: {args.resume}", flush=True)

    shared_episodes: Dict[Tuple[str, str, str], List[Path]] = {}
    if not args.aggregate_only and not args.skip_generate:
        print("\n[SHARED EPISODES] Ensuring shared test episodes per topology...", flush=True)
        for topology, allocation, difficulty in experiments:
            key = (topology, allocation, difficulty)
            if key not in shared_episodes:
                eps = ensure_shared_test_episodes(
                    topology, allocation, difficulty,
                    testcase_dir, test_root,
                    args.num_test_episodes, args.test_seed, args.dry_run,
                )
                shared_episodes[key] = eps
                if args.num_val_episodes > 0:
                    ensure_shared_test_episodes(
                        topology, allocation, difficulty,
                        testcase_dir, val_root,
                        args.num_val_episodes, args.test_seed + 10_000, args.dry_run,
                    )
                names = [p.name for p in eps]
                print(f"  {topology}/{allocation}/{difficulty}: {names}", flush=True)
    elif not args.aggregate_only:
        for topology, allocation, difficulty in experiments:
            key = (topology, allocation, difficulty)
            d = shared_test_dir(test_root, topology, allocation, difficulty)
            shared_episodes[key] = _list_episode_files(d)[:args.num_test_episodes]
            print(f"[SKIP generate] using existing: {d} ({len(shared_episodes[key])} files)", flush=True)

    succeeded, failed = [], []
    status_path = results_root / "experiments_status.json"
    status_board: Dict[str, Any] = {}
    if status_path.is_file():
        try:
            with open(status_path) as sf:
                status_board = json.load(sf)
        except (OSError, json.JSONDecodeError):
            status_board = {}
    session_t0 = time.perf_counter()
    observed_durations: List[float] = []
    budget_sec = args.time_budget_hours * 3600.0 if args.time_budget_hours else None

    if not args.aggregate_only:
        for topology, allocation, difficulty in experiments:
            tc = testcase_path(testcase_dir, topology, allocation, difficulty)
            key = (topology, allocation, difficulty)
            episode_files = shared_episodes.get(key, [])
            val_dir = shared_test_dir(val_root, topology, allocation, difficulty) if args.num_val_episodes > 0 else None
            for seed in seeds:
                exp_id = experiment_id_for(topology, allocation, difficulty, args.ablation, seed, multi_seed=multi_seed)
                if budget_sec is not None and observed_durations:
                    avg_d = float(np.mean(observed_durations))
                    if (time.perf_counter() - session_t0) + avg_d > budget_sec:
                        status_board[exp_id] = {"status": "deferred_budget", "ablation": args.ablation, "seed": seed}
                        if not args.dry_run:
                            _atomic_write_json(status_path, status_board)
                        print(f"[BUDGET STOP] Deferring remaining experiments before {exp_id}.", flush=True)
                        break
                if not tc.is_file():
                    print(f"[ERROR] missing: {tc}", flush=True)
                    failed.append(exp_id)
                    status_board[exp_id] = {"status": "failed", "reason": "missing_testcase"}
                    if args.stop_on_error:
                        break
                    continue

                ok, state_str, dur, _ = run_experiment(
                    topology, allocation, difficulty,
                    testcase_dir=testcase_dir, test_root=test_root,
                    results_root=results_root, ckpt_root=ckpt_root, log_root=log_root,
                    epochs=args.epochs, passes_per_file=args.passes_per_file,
                    warmup_epochs=args.warmup_epochs, num_test_episodes=args.num_test_episodes,
                    seed=seed, pareto_points=args.pareto_points,
                    pretrained_vgae=args.pretrained_vgae, ablation=args.ablation,
                    skip_train=args.skip_train, skip_generate=True,
                    skip_evaluate=args.skip_evaluate, dry_run=args.dry_run,
                    resume=args.resume, ref_acc=args.ref_acc, ref_cost=args.ref_cost,
                    c_min=args.c_min, c_max=args.c_max,
                    pareto_episode=args.pareto_episode,
                    pareto_episode_idx=args.pareto_episode_idx,
                    test_seed=args.test_seed, multi_seed=multi_seed,
                    deterministic=args.deterministic, val_dir=val_dir,
                    use_best_ckpt=args.use_best_ckpt, resume_train=args.resume_train,
                    device=args.device,
                    shared_episode_files=episode_files,
                )
                (succeeded if ok else failed).append(exp_id)
                if state_str == "completed" and dur > 0:
                    observed_durations.append(dur)
                status_board[exp_id] = {"status": state_str, "ablation": args.ablation, "seed": seed, "duration_sec": round(dur, 2)}
                if not args.dry_run:
                    _atomic_write_json(status_path, status_board)
                if not ok and args.stop_on_error:
                    print("[ABORT] --stop-on-error set", flush=True)
                    break
            if (failed and args.stop_on_error) or (budget_sec is not None and observed_durations and (time.perf_counter() - session_t0) + float(np.mean(observed_durations)) > budget_sec):
                break

    if args.compute_cross_mo and not args.dry_run:
        cross_ablations = args.cross_mo_ablations or list(ABLATIONS)
        print(f"\n[CROSS-MO] Computing cross-ablation indicators for: {cross_ablations}", flush=True)
        for topology, allocation, difficulty in experiments:
            exp_base = f"{topology}_{allocation}_{difficulty}"
            compute_and_save_mo_indicators_for_experiment(
                exp_base, cross_ablations, results_root,
                c_min_override=args.c_min, c_max_override=args.c_max,
                ref_acc_override=args.ref_acc, ref_cost_override=args.ref_cost,
                seeds=seeds, multi_seed=multi_seed,
            )

    if args.aggregate and not args.dry_run:
        from utils.stats_agg import aggregate_experiment_records
        manifest_records = []
        for mpath in sorted(results_root.glob("*/manifest.json")):
            with open(mpath) as mf:
                m = json.load(mf)
            ec = m.get("effective_config") or {}
            manifest_records.append({
                "experiment_id": m.get("experiment_id"),
                "base_instance": ec.get("base_instance"),
                "topology": ec.get("topology"),
                "allocation": ec.get("allocation"),
                "difficulty": ec.get("difficulty"),
                "ablation": ec.get("ablation"),
                "seed": ec.get("seed"),
                "status": m.get("status"),
                "metrics": m.get("metrics") or {},
            })
        agg = aggregate_experiment_records(
            manifest_records,
            metrics=("mean_ar", "mean_deploy_cost", "hypervolume", "hv_normalized", "r2", "igd_plus"),
        )
        _atomic_write_json(results_root / "aggregated_stats.json", agg)
        print(f"[AGG] saved -> {results_root / 'aggregated_stats.json'}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Done.  OK={len(succeeded)}  FAIL={len(failed)}", flush=True)
    if succeeded:
        print("Succeeded:", ", ".join(succeeded), flush=True)
    if failed:
        print("Failed   :", ", ".join(failed), flush=True)


if __name__ == "__main__":
    main()