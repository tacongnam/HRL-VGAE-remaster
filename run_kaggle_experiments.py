"""
Kaggle experiment runner for HRL-VGAE.

Pipeline per testcase (independent experiment):
  1. Train on data/testcase/{topology}_{allocation}_{difficulty}.json
  2. Reuse shared test episodes per topology (generated once, shared across ablations)
  3. Screen all 10 tests with checkpoint weights (--ar-cost-only) -> AR + cost
  4. Select the episode with the highest AR
  5. Full Pareto scan ONLY on that episode -> HV / R2 / IGD+ / Spacing / Spread

Topology : nsf, cogent, conus
Allocation: centers, uniform, rural, urban
Difficulty: easy, normal, hard
"""

from __future__ import annotations

import numpy as np
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TOPOLOGIES = ("nsf", "cogent", "conus")
ALLOCATIONS = ("centers", "uniform", "rural", "urban")
DIFFICULTIES = ("easy", "normal", "hard")

ABLATIONS = ("DefaultBaseline", "Random-Fit", "GreedyFIFO", "GreedyLowLevel", "HRL-VGAE", "HRL-noVGAE")
DEFAULT_ABLATION = "HRL-VGAE"

NUM_TEST_EPISODES = 10
DEFAULT_SEED = 42


def ablation_suffix(ablation: str) -> str:
    return "" if ablation == DEFAULT_ABLATION else f"_{ablation}"


# ---------------------------------------------------------------------------
# Shared test episode management (Section 1 of requirements)
# ---------------------------------------------------------------------------

def shared_test_dir(test_root: Path, topology: str, allocation: str, difficulty: str) -> Path:
    return test_root / topology / allocation / difficulty


def _list_episode_files(d: Path) -> List[Path]:
    return sorted(d.glob("episode_*.json")) if d.is_dir() else []


def _episode_id(p: Path) -> int:
    stem = p.stem  # episode_0003 -> 0003
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
        # Generate all
        rc = generate_tests(tc, out_dir, num_episodes, seed, dry_run)
        if rc != 0:
            print(f"[FAIL] generate episodes: {topology}_{allocation}_{difficulty}", flush=True)
        existing = _list_episode_files(out_dir)
        return existing[:num_episodes]

    # Partial: fill missing IDs
    present_ids = {_episode_id(p) for p in existing}
    missing_count = num_episodes - len(existing)
    # Determine which IDs [0..num_episodes-1] are absent
    missing_ids = [i for i in range(num_episodes) if i not in present_ids][:missing_count]
    if missing_ids:
        print(
            f"[EPISODES] {len(existing)}/{num_episodes} present; generating missing IDs {missing_ids}",
            flush=True,
        )
        for ep_id in missing_ids:
            ep_seed = seed + ep_id  # deterministic per-ID seed offset
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
    # generate_episodes.py with num-episodes=1; rename output deterministically
    tmp_dir = out_dir / f"_tmp_ep{ep_id}"
    ensure_dir(tmp_dir)
    cmd = [
        sys.executable, "data/generate_episodes.py",
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


# ---------------------------------------------------------------------------
# Multi-objective indicators (Section 2 of requirements)
# ---------------------------------------------------------------------------

def _load_pareto_indicators():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.pareto_indicators import (
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
        "hv_normalized": float(hv),
        "r2": float(r2),
        "pareto_set_size": int(pareto_set.shape[0]),
        "cost_min_used": float(cost_min),
        "cost_max_used": float(cost_max),
    }
    if igd_plus is not None:
        result["igd_plus"] = float(igd_plus)
    return result


def _collect_all_pareto_sets_for_instance(
    experiment_name: str,
    ablations: List[str],
    results_root: Path,
) -> Dict[str, np.ndarray]:
    (build_pareto_set_from_scan, *_) = _load_pareto_indicators()
    sets = {}
    for abl in ablations:
        suffix = ablation_suffix(abl)
        name = f"{experiment_name}{suffix}"
        pf = results_root / name / "pareto_front.json"
        if pf.is_file():
            with open(pf) as f:
                scan = json.load(f)
            ps = build_pareto_set_from_scan(scan)
            if ps.shape[0] > 0:
                sets[abl] = ps
    return sets


def _infer_cost_bounds(results_root: Path, experiment_name: str, ablations: List[str]) -> Tuple[float, float]:
    (build_pareto_set_from_scan, *_) = _load_pareto_indicators()
    all_costs = []
    for abl in ablations:
        suffix = ablation_suffix(abl)
        name = f"{experiment_name}{suffix}"
        for fname in ("pareto_front.json", "screen_ar_cost.json"):
            p = results_root / name / fname
            if p.is_file():
                with open(p) as f:
                    data = json.load(f)
                all_costs.extend(float(r["deploy_cost"]) for r in data if "deploy_cost" in r)
    if not all_costs:
        return 0.0, 1.0
    return min(all_costs), max(all_costs)


def compute_and_save_mo_indicators_for_experiment(
    experiment_name: str,
    ablations: List[str],
    results_root: Path,
    n_weights: int = 101,
):
    (
        build_pareto_set_from_scan,
        normalize_pareto_objectives,
        compute_hv_normalized,
        compute_r2,
        compute_igd_plus,
        build_empirical_reference_front,
    ) = _load_pareto_indicators()

    cost_min, cost_max = _infer_cost_bounds(results_root, experiment_name, ablations)
    hv_ref = np.array([1.1, 1.1])
    ideal = np.array([0.0, 0.0])

    all_sets = _collect_all_pareto_sets_for_instance(experiment_name, ablations, results_root)
    if not all_sets:
        return

    ref_front_raw = build_empirical_reference_front(list(all_sets.values()))
    ref_front_norm = (
        normalize_pareto_objectives(ref_front_raw, cost_min, cost_max)
        if ref_front_raw.shape[0] > 0
        else np.empty((0, 2))
    )

    mo_results = {}
    for abl, ps in all_sets.items():
        ps_norm = normalize_pareto_objectives(ps, cost_min, cost_max)
        hv = compute_hv_normalized(ps_norm, hv_ref)
        r2 = compute_r2(ps_norm, ideal, n_weights)
        igd = compute_igd_plus(ps_norm, ref_front_norm) if ref_front_norm.shape[0] > 0 else None
        mo_results[abl] = {
            "hv_normalized": float(hv),
            "r2": float(r2),
            "pareto_set_size": int(ps.shape[0]),
        }
        if igd is not None:
            mo_results[abl]["igd_plus"] = float(igd)

    out_path = results_root / f"{experiment_name}_mo_indicators.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment": experiment_name,
            "cost_bounds": {"min": float(cost_min), "max": float(cost_max)},
            "empirical_ref_front_size": int(ref_front_raw.shape[0]),
            "indicators": mo_results,
        }, f, indent=2)
    print(f"[MO] indicators saved -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Path helpers (unchanged from original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Command runners
# ---------------------------------------------------------------------------

def run_cmd(cmd: Sequence[str], dry_run: bool = False) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(list(cmd), check=False)
    if proc.returncode != 0:
        print(f"[ERROR] exit={proc.returncode}: {' '.join(cmd)}", flush=True)
    return proc.returncode


def train_one(train_file, checkpoint, csv_path, epochs, passes_per_file, warmup_epochs,
              pretrained_vgae, ablation, dry_run) -> int:
    ensure_dir(checkpoint.parent)
    ensure_dir(csv_path.parent)
    cmd = [
        sys.executable, "train.py",
        "--train-file", str(train_file),
        "--checkpoint", str(checkpoint),
        "--csv", str(csv_path),
        "--epochs", str(epochs),
        "--passes-per-file", str(passes_per_file),
        "--warmup-epochs", str(warmup_epochs),
        "--ablation", ablation,
    ]
    if pretrained_vgae:
        cmd.extend(["--pretrained-vgae", pretrained_vgae])
    return run_cmd(cmd, dry_run=dry_run)


def resolve_testcase_dir(path: Path) -> Path:
    if path.is_dir():
        return path
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for candidate in kaggle_input.glob("**/testcase"):
            if candidate.is_dir():
                return candidate
    return path


def generate_tests(topology_json, out_dir, num_episodes, seed, dry_run) -> int:
    ensure_dir(out_dir)
    cmd = [
        sys.executable, "data/generate_episodes.py",
        "--topology", str(topology_json),
        "--num-episodes", str(num_episodes),
        "--infer-cfg",
        "--seed", str(seed),
        "--out-dir", str(out_dir),
    ]
    return run_cmd(cmd, dry_run=dry_run)


def evaluate_screen_single(checkpoint, episode_path, screen_out, ablation, dry_run) -> int:
    ensure_dir(screen_out.parent)
    cmd = [
        sys.executable, "evaluate.py",
        "--ar-cost-only",
        "--checkpoint", str(checkpoint),
        "--single-episode", str(episode_path),
        "--ar-cost-out", str(screen_out),
        "--ablation", ablation,
    ]
    return run_cmd(cmd, dry_run=dry_run)


def evaluate_screen_all(checkpoint, data_dir, screen_out, ablation, dry_run) -> int:
    ensure_dir(screen_out.parent)
    cmd = [
        sys.executable, "evaluate.py",
        "--ar-cost-only",
        "--checkpoint", str(checkpoint),
        "--data-dir", str(data_dir),
        "--ar-cost-out", str(screen_out),
        "--ablation", ablation,
    ]
    return run_cmd(cmd, dry_run=dry_run)


def evaluate_screen_filelist(checkpoint, episode_files, screen_out, ablation, dry_run) -> int:
    ensure_dir(screen_out.parent)
    # Write a temp file list and pass via data-dir of a temp symlink dir,
    # or evaluate each file individually and merge results.
    results = []
    for ep in episode_files:
        tmp_out = screen_out.parent / f"_tmp_screen_{ep.stem}.json"
        rc = evaluate_screen_single(checkpoint, ep, tmp_out, ablation, dry_run)
        if rc == 0 and not dry_run and tmp_out.is_file():
            with open(tmp_out) as f:
                results.extend(json.load(f))
            tmp_out.unlink()
    if not dry_run:
        with open(screen_out, "w") as f:
            json.dump(results, f, indent=2)
    return 0


def evaluate_pareto_single(checkpoint, episode_path, pareto_out, hv_out, pareto_points,
                            ablation, dry_run, ref_acc=None, ref_cost=None) -> int:
    ensure_dir(pareto_out.parent)
    ensure_dir(hv_out.parent)
    cmd = [
        sys.executable, "evaluate.py",
        "--pareto-scan",
        "--checkpoint", str(checkpoint),
        "--single-episode", str(episode_path),
        "--pareto-points", str(pareto_points),
        "--pareto-out", str(pareto_out),
        "--compute-hv",
        "--hv-out", str(hv_out),
        "--ablation", ablation,
    ]
    if ref_acc is not None and ref_cost is not None:
        cmd.extend(["--ref-acc", str(ref_acc), "--ref-cost", str(ref_cost)])
    return run_cmd(cmd, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Select best-AR episode
# ---------------------------------------------------------------------------

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
    best = ranked[0]
    print(
        f"[SELECT] best AR episode: {best['path']}  "
        f"AR={best['acc_ratio']:.4f}  cost={best['deploy_cost']:.2f}",
        flush=True,
    )
    for i, r in enumerate(ranked, 1):
        mark = " <-- PARETO" if i == 1 else ""
        print(
            f"  {i:2d}. AR={r['acc_ratio']:.4f}  cost={r['deploy_cost']:.2f}  "
            f"{Path(r['path']).name}{mark}",
            flush=True,
        )
    return best


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

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

    all_screen = []
    if screen_json.is_file():
        with open(screen_json) as f:
            all_screen = json.load(f)

    summary = {
        "experiment": name, "ablation": ablation,
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
            "selected_for_pareto": {
                "path": best.get("path"),
                "acc_ratio": best.get("acc_ratio"),
                "deploy_cost": best.get("deploy_cost"),
            },
            "pareto_scan": str(pareto_out),
            "hv_result": str(hv_out),
            "metrics": hv_metrics,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
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

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in merged:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[TABLE] updated -> {csv_path}", flush=True)


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

def iter_experiments(topologies, allocations, difficulties):
    return [(t, a, d) for t in topologies for a in allocations for d in difficulties]


def run_experiment(
    topology, allocation, difficulty, *,
    testcase_dir, test_root, results_root, ckpt_root, log_root,
    epochs, passes_per_file, warmup_epochs, num_test_episodes, seed,
    pareto_points, pretrained_vgae, ablation,
    skip_train, skip_generate, skip_evaluate, dry_run,
    resume=True, ref_acc=None, ref_cost=None,
    shared_episode_files: Optional[List[Path]] = None,
) -> bool:
    suffix = ablation_suffix(ablation)
    name = f"{topology}_{allocation}_{difficulty}{suffix}"
    tc = testcase_path(testcase_dir, topology, allocation, difficulty)

    if not tc.is_file():
        print(f"[ERROR] missing original testcase: {tc}", flush=True)
        return False

    ckpt = checkpoint_path(ckpt_root, topology, allocation, difficulty, suffix)
    csv_path = log_csv_path(log_root, topology, allocation, difficulty, suffix)
    # Use shared episode dir (no suffix — same for all ablations)
    out_test = shared_test_dir(test_root, topology, allocation, difficulty)
    res_dir = result_dir(results_root, topology, allocation, difficulty, suffix)

    orig_screen_out = res_dir / "original_test_ar_cost.json"
    orig_pareto_out = res_dir / "original_test_pareto_front.json"
    orig_hv_out = res_dir / "original_test_hv.json"
    screen_out = res_dir / "screen_ar_cost.json"
    pareto_out = res_dir / "pareto_front.json"
    hv_out = res_dir / "hv_result.json"
    summary_out = res_dir / "summary.json"
    summary_table_csv = Path(results_root) / "ablation_summary.csv"

    if resume and summary_out.is_file() and hv_out.is_file() and pareto_out.is_file():
        print(f"\n[RESUME SKIP] {name} already completed.", flush=True)
        return True

    print(f"\n{'='*60}\n EXPERIMENT: {name}  [ablation={ablation}]\n{'='*60}", flush=True)

    for d in (ckpt_root, log_root, res_dir):
        ensure_dir(d)

    t0 = time.perf_counter()
    ok = True

    # -- Train --
    should_train = not skip_train
    if resume and ckpt.is_file() and not skip_train:
        print(f"[RESUME train] reusing: {ckpt}", flush=True)
        should_train = False
    if should_train:
        rc = train_one(tc, ckpt, csv_path, epochs, passes_per_file, warmup_epochs,
                       pretrained_vgae, ablation, dry_run)
        if rc != 0:
            print(f"[FAIL] train: {name}", flush=True)
            ok = False
    elif skip_train:
        print(f"[SKIP train] {name}", flush=True)

    # -- Episodes: use shared_episode_files passed in --
    episode_files = shared_episode_files or _list_episode_files(out_test)[:num_test_episodes]

    if ok and not skip_evaluate:
        if not dry_run and not ckpt.is_file():
            print(f"[FAIL] no checkpoint: {ckpt}", flush=True)
            ok = False
        else:
            # Original test
            rc_os = evaluate_screen_single(ckpt, tc, orig_screen_out, ablation, dry_run)
            if rc_os != 0:
                ok = False
            if ok:
                rc_op = evaluate_pareto_single(ckpt, tc, orig_pareto_out, orig_hv_out,
                                               pareto_points, ablation, dry_run, ref_acc, ref_cost)
                if rc_op != 0:
                    ok = False

            # Episode screening using shared episode files
            if ok:
                print(f"\n[EPISODE TESTS] screening {len(episode_files)} shared episodes", flush=True)
                rc = evaluate_screen_filelist(ckpt, episode_files, screen_out, ablation, dry_run)
                if rc != 0:
                    ok = False
                else:
                    if dry_run:
                        best = {"path": str(episode_files[0]) if episode_files else str(out_test / "episode_0000.json"),
                                "acc_ratio": 0.0, "deploy_cost": 0.0, "avg_cost_per_accepted": 0.0}
                    else:
                        best = select_best_ar_episode(screen_out)

                    if best is None:
                        ok = False
                    else:
                        episode_path = Path(best["path"])
                        rc = evaluate_pareto_single(ckpt, episode_path, pareto_out, hv_out,
                                                    pareto_points, ablation, dry_run, ref_acc, ref_cost)
                        if rc != 0:
                            ok = False
                        elif not dry_run:
                            # Compute MO indicators for this experiment
                            mo_ind = {}
                            if pareto_out.is_file():
                                try:
                                    with open(pareto_out) as f:
                                        scan_data = json.load(f)
                                    costs = [r["deploy_cost"] for r in scan_data if "deploy_cost" in r]
                                    cb = (min(costs), max(costs)) if costs else (0.0, 1.0)
                                    mo_ind = compute_mo_indicators(
                                        pareto_out, cb, np.array([0.0, 0.0]))
                                except Exception as e:
                                    print(f"[WARN] MO indicators failed: {e}", flush=True)

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
    print(f"[{status}] {name}  ({elapsed:.1f}s)", flush=True)
    return ok


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
        for r in ep_data:
            is_best = r.get("path") == best.get("path")
            rows.append({
                "experiment": name, "ablation": ablation, "test_type": "episode_test",
                "path": r.get("path", ""),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    is_kaggle = Path("/kaggle/working").is_dir()
    default_root = Path("/kaggle/working") if is_kaggle else Path(".")
    default_test_root = default_root / "data" / "test" if is_kaggle else Path("data/test")

    parser.add_argument("--testcase-dir", type=str, default="data/testcase")
    parser.add_argument("--test-root", type=str, default=str(default_test_root))
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pareto-points", type=int, default=11)
    parser.add_argument("--pretrained-vgae", type=str, default="")
    parser.add_argument("--ablation", type=str, default=DEFAULT_ABLATION, choices=list(ABLATIONS))

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ref-acc", type=float, default=None)
    parser.add_argument("--ref-cost", type=float, default=None)
    # Cross-ablation MO indicators (Experiment 3)
    parser.add_argument("--compute-cross-mo", action="store_true",
                        help="After all experiments, compute cross-ablation MO indicators (HV, R2, IGD+)")
    parser.add_argument("--cross-mo-ablations", type=str, nargs="*", default=None,
                        help="Ablations to include in cross-MO (default: all ABLATIONS)")

    args = parser.parse_args()

    topologies = tuple(args.topology) if args.topology else TOPOLOGIES
    allocations = tuple(args.allocation) if args.allocation else ALLOCATIONS
    difficulties = tuple(args.difficulty) if args.difficulty else DIFFICULTIES

    testcase_dir = resolve_testcase_dir(Path(args.testcase_dir))
    test_root = Path(args.test_root)
    results_root = Path(args.results_root)
    ckpt_root = Path(args.ckpt_root)
    log_root = Path(args.log_root)

    for d in (test_root, results_root, ckpt_root, log_root):
        ensure_dir(d)

    experiments = iter_experiments(topologies, allocations, difficulties)
    print(f"Scheduled experiments: {len(experiments)}", flush=True)
    print(f"Ablation: {args.ablation}  Resume: {args.resume}", flush=True)

    # -----------------------------------------------------------------------
    # Section 1: Ensure shared test episodes BEFORE ablation loop
    # -----------------------------------------------------------------------
    shared_episodes: Dict[Tuple[str, str, str], List[Path]] = {}
    if not args.skip_generate:
        print("\n[SHARED EPISODES] Ensuring shared test episodes per topology...", flush=True)
        for topology, allocation, difficulty in experiments:
            key = (topology, allocation, difficulty)
            if key not in shared_episodes:
                eps = ensure_shared_test_episodes(
                    topology, allocation, difficulty,
                    testcase_dir, test_root,
                    args.num_test_episodes, args.seed, args.dry_run,
                )
                shared_episodes[key] = eps
                names = [p.name for p in eps]
                print(f"  {topology}/{allocation}/{difficulty}: {names}", flush=True)
    else:
        for topology, allocation, difficulty in experiments:
            key = (topology, allocation, difficulty)
            d = shared_test_dir(test_root, topology, allocation, difficulty)
            shared_episodes[key] = _list_episode_files(d)[:args.num_test_episodes]
            print(f"[SKIP generate] using existing: {d} ({len(shared_episodes[key])} files)", flush=True)

    succeeded, failed = [], []

    for topology, allocation, difficulty in experiments:
        tc = testcase_path(testcase_dir, topology, allocation, difficulty)
        if not tc.is_file():
            print(f"[ERROR] missing: {tc}", flush=True)
            failed.append(f"{topology}_{allocation}_{difficulty}")
            if args.stop_on_error:
                break
            continue

        key = (topology, allocation, difficulty)
        episode_files = shared_episodes.get(key, [])

        ok = run_experiment(
            topology, allocation, difficulty,
            testcase_dir=testcase_dir, test_root=test_root,
            results_root=results_root, ckpt_root=ckpt_root, log_root=log_root,
            epochs=args.epochs, passes_per_file=args.passes_per_file,
            warmup_epochs=args.warmup_epochs, num_test_episodes=args.num_test_episodes,
            seed=args.seed, pareto_points=args.pareto_points,
            pretrained_vgae=args.pretrained_vgae, ablation=args.ablation,
            skip_train=args.skip_train, skip_generate=True,  # already handled above
            skip_evaluate=args.skip_evaluate, dry_run=args.dry_run,
            resume=args.resume, ref_acc=args.ref_acc, ref_cost=args.ref_cost,
            shared_episode_files=episode_files,
        )
        name = f"{topology}_{allocation}_{difficulty}{ablation_suffix(args.ablation)}"
        (succeeded if ok else failed).append(name)
        if not ok and args.stop_on_error:
            print("[ABORT] --stop-on-error set", flush=True)
            break

    # -----------------------------------------------------------------------
    # Section 2: Cross-ablation MO indicators (Experiment 3)
    # -----------------------------------------------------------------------
    if args.compute_cross_mo and not args.dry_run:
        cross_ablations = args.cross_mo_ablations or list(ABLATIONS)
        print(f"\n[CROSS-MO] Computing cross-ablation indicators for: {cross_ablations}", flush=True)
        for topology, allocation, difficulty in experiments:
            exp_base = f"{topology}_{allocation}_{difficulty}"
            compute_and_save_mo_indicators_for_experiment(
                exp_base, cross_ablations, results_root)

    print(f"\n{'='*60}", flush=True)
    print(f"Done.  OK={len(succeeded)}  FAIL={len(failed)}", flush=True)
    if succeeded:
        print("Succeeded:", ", ".join(succeeded), flush=True)
    if failed:
        print("Failed   :", ", ".join(failed), flush=True)


if __name__ == "__main__":
    main()