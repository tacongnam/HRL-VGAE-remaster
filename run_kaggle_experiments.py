"""
Kaggle experiment runner for HRL-VGAE.

Pipeline per testcase (independent experiment):
  1. Train on data/testcase/{topology}_{allocation}_{difficulty}.json
  2. Generate 10 test episodes → data/test/{topology}/{allocation}/{difficulty}/
  3. Screen all 10 tests with checkpoint weights (--ar-cost-only) → AR + cost
  4. Select the episode with the highest AR
  5. Full Pareto scan ONLY on that episode → Hypervolume / Spacing / Spread

Topology : nsf, cogent, conus
Allocation: centers, uniform, rural, urban
Difficulty: easy, normal, hard

Usage:
  python run_kaggle_experiments.py
  python run_kaggle_experiments.py --topology nsf
  python run_kaggle_experiments.py --topology nsf --allocation centers --difficulty easy
  python run_kaggle_experiments.py --topology nsf --allocation centers --difficulty hard
  python run_kaggle_experiments.py --skip-train --skip-generate
  python run_kaggle_experiments.py --ablation GreedyLowLevel
  python run_kaggle_experiments.py --ablation Random-Fit --topology nsf
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

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

TOPOLOGIES = ("nsf", "cogent", "conus")
ALLOCATIONS = ("centers", "uniform", "rural", "urban")
DIFFICULTIES = ("easy", "normal", "hard")

# Ablation modes — keep in sync with the `choices` in train.py and evaluate.py.
ABLATIONS = ("DefaultBaseline", "Random-Fit", "GreedyFIFO", "GreedyLowLevel", "HRL-VGAE")
DEFAULT_ABLATION = "HRL-VGAE"

NUM_TEST_EPISODES = 10
DEFAULT_SEED = 42


def ablation_suffix(ablation: str) -> str:
    """Output-name suffix for non-default ablations.

    The default ``HRL-VGAE`` keeps the historical paths (no suffix) so existing
    results / checkpoints / logs remain valid. Every other mode gets ``_<ablation>``
    appended so ablations never overwrite one another.
    """
    return "" if ablation == DEFAULT_ABLATION else f"_{ablation}"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def evaluate_screen_single(
    checkpoint: Path,
    episode_path: Path,
    screen_out: Path,
    ablation: str,
    dry_run: bool,
) -> int:
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


def evaluate_screen_all(
    checkpoint: Path,
    data_dir: Path,
    screen_out: Path,
    ablation: str,
    dry_run: bool,
) -> int:
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


def evaluate_pareto_single(
    checkpoint: Path,
    episode_path: Path,
    pareto_out: Path,
    hv_out: Path,
    pareto_points: int,
    ablation: str,
    dry_run: bool,
    ref_acc: Optional[float] = None,
    ref_cost: Optional[float] = None,
    **kwargs: Any,
) -> int:
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

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def testcase_path(testcase_dir: Path, topology: str, allocation: str, difficulty: str) -> Path:
    return testcase_dir / f"{topology}_{allocation}_{difficulty}.json"


def test_out_dir(test_root: Path, topology: str, allocation: str, difficulty: str) -> Path:
    return test_root / topology / allocation / difficulty


def result_dir(
    results_root: Path,
    topology: str,
    allocation: str,
    difficulty: str,
    suffix: str = "",
) -> Path:
    return results_root / f"{topology}_{allocation}_{difficulty}{suffix}"


def checkpoint_path(
    ckpt_root: Path,
    topology: str,
    allocation: str,
    difficulty: str,
    suffix: str = "",
) -> Path:
    return ckpt_root / f"{topology}_{allocation}_{difficulty}{suffix}.pt"


def log_csv_path(
    log_root: Path,
    topology: str,
    allocation: str,
    difficulty: str,
    suffix: str = "",
) -> Path:
    return log_root / f"{topology}_{allocation}_{difficulty}{suffix}.csv"


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


def train_one(
    train_file: Path,
    checkpoint: Path,
    csv_path: Path,
    epochs: int,
    passes_per_file: int,
    warmup_epochs: int,
    pretrained_vgae: str,
    ablation: str,
    dry_run: bool,
) -> int:
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

def generate_tests(
    topology_json: Path,
    out_dir: Path,
    num_episodes: int,
    seed: int,
    dry_run: bool,
) -> int:
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


def evaluate_screen_all(
    checkpoint: Path,
    data_dir: Path,
    screen_out: Path,
    ablation: str,
    dry_run: bool,
) -> int:
    """
    Run all episodes once using weights from the checkpoint (no Pareto grid).
    Produces AR + deploy_cost per episode for ranking.
    """
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


def evaluate_pareto_single(
    checkpoint: Path,
    episode_path: Path,
    pareto_out: Path,
    hv_out: Path,
    pareto_points: int,
    ablation: str,
    dry_run: bool,
) -> int:
    """Full Pareto scan on one episode only, then compute HV metrics."""
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
    return run_cmd(cmd, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Select best-AR episode from screening JSON
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
                "path": path,
                "acc_ratio": ar,
                "deploy_cost": cost,
                "avg_cost_per_accepted": avg_cost,
                "accepted": row.get("accepted"),
                "rejected": row.get("rejected"),
                "w_accept": row.get("w_accept"),
                "w_cost": row.get("w_cost"),
            }

    ranked = sorted(
        best_by_path.values(),
        key=lambda r: (-r["acc_ratio"], r["deploy_cost"], r["path"]),
    )
    best = ranked[0]
    print(
        f"[SELECT] best AR episode: {best['path']}  "
        f"AR={best['acc_ratio']:.4f}  cost={best['deploy_cost']:.2f}  "
        f"avg_cost_per_accepted={best.get('avg_cost_per_accepted', 0.0):.2f}",
        flush=True,
    )
    print("[SELECT] ranking (all tests):", flush=True)
    for i, r in enumerate(ranked, 1):
        mark = " <-- PARETO" if i == 1 else ""
        print(
            f"  {i:2d}. AR={r['acc_ratio']:.4f}  cost={r['deploy_cost']:.2f}  "
            f"avg_cost={r.get('avg_cost_per_accepted', 0.0):.2f}  "
            f"{Path(r['path']).name}{mark}",
            flush=True,
        )
    return best


def write_summary(
    summary_path: Path,
    *,
    name: str,
    ablation: str,
    orig_screen_json: Path,
    orig_hv_json: Path,
    screen_json: Path,
    best: Dict[str, Any],
    pareto_out: Path,
    hv_out: Path,
) -> None:
    ensure_dir(summary_path.parent)

    orig_metrics: Dict[str, Any] = {}
    if orig_hv_json.is_file():
        with open(orig_hv_json) as f:
            orig_hv_data = json.load(f)
        orig_metrics = {
            "hypervolume": orig_hv_data.get("hypervolume"),
            "spacing": orig_hv_data.get("spacing"),
            "spread": orig_hv_data.get("spread"),
            "delta": orig_hv_data.get("delta"),
            "front_size": len(orig_hv_data.get("pareto_front", [])),
        }

    orig_screen_data: List[Dict[str, Any]] = []
    if orig_screen_json.is_file():
        with open(orig_screen_json) as f:
            orig_screen_data = json.load(f)
    orig_entry = orig_screen_data[0] if orig_screen_data else {}

    hv_metrics: Dict[str, Any] = {}
    if hv_out.is_file():
        with open(hv_out) as f:
            hv = json.load(f)
        hv_metrics = {
            "hypervolume": hv.get("hypervolume"),
            "spacing": hv.get("spacing"),
            "spread": hv.get("spread"),
            "delta": hv.get("delta"),
            "front_size": len(hv.get("pareto_front", [])),
        }

    all_screen: List[Dict[str, Any]] = []
    if screen_json.is_file():
        with open(screen_json) as f:
            all_screen = json.load(f)

    summary = {
        "experiment": name,
        "ablation": ablation,
        "original_test": {
            "file": str(orig_screen_json),
            "path": orig_entry.get("path"),
            "acc_ratio": orig_entry.get("acc_ratio"),
            "deploy_cost": orig_entry.get("deploy_cost"),
            "avg_cost_per_accepted": orig_entry.get("avg_cost_per_accepted"),
            "accepted": orig_entry.get("accepted"),
            "rejected": orig_entry.get("rejected"),
            "hv_result": str(orig_hv_json),
            "metrics": orig_metrics,
        },
        "episode_tests": {
            "screening_file": str(screen_json),
            "num_entries": len(all_screen),
            "all_tests": [
                {
                    "path": r.get("path"),
                    "acc_ratio": r.get("acc_ratio"),
                    "deploy_cost": r.get("deploy_cost"),
                    "avg_cost_per_accepted": r.get("avg_cost_per_accepted"),
                    "accepted": r.get("accepted"),
                    "rejected": r.get("rejected"),
                    "w_accept": r.get("w_accept"),
                    "w_cost": r.get("w_cost"),
                }
                for r in all_screen
            ],
            "selected_for_pareto": {
                "path": best.get("path"),
                "acc_ratio": best.get("acc_ratio"),
                "deploy_cost": best.get("deploy_cost"),
                "avg_cost_per_accepted": best.get("avg_cost_per_accepted"),
            },
            "pareto_scan": str(pareto_out),
            "hv_result": str(hv_out),
            "metrics": hv_metrics,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SUMMARY] saved → {summary_path}", flush=True)


def update_ablation_summary_table(
    csv_path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = [
        "experiment",
        "ablation",
        "test_type",
        "path",
        "acc_ratio",
        "deploy_cost",
        "avg_cost_per_accepted",
        "accepted",
        "rejected",
        "hypervolume",
        "spacing",
        "spread",
    ]

    existing_rows = []
    if csv_path.is_file():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(r)

    def row_key(d: Dict[str, Any]) -> Tuple[str, str, str, str]:
        return (
            str(d.get("experiment", "")),
            str(d.get("ablation", "")),
            str(d.get("test_type", "")),
            str(d.get("path", "")),
        )

    updated_keys = {row_key(r) for r in rows}
    merged_rows = [r for r in existing_rows if row_key(r) not in updated_keys]
    merged_rows.extend(rows)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in merged_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[TABLE] updated ablation summary table → {csv_path}", flush=True)

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

def iter_experiments(
    topologies: Iterable[str],
    allocations: Iterable[str],
    difficulties: Iterable[str],
) -> List[Tuple[str, str, str]]:
    return [
        (t, a, d)
        for t in topologies
        for a in allocations
        for d in difficulties
    ]

def run_experiment(
    topology: str,
    allocation: str,
    difficulty: str,
    *,
    testcase_dir: Path,
    test_root: Path,
    results_root: Path,
    ckpt_root: Path,
    log_root: Path,
    epochs: int,
    passes_per_file: int,
    warmup_epochs: int,
    num_test_episodes: int,
    seed: int,
    pareto_points: int,
    pretrained_vgae: str,
    ablation: str,
    skip_train: bool,
    skip_generate: bool,
    skip_evaluate: bool,
    dry_run: bool,
    resume: bool = True,
    ref_acc: Optional[float] = None,
    ref_cost: Optional[float] = None,
) -> bool:
    suffix = ablation_suffix(ablation)
    name = f"{topology}_{allocation}_{difficulty}{suffix}"
    tc = testcase_path(testcase_dir, topology, allocation, difficulty)

    if not tc.is_file():
        print(f"[ERROR] missing original testcase: {tc}", flush=True)
        return False

    ckpt = checkpoint_path(ckpt_root, topology, allocation, difficulty, suffix)
    csv_path = log_csv_path(log_root, topology, allocation, difficulty, suffix)
    out_test = test_out_dir(test_root, topology, allocation, difficulty)
    res_dir = result_dir(results_root, topology, allocation, difficulty, suffix)

    orig_screen_out = res_dir / "original_test_ar_cost.json"
    orig_pareto_out = res_dir / "original_test_pareto_front.json"
    orig_hv_out = res_dir / "original_test_hv.json"

    screen_out = res_dir / "screen_ar_cost.json"
    pareto_out = res_dir / "pareto_front.json"
    hv_out = res_dir / "hv_result.json"
    summary_out = res_dir / "summary.json"
    summary_table_csv = results_root / "ablation_summary.csv"

    if (
        resume
        and summary_out.is_file()
        and hv_out.is_file()
        and pareto_out.is_file()
        and orig_screen_out.is_file()
        and orig_hv_out.is_file()
    ):
        print(f"\n[RESUME SKIP] Experiment {name} already completed.", flush=True)
        return True

    print(f"\n{'=' * 60}", flush=True)
    print(f" EXPERIMENT: {name}  [ablation={ablation}]", flush=True)
    print(f"{'=' * 60}", flush=True)

    for d in (ckpt_root, log_root, test_root, results_root, out_test, res_dir):
        ensure_dir(d)

    t0 = time.perf_counter()
    ok = True

    should_train = not skip_train
    if resume and ckpt.is_file() and not skip_train:
        print(f"[RESUME train] Reusing existing checkpoint: {ckpt}", flush=True)
        should_train = False

    if should_train:
        rc = train_one(
            train_file=tc,
            checkpoint=ckpt,
            csv_path=csv_path,
            epochs=epochs,
            passes_per_file=passes_per_file,
            warmup_epochs=warmup_epochs,
            pretrained_vgae=pretrained_vgae,
            ablation=ablation,
            dry_run=dry_run,
        )
        if rc != 0:
            print(f"[FAIL] train: {name}", flush=True)
            ok = False
    elif skip_train:
        print(f"[SKIP train] {name}", flush=True)
        if not dry_run and not ckpt.is_file():
            print(f"[WARN] checkpoint missing: {ckpt}", flush=True)

    existing_episodes = len(list(out_test.glob("episode_*.json"))) if out_test.is_dir() else 0
    should_generate = ok and not skip_generate
    if resume and existing_episodes >= num_test_episodes and not skip_generate:
        print(f"[RESUME generate] Reusing {existing_episodes} test episodes in {out_test}", flush=True)
        should_generate = False

    if should_generate:
        rc = generate_tests(
            topology_json=tc,
            out_dir=out_test,
            num_episodes=num_test_episodes,
            seed=seed,
            dry_run=dry_run,
        )
        if rc != 0:
            print(f"[FAIL] generate: {name}", flush=True)
            ok = False
    elif skip_generate:
        print(f"[SKIP generate] {name}", flush=True)

    if ok and not skip_evaluate:
        if not dry_run and not ckpt.is_file():
            print(f"[FAIL] no checkpoint for evaluate: {ckpt}", flush=True)
            ok = False
        else:
            print(f"\n[ORIGINAL TEST] evaluating on train testcase: {tc}", flush=True)
            rc_orig_screen = evaluate_screen_single(
                checkpoint=ckpt,
                episode_path=tc,
                screen_out=orig_screen_out,
                ablation=ablation,
                dry_run=dry_run,
            )
            if rc_orig_screen != 0:
                print(f"[FAIL] original test screen: {name}", flush=True)
                ok = False

            if ok:
                rc_orig_pareto = evaluate_pareto_single(
                    checkpoint=ckpt,
                    episode_path=tc,
                    pareto_out=orig_pareto_out,
                    hv_out=orig_hv_out,
                    pareto_points=pareto_points,
                    ablation=ablation,
                    dry_run=dry_run,
                    ref_acc=ref_acc,
                    ref_cost=ref_cost,
                )
                if rc_orig_pareto != 0:
                    print(f"[FAIL] original test pareto: {name}", flush=True)
                    ok = False

            if ok:
                print(
                    f"\n[EPISODE TESTS] screening on episodes in {out_test}",
                    flush=True,
                )
                rc = evaluate_screen_all(
                    checkpoint=ckpt,
                    data_dir=out_test,
                    screen_out=screen_out,
                    ablation=ablation,
                    dry_run=dry_run,
                )
                if rc != 0:
                    print(f"[FAIL] episode tests screen: {name}", flush=True)
                    ok = False
                else:
                    if dry_run:
                        best = {
                            "path": str(out_test / "episode_0000.json"),
                            "acc_ratio": 0.0,
                            "deploy_cost": 0.0,
                            "avg_cost_per_accepted": 0.0,
                        }
                        print(f"[DRY-RUN] would select best-AR episode from {screen_out}", flush=True)
                    else:
                        best = select_best_ar_episode(screen_out)

                    if best is None:
                        ok = False
                    else:
                        episode_path = Path(best["path"])
                        print(
                            f"[EPISODE TESTS] full Pareto scan on best-AR episode "
                            f"({pareto_points} weights): {episode_path}",
                            flush=True,
                        )
                        rc = evaluate_pareto_single(
                            checkpoint=ckpt,
                            episode_path=episode_path,
                            pareto_out=pareto_out,
                            hv_out=hv_out,
                            pareto_points=pareto_points,
                            ablation=ablation,
                            dry_run=dry_run,
                            ref_acc=ref_acc,
                            ref_cost=ref_cost,
                        )
                        if rc != 0:
                            print(f"[FAIL] episode pareto: {name}", flush=True)
                            ok = False
                        elif not dry_run:
                            write_summary(
                                summary_out,
                                name=name,
                                ablation=ablation,
                                orig_screen_json=orig_screen_out,
                                orig_hv_json=orig_hv_out,
                                screen_json=screen_out,
                                best=best,
                                pareto_out=pareto_out,
                                hv_out=hv_out,
                            )

                            rows_to_record = []
                            if orig_screen_out.is_file():
                                with open(orig_screen_out) as f:
                                    orig_s_data = json.load(f)
                                orig_row = orig_s_data[0] if orig_s_data else {}
                                orig_hv_val, orig_sp_val, orig_spread_val = "", "", ""
                                if orig_hv_out.is_file():
                                    with open(orig_hv_out) as f:
                                        orig_h_data = json.load(f)
                                    orig_hv_val = orig_h_data.get("hypervolume", "")
                                    orig_sp_val = orig_h_data.get("spacing", "")
                                    orig_spread_val = orig_h_data.get("spread", orig_h_data.get("delta", ""))
                                rows_to_record.append({
                                    "experiment": name,
                                    "ablation": ablation,
                                    "test_type": "original_test",
                                    "path": orig_row.get("path", str(tc)),
                                    "acc_ratio": orig_row.get("acc_ratio", ""),
                                    "deploy_cost": orig_row.get("deploy_cost", ""),
                                    "avg_cost_per_accepted": orig_row.get("avg_cost_per_accepted", ""),
                                    "accepted": orig_row.get("accepted", ""),
                                    "rejected": orig_row.get("rejected", ""),
                                    "hypervolume": orig_hv_val,
                                    "spacing": orig_sp_val,
                                    "spread": orig_spread_val,
                                })

                            if screen_out.is_file():
                                with open(screen_out) as f:
                                    ep_s_data = json.load(f)
                                best_hv_val, best_sp_val, best_spread_val = "", "", ""
                                if hv_out.is_file():
                                    with open(hv_out) as f:
                                        ep_h_data = json.load(f)
                                    best_hv_val = ep_h_data.get("hypervolume", "")
                                    best_sp_val = ep_h_data.get("spacing", "")
                                    best_spread_val = ep_h_data.get("spread", ep_h_data.get("delta", ""))

                                for ep_r in ep_s_data:
                                    is_best = ep_r.get("path") == best.get("path")
                                    rows_to_record.append({
                                        "experiment": name,
                                        "ablation": ablation,
                                        "test_type": "episode_test",
                                        "path": ep_r.get("path", ""),
                                        "acc_ratio": ep_r.get("acc_ratio", ""),
                                        "deploy_cost": ep_r.get("deploy_cost", ""),
                                        "avg_cost_per_accepted": ep_r.get("avg_cost_per_accepted", ""),
                                        "accepted": ep_r.get("accepted", ""),
                                        "rejected": ep_r.get("rejected", ""),
                                        "hypervolume": best_hv_val if is_best else "",
                                        "spacing": best_sp_val if is_best else "",
                                        "spread": best_spread_val if is_best else "",
                                    })

                                if ep_s_data:
                                    ars = [float(r.get("acc_ratio", 0.0)) for r in ep_s_data]
                                    costs = [float(r.get("deploy_cost", 0.0)) for r in ep_s_data]
                                    avg_costs = [float(r.get("avg_cost_per_accepted", 0.0)) for r in ep_s_data]
                                    accs = [int(r.get("accepted", 0)) for r in ep_s_data]
                                    rejs = [int(r.get("rejected", 0)) for r in ep_s_data]
                                    rows_to_record.append({
                                        "experiment": name,
                                        "ablation": ablation,
                                        "test_type": "episode_test_mean",
                                        "path": f"{len(ep_s_data)}_episodes_mean",
                                        "acc_ratio": f"{np.mean(ars):.4f}",
                                        "deploy_cost": f"{np.mean(costs):.2f}",
                                        "avg_cost_per_accepted": f"{np.mean(avg_costs):.2f}",
                                        "accepted": f"{np.mean(accs):.1f}",
                                        "rejected": f"{np.mean(rejs):.1f}",
                                        "hypervolume": best_hv_val,
                                        "spacing": best_sp_val,
                                        "spread": best_spread_val,
                                    })

                            update_ablation_summary_table(summary_table_csv, rows_to_record)
    elif skip_evaluate:
        print(f"[SKIP evaluate] {name}", flush=True)

    elapsed = time.perf_counter() - t0
    status = "OK" if ok else "FAILED"
    print(f"[{status}] {name}  ({elapsed:.1f}s)", flush=True)
    return ok

def main():
    parser = argparse.ArgumentParser(
        description="HRL-VGAE Kaggle runner: screen all tests by AR, Pareto on best only"
    )
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
    parser.add_argument(
        "--pareto-points",
        type=int,
        default=11,
        help="Weight grid size for the BEST-AR episode only (default 11). "
             "Screening of all 10 tests uses checkpoint weights (--ar-cost-only).",
    )
    parser.add_argument("--pretrained-vgae", type=str, default="")
    parser.add_argument(
        "--ablation",
        type=str,
        default=DEFAULT_ABLATION,
        choices=list(ABLATIONS),
        help="Ablation mode shared by train.py and evaluate.py (default HRL-VGAE). "
             "Non-default modes get a '_<ablation>' suffix on checkpoint/csv/results.",
    )

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume execution by skipping experiments that already have summary.json (default: True).",
    )

    parser.add_argument("--ref-acc", type=float, default=None)
    parser.add_argument("--ref-cost", type=float, default=None)

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
    for t, a, d in experiments:
        print(f"  - {t}_{a}_{d}", flush=True)
    print(
        f"Pipeline: train → generate {args.num_test_episodes} tests → "
        f"original test evaluation → episode tests screening & Pareto",
        flush=True,
    )
    print(f"Ablation: {args.ablation}", flush=True)
    print(f"Resume  : {args.resume}", flush=True)

    succeeded, failed, skipped = [], [], []

    for topology, allocation, difficulty in experiments:
        tc = testcase_path(testcase_dir, topology, allocation, difficulty)
        if not tc.is_file():
            print(f"[ERROR] missing original testcase: {tc}", flush=True)
            failed.append(f"{topology}_{allocation}_{difficulty}")
            if args.stop_on_error:
                break
            continue

        ok = run_experiment(
            topology, allocation, difficulty,
            testcase_dir=testcase_dir,
            test_root=test_root,
            results_root=results_root,
            ckpt_root=ckpt_root,
            log_root=log_root,
            epochs=args.epochs,
            passes_per_file=args.passes_per_file,
            warmup_epochs=args.warmup_epochs,
            num_test_episodes=args.num_test_episodes,
            seed=args.seed,
            pareto_points=args.pareto_points,
            pretrained_vgae=args.pretrained_vgae,
            ablation=args.ablation,
            skip_train=args.skip_train,
            skip_generate=args.skip_generate,
            skip_evaluate=args.skip_evaluate,
            dry_run=args.dry_run,
            resume=args.resume,
            ref_acc=args.ref_acc,
            ref_cost=args.ref_cost,
        )
        name = f"{topology}_{allocation}_{difficulty}{ablation_suffix(args.ablation)}"
        if ok:
            succeeded.append(name)
        else:
            failed.append(name)
            if args.stop_on_error:
                print("[ABORT] --stop-on-error set", flush=True)
                break

    print("\n" + "=" * 60, flush=True)
    print(f"Done.  OK={len(succeeded)}  FAIL={len(failed)}  SKIP={len(skipped)}", flush=True)
    if succeeded:
        print("Succeeded:", ", ".join(succeeded), flush=True)
    if failed:
        print("Failed   :", ", ".join(failed), flush=True)
    if skipped:
        print("Skipped  :", ", ".join(skipped), flush=True)

if __name__ == "__main__":
    main()
