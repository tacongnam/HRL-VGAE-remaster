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
  python run_kaggle_experiments.py --topology nsf --allocation centers
  python run_kaggle_experiments.py --topology nsf --allocation centers --difficulty hard
  python run_kaggle_experiments.py --skip-train --skip-generate
"""

from __future__ import annotations

import argparse
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

NUM_TEST_EPISODES = 10
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def testcase_path(testcase_dir: Path, topology: str, allocation: str, difficulty: str) -> Path:
    return testcase_dir / f"{topology}_{allocation}_{difficulty}.json"


def test_out_dir(test_root: Path, topology: str, allocation: str, difficulty: str) -> Path:
    return test_root / topology / allocation / difficulty


def result_dir(results_root: Path, topology: str, allocation: str, difficulty: str) -> Path:
    return results_root / f"{topology}_{allocation}_{difficulty}"


def checkpoint_path(ckpt_root: Path, topology: str, allocation: str, difficulty: str) -> Path:
    return ckpt_root / f"{topology}_{allocation}_{difficulty}.pt"


def log_csv_path(log_root: Path, topology: str, allocation: str, difficulty: str) -> Path:
    return log_root / f"{topology}_{allocation}_{difficulty}.csv"


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
    ]
    if pretrained_vgae:
        cmd.extend(["--pretrained-vgae", pretrained_vgae])
    return run_cmd(cmd, dry_run=dry_run)


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
    ]
    return run_cmd(cmd, dry_run=dry_run)


def evaluate_pareto_single(
    checkpoint: Path,
    episode_path: Path,
    pareto_out: Path,
    hv_out: Path,
    pareto_points: int,
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
    ]
    return run_cmd(cmd, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Select best-AR episode from screening JSON
# ---------------------------------------------------------------------------

def select_best_ar_episode(screen_json: Path) -> Optional[Dict[str, Any]]:
    """
    Read screening scan (1 weight × N episodes).
    Return the entry with highest acc_ratio.
    Tie-break: lower deploy_cost, then path name.
    """
    if not screen_json.is_file():
        print(f"[ERROR] screen file missing: {screen_json}", flush=True)
        return None

    with open(screen_json) as f:
        data = json.load(f)

    if not data:
        print(f"[ERROR] empty screen results: {screen_json}", flush=True)
        return None

    # Aggregate by path in case of duplicate weights (should be 1)
    best_by_path: Dict[str, Dict[str, Any]] = {}
    for row in data:
        path = row.get("path", "")
        ar = float(row.get("acc_ratio", 0.0))
        cost = float(row.get("deploy_cost", float("inf")))
        prev = best_by_path.get(path)
        if prev is None or ar > prev["acc_ratio"] or (
            ar == prev["acc_ratio"] and cost < prev["deploy_cost"]
        ):
            best_by_path[path] = {
                "path": path,
                "acc_ratio": ar,
                "deploy_cost": cost,
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
        f"AR={best['acc_ratio']:.4f}  cost={best['deploy_cost']:.2f}",
        flush=True,
    )
    print("[SELECT] ranking (all tests):", flush=True)
    for i, r in enumerate(ranked, 1):
        mark = " <-- PARETO" if i == 1 else ""
        print(
            f"  {i:2d}. AR={r['acc_ratio']:.4f}  cost={r['deploy_cost']:.2f}  "
            f"{Path(r['path']).name}{mark}",
            flush=True,
        )
    return best


def write_summary(
    summary_path: Path,
    *,
    name: str,
    screen_json: Path,
    best: Dict[str, Any],
    pareto_out: Path,
    hv_out: Path,
) -> None:
    """Save a small summary linking screening + chosen Pareto episode + metrics."""
    ensure_dir(summary_path.parent)
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
        "screening": {
            "file": str(screen_json),
            "num_entries": len(all_screen),
            "all_tests": [
                {
                    "path": r.get("path"),
                    "acc_ratio": r.get("acc_ratio"),
                    "deploy_cost": r.get("deploy_cost"),
                    "w_accept": r.get("w_accept"),
                    "w_cost": r.get("w_cost"),
                }
                for r in all_screen
            ],
        },
        "selected_for_pareto": {
            "path": best.get("path"),
            "acc_ratio": best.get("acc_ratio"),
            "deploy_cost": best.get("deploy_cost"),
        },
        "pareto_scan": str(pareto_out),
        "hv_result": str(hv_out),
        "metrics": hv_metrics,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SUMMARY] saved → {summary_path}", flush=True)


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
    skip_train: bool,
    skip_generate: bool,
    skip_evaluate: bool,
    dry_run: bool,
) -> bool:
    """
    One independent experiment:
      train → generate 10 tests
      → screen AR/cost on all 10 (checkpoint weights via --ar-cost-only)
      → pick highest-AR episode → full Pareto + HV on that episode only.
    """
    name = f"{topology}_{allocation}_{difficulty}"
    tc = testcase_path(testcase_dir, topology, allocation, difficulty)

    if not tc.is_file():
        print(f"[SKIP] missing testcase: {tc}", flush=True)
        return False

    print(f"\n{'=' * 60}", flush=True)
    print(f" EXPERIMENT: {name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    ckpt = checkpoint_path(ckpt_root, topology, allocation, difficulty)
    csv_path = log_csv_path(log_root, topology, allocation, difficulty)
    out_test = test_out_dir(test_root, topology, allocation, difficulty)
    res_dir = result_dir(results_root, topology, allocation, difficulty)

    screen_out = res_dir / "screen_ar_cost.json"       # all 10 tests, 1 weight
    pareto_out = res_dir / "pareto_front.json"         # full Pareto on best AR
    hv_out = res_dir / "hv_result.json"
    summary_out = res_dir / "summary.json"

    for d in (ckpt_root, log_root, test_root, results_root, out_test, res_dir):
        ensure_dir(d)

    t0 = time.perf_counter()
    ok = True

    # ----- 1. Train -----
    if not skip_train:
        rc = train_one(
            train_file=tc,
            checkpoint=ckpt,
            csv_path=csv_path,
            epochs=epochs,
            passes_per_file=passes_per_file,
            warmup_epochs=warmup_epochs,
            pretrained_vgae=pretrained_vgae,
            dry_run=dry_run,
        )
        if rc != 0:
            print(f"[FAIL] train: {name}", flush=True)
            ok = False
    else:
        print(f"[SKIP train] {name}", flush=True)
        if not dry_run and not ckpt.is_file():
            print(f"[WARN] checkpoint missing: {ckpt}", flush=True)

    # ----- 2. Generate 10 test episodes -----
    if ok and not skip_generate:
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

    # ----- 3–5. Screen all → pick best AR → Pareto on that episode -----
    if ok and not skip_evaluate:
        if not dry_run and not ckpt.is_file():
            print(f"[FAIL] no checkpoint for evaluate: {ckpt}", flush=True)
            ok = False
        else:
            # 3. Screen: AR + cost on all 10 using checkpoint weights
            print(
                f"[SCREEN] AR/cost (checkpoint weights) on all episodes in {out_test}",
                flush=True,
            )
            rc = evaluate_screen_all(
                checkpoint=ckpt,
                data_dir=out_test,
                screen_out=screen_out,
                dry_run=dry_run,
            )
            if rc != 0:
                print(f"[FAIL] screen: {name}", flush=True)
                ok = False
            else:
                # 4. Select highest-AR episode
                if dry_run:
                    best = {
                        "path": str(out_test / "episode_0000.json"),
                        "acc_ratio": 0.0,
                        "deploy_cost": 0.0,
                    }
                    print(f"[DRY-RUN] would select best-AR episode from {screen_out}", flush=True)
                else:
                    best = select_best_ar_episode(screen_out)

                if best is None:
                    ok = False
                else:
                    # 5. Full Pareto only on the selected episode
                    episode_path = Path(best["path"])
                    print(
                        f"[PARETO] full scan on best-AR episode "
                        f"({pareto_points} weights): {episode_path}",
                        flush=True,
                    )
                    rc = evaluate_pareto_single(
                        checkpoint=ckpt,
                        episode_path=episode_path,
                        pareto_out=pareto_out,
                        hv_out=hv_out,
                        pareto_points=pareto_points,
                        dry_run=dry_run,
                    )
                    if rc != 0:
                        print(f"[FAIL] pareto: {name}", flush=True)
                        ok = False
                    elif not dry_run:
                        write_summary(
                            summary_out,
                            name=name,
                            screen_json=screen_out,
                            best=best,
                            pareto_out=pareto_out,
                            hv_out=hv_out,
                        )
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
    default_root = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")

    parser.add_argument("--testcase-dir", type=str, default="data/testcase")
    parser.add_argument("--test-root", type=str, default="data/test")
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

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")

    args = parser.parse_args()

    topologies = tuple(args.topology) if args.topology else TOPOLOGIES
    allocations = tuple(args.allocation) if args.allocation else ALLOCATIONS
    difficulties = tuple(args.difficulty) if args.difficulty else DIFFICULTIES

    testcase_dir = Path(args.testcase_dir)
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
        f"screen AR/cost (ckpt weights) → Pareto ({args.pareto_points} pts) on best-AR only",
        flush=True,
    )

    succeeded, failed, skipped = [], [], []

    for topology, allocation, difficulty in experiments:
        tc = testcase_path(testcase_dir, topology, allocation, difficulty)
        if not tc.is_file():
            print(f"[SKIP] missing: {tc}", flush=True)
            skipped.append(f"{topology}_{allocation}_{difficulty}")
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
            skip_train=args.skip_train,
            skip_generate=args.skip_generate,
            skip_evaluate=args.skip_evaluate,
            dry_run=args.dry_run,
        )
        name = f"{topology}_{allocation}_{difficulty}"
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
