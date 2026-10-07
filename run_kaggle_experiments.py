"""
Kaggle experiment runner for HRL-VGAE.

Pipeline per testcase (independent experiment):
  1. Train on data/testcase/{topology}_{allocation}_{difficulty}.json
  2. Generate 10 test episodes → data/test/{topology}/{allocation}/{difficulty}/
  3. Pareto scan + Hypervolume / Spacing / Spread
     → results/{topology}_{allocation}_{difficulty}/

Topology : nsf, cogent, conus
Allocation: centers, uniform, rural, urban
Difficulty: easy, normal, hard

Usage (Kaggle notebook or script):
  # Full grid (36 experiments) — may exceed 12h
  python run_kaggle_experiments.py

  # Subset
  python run_kaggle_experiments.py --topology nsf
  python run_kaggle_experiments.py --topology nsf --allocation centers
  python run_kaggle_experiments.py --topology nsf --allocation centers --difficulty hard
  python run_kaggle_experiments.py --skip-train --skip-generate   # evaluate only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

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
    """Create directory if missing; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def testcase_path(testcase_dir: Path, topology: str, allocation: str, difficulty: str) -> Path:
    """data/testcase/{topology}_{allocation}_{difficulty}.json"""
    return testcase_dir / f"{topology}_{allocation}_{difficulty}.json"


def test_out_dir(test_root: Path, topology: str, allocation: str, difficulty: str) -> Path:
    """data/test/{topology}/{allocation}/{difficulty}/"""
    return test_root / topology / allocation / difficulty


def result_dir(results_root: Path, topology: str, allocation: str, difficulty: str) -> Path:
    """results/{topology}_{allocation}_{difficulty}/"""
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


def evaluate_one(
    checkpoint: Path,
    data_dir: Path,
    pareto_out: Path,
    hv_out: Path,
    pareto_points: int,
    dry_run: bool,
) -> int:
    ensure_dir(pareto_out.parent)
    ensure_dir(hv_out.parent)
    cmd = [
        sys.executable, "evaluate.py",
        "--pareto-scan",
        "--checkpoint", str(checkpoint),
        "--data-dir", str(data_dir),
        "--pareto-points", str(pareto_points),
        "--pareto-out", str(pareto_out),
        "--compute-hv",
        "--hv-out", str(hv_out),
    ]
    return run_cmd(cmd, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

def iter_experiments(
    topologies: Iterable[str],
    allocations: Iterable[str],
    difficulties: Iterable[str],
) -> List[tuple[str, str, str]]:
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
    Run one independent experiment.
    Returns True on success (all requested stages OK), False otherwise.
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
    pareto_out = res_dir / "pareto_front.json"
    hv_out = res_dir / "hv_result.json"

    # Ensure all output roots exist up front
    ensure_dir(ckpt_root)
    ensure_dir(log_root)
    ensure_dir(test_root)
    ensure_dir(results_root)
    ensure_dir(out_test)
    ensure_dir(res_dir)

    t0 = time.perf_counter()
    ok = True

    # 1. Train
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
        if not ckpt.is_file() and not dry_run:
            print(f"[WARN] checkpoint missing: {ckpt}", flush=True)

    # 2. Generate 10 test episodes
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
    else:
        if skip_generate:
            print(f"[SKIP generate] {name}", flush=True)

    # 3. Evaluate
    if ok and not skip_evaluate:
        if not dry_run and not ckpt.is_file():
            print(f"[FAIL] no checkpoint for evaluate: {ckpt}", flush=True)
            ok = False
        else:
            rc = evaluate_one(
                checkpoint=ckpt,
                data_dir=out_test,
                pareto_out=pareto_out,
                hv_out=hv_out,
                pareto_points=pareto_points,
                dry_run=dry_run,
            )
            if rc != 0:
                print(f"[FAIL] evaluate: {name}", flush=True)
                ok = False
    else:
        if skip_evaluate:
            print(f"[SKIP evaluate] {name}", flush=True)

    elapsed = time.perf_counter() - t0
    status = "OK" if ok else "FAILED"
    print(f"[{status}] {name}  ({elapsed:.1f}s)", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Run HRL-VGAE independent experiments on Kaggle"
    )
    # Paths (Kaggle-friendly defaults under /kaggle/working when present)
    default_root = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
    parser.add_argument("--testcase-dir", type=str, default="data/testcase",
                        help="Directory of original testcase JSONs")
    parser.add_argument("--test-root", type=str, default="data/test",
                        help="Root for generated test episodes")
    parser.add_argument("--results-root", type=str,
                        default=str(default_root / "results"),
                        help="Root for pareto_front.json / hv_result.json")
    parser.add_argument("--ckpt-root", type=str,
                        default=str(default_root / "checkpoints"),
                        help="Directory for per-experiment checkpoints")
    parser.add_argument("--log-root", type=str,
                        default=str(default_root / "logs"),
                        help="Directory for training CSVs")

    # Grid filters
    parser.add_argument("--topology", type=str, nargs="*", default=None,
                        choices=list(TOPOLOGIES),
                        help="Subset of topologies (default: all)")
    parser.add_argument("--allocation", type=str, nargs="*", default=None,
                        choices=list(ALLOCATIONS),
                        help="Subset of allocations (default: all)")
    parser.add_argument("--difficulty", type=str, nargs="*", default=None,
                        choices=list(DIFFICULTIES),
                        help="Subset of difficulties (default: all)")

    # Train / generate / evaluate hyperparams
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--passes-per-file", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--num-test-episodes", type=int, default=NUM_TEST_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pareto-points", type=int, default=11)
    parser.add_argument("--pretrained-vgae", type=str, default="")

    # Stage control
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands only, do not execute")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort entire grid on first failed experiment")

    args = parser.parse_args()

    topologies = tuple(args.topology) if args.topology else TOPOLOGIES
    allocations = tuple(args.allocation) if args.allocation else ALLOCATIONS
    difficulties = tuple(args.difficulty) if args.difficulty else DIFFICULTIES

    testcase_dir = Path(args.testcase_dir)
    test_root = Path(args.test_root)
    results_root = Path(args.results_root)
    ckpt_root = Path(args.ckpt_root)
    log_root = Path(args.log_root)

    # Create roots once
    for d in (test_root, results_root, ckpt_root, log_root):
        ensure_dir(d)

    experiments = iter_experiments(topologies, allocations, difficulties)
    print(f"Scheduled experiments: {len(experiments)}", flush=True)
    for t, a, d in experiments:
        print(f"  - {t}_{a}_{d}", flush=True)

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
