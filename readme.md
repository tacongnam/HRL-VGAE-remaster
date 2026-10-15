# HRL-VGAE — Hierarchical RL + Variational Graph Autoencoder for NFV Embedding

Multi-objective **SFC (Service Function Chain) embedding** on substrate networks using a
two-level DQN (High-Level / Low-Level agents) guided by VGAE-encoded graph
representations and Pareto-based scalarization.

This README is a complete, runnable guide covering: **train → test → evaluate →
ablation → plots**, plus the automated all-experiments runner.

---

## Table of contents

1. [Requirements & install](#1-requirements--install)
2. [Project structure](#2-project-structure)
3. [Quick start (TL;DR)](#3-quick-start-tldr)
4. [Core experiment model](#4-core-experiment-model)
5. [Data: topologies & testcases](#5-data-topologies--testcases)
6. [Full manual pipeline (per experiment)](#6-full-manual-pipeline-per-experiment)
7. [Automated runner — run all experiments](#7-automated-runner--run-all-experiments)
8. [Running subsets](#8-running-subsets)
9. [Pre-train the VGAE](#9-pre-train-the-vgae)
10. [Ablation study](#10-ablation-study)
11. [Plots & graphs](#11-plots--graphs)
12. [Output file formats](#12-output-file-formats)
13. [Configuration](#13-configuration)
14. [Legacy / alternative scripts](#14-legacy--alternative-scripts)
15. [Limitations](#15-limitations)
16. [Command cheatsheet](#16-command-cheatsheet)

---

## 1. Requirements & install

- **Python 3.10+**
- **PyTorch** (CUDA optional — the code auto-selects `cuda` if available, else `cpu`)
- **torch-geometric**, **networkx**, **numpy**, **matplotlib**, **tensorboard**

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins: `torch>=2.0`, `torch_geometric>=2.3`, `networkx>=3.1`,
`numpy>=1.24`, `matplotlib>=3.7`, `tensorboard>=2.13`.

> All entry-point scripts assume you run them **from the repository root** (e.g.
> `python train.py ...`, `python data/generate_episodes.py ...`).

---

## 2. Project structure

```
config.py                    # All hyperparameters (NetworkConfig, QNetConfig, ...)
train.py                     # Training entry point (HL/LL DQN + VGAE end-to-end)
evaluate.py                  # Pareto scan / AR-cost screening / hypervolume metrics
pretrain_vgae.py             # Standalone VGAE pre-training on one topology
run_kaggle_experiments.py    # ONE-COMMAND pipeline for all experiments (recommended)
agents/
  hl_agent.py                # High-level agent (SFC selection) + ablation branches
  ll_agent.py                # Low-level agent (node placement) + ablation branches
models/
  vgae.py                    # MN-VGAE encoder (GCN + temporal GRU)
  hl_scorer.py               # HL Q-network
  ll_dqn.py                  # LL Q-network
env/
  nfv_env.py                 # NFV substrate environment (deploy / release / heap)
data/
  loader.py                  # Episode JSON parser (parse_episode, discover_episodes, ...)
  generate_topology.py       # Generate a fixed topology JSON {V,E,F}
  generate_episodes.py       # Generate N episode JSONs (same V/E/F, fresh R)
  generate.py                # Legacy generator (train-file → test episodes)
  prepare_pipeline.py        # Legacy bulk train/test split (shared-train design)
  testcase/                  # 36 independent experiments (see §4)
  topologies/                # Raw topology JSONs
utils/
  pareto.py                  # Pareto front, hypervolume, scalarizer, coverage/spacing/spread
  dijkstra.py                # Custom Dijkstra with bandwidth constraint
  deploy_cost.py             # Deployment cost computation
  metrics.py                 # EpisodeMetrics, MetricsTracker
  logger.py                  # CSV + TensorBoard logging
scripts/
  plot_results.py            # Plot training curves, screening, Pareto front, metrics
```

Default artifact folders (created on demand): `checkpoints/`, `logs/`, `results/`,
`data/test/<topology>/<allocation>/<difficulty>/`, `plots/`, `runs/` (TensorBoard).

---

## 3. Quick start (TL;DR)

**Option A — one command, runs the full pipeline for every experiment:**

```bash
python run_kaggle_experiments.py
```

This trains, generates 10 tests, screens them, and Pareto-evaluates the best one for
each of the 36 testcases (see §7).

**Option B — one experiment, fully manual (train → generate → evaluate → plot):**

```bash
# 1) Train on a single testcase
python train.py \
    --train-file data/testcase/nsf_centers_easy.json \
    --checkpoint checkpoints/nsf_centers_easy.pt \
    --csv logs/nsf_centers_easy.csv \
    --epochs 5 --passes-per-file 1 --warmup-epochs 1

# 2) Generate 10 test episodes (same topology, new requests)
python data/generate_episodes.py \
    --topology data/testcase/nsf_centers_easy.json \
    --num-episodes 10 --infer-cfg --seed 42 \
    --out-dir data/test/nsf/centers/easy

# 3) Evaluate (Pareto scan + hypervolume)
python evaluate.py \
    --pareto-scan \
    --checkpoint checkpoints/nsf_centers_easy.pt \
    --data-dir data/test/nsf/centers/easy \
    --pareto-points 11 \
    --pareto-out results/nsf_centers_easy/pareto_front.json \
    --compute-hv --hv-out results/nsf_centers_easy/hv_result.json

# 4) Plot everything for that experiment
python scripts/plot_results.py \
    --csv logs/nsf_centers_easy.csv \
    --pareto-json results/nsf_centers_hard/pareto_front.json \
    --hv-json results/nsf_centers_easy/hv_result.json \
    --out plots/nsf_centers_easy
```

---

## 4. Core experiment model

**One file in `data/testcase/` = one independent experiment.**

There is **no shared training set** across testcases. Do not train one model on all
originals and then evaluate everywhere. Each testcase is **trained, tested, and
evaluated on its own** (its own checkpoint, its own test episodes, its own metrics).

```
data/testcase/
    ├── <topology>_<allocation>_<difficulty>.json   ← experiment 1
    ├── <topology>_<allocation>_<difficulty>.json   ← experiment 2
    └── ...
```

For each testcase the pipeline is:

```
data/testcase/<topology>_<allocation>_<difficulty>.json
        │
        ├── 1. TRAIN on this file only  (--train-file)
        │        └── checkpoints/<name>.pt
        │
        ├── 2. generate_episodes → 10 NEW test episodes
        │        (same topology V/E/F, new request workloads R)
        │        └── data/test/<topology>/<allocation>/<difficulty>/episode_*.json
        │
        ├── 3. EVALUATE the 10 test episodes (Pareto scan / AR screening)
        │        └── results/<name>/{screen_ar_cost,pareto_front,hv_result,summary}.json
        │
        └── 4. Metrics: Hypervolume, Spread, Spacing, Coverage
```

---

## 5. Data: topologies & testcases

### 5.1 The 36 testcases

`data/testcase/` ships **36** independent experiments. Filenames follow:

```
{topology}_{allocation}_{difficulty}.json
```

| Component   | Values                                       |
|-------------|----------------------------------------------|
| topology    | `cogent`, `conus`, `nsf`                     |
| allocation  | `centers`, `uniform`, `rural`, `urban`       |
| difficulty  | `easy`, `normal`, `hard`                     |

Examples: `nsf_centers_hard.json`, `cogent_urban_easy.json`, `conus_rural_normal.json`.

Parse components by splitting on `_` (topology = first token, difficulty = last token,
allocation = the middle token).

### 5.2 Episode / test JSON schema

Every episode/testcase file has the same shape:

```json
{
  "V": [ { "id": "v0", "cpu_total": ..., "c_v": ..., ... } ],
  "E": [ { "u": "v0", "v": "v1", "b_l": ..., "d_l": ... } ],
  "F": [ { "id": 0, "cpu": ..., ... } ],
  "R": [ { "T": 0.0, "st_r": "v0", "d_r": "v9", "F_r": [..], "b_r": .., "d_max": .. } ]
}
```

- `V` nodes, `E` links, `F` VNF types define the **topology** (shared across episodes).
- `R` is the **request workload** (differs per episode).

### 5.3 Generating a brand-new topology (optional)

`data/generate_topology.py` builds a fixed topology `{V,E,F}` (R is empty). Use it only
if you want a new substrate network beyond the 36 shipped testcases.

```bash
# Real topology from CSV nodes + TXT edges (e.g. nsf)
python data/generate_topology.py \
    --mode file \
    --node-csv data/topology/nsf_nodes.csv \
    --edge-txt data/topology/nsf_edges.txt \
    --server-dist centers --vnf-types 10 --seed 42 \
    --out data/topologies/nsf_centers.json

# Random Barabási–Albert graph (no external files needed)
python data/generate_topology.py \
    --mode random --num-nodes 50 --seed 42 \
    --out data/topologies/random_50.json
```

| Argument        | Default    | Description                                        |
|-----------------|------------|----------------------------------------------------|
| `--mode`        | `random`   | `file` (real topology) or `random` (BA graph)       |
| `--node-csv`    | —          | [file mode] node CSV path                          |
| `--edge-txt`    | —          | [file mode] edge TXT path                          |
| `--server-dist` | `centers`  | [file mode] `uniform`/`urban`/`rural`/`centers`    |
| `--num-nodes`   | 50         | [random mode] node count                           |
| `--vnf-types`   | 10         | Number of VNF types                                |
| `--seed`        | 42         | RNG seed                                           |
| `--out`         | required   | Output topology JSON path                          |

### 5.4 Test data layout (generated)

```
data/test/
└── <topology>/
    └── <allocation>/
        └── <difficulty>/
            ├── episode_0000.json
            ├── episode_0001.json
            ├── ...
            └── episode_0009.json   # 10 episodes (--num-episodes 10)
```

> **Filenames:** `generate_episodes.py` always writes `episode_NNNN.json` (4-digit
> zero-padded). There is no CLI option to rename them. Use the directory hierarchy above
> to identify topology / allocation / difficulty / episode index.

---

## 6. Full manual pipeline (per experiment)

### 6.1 Train on a single testcase

`train.py` trains HL + LL DQNs and the VGAE end-to-end. Pass `--train-file` to train on
**one** file (recommended); omit it to train on every `episode_*.json` in `--train-dir`.

```bash
python train.py \
    --train-file data/testcase/nsf_centers_hard.json \
    --checkpoint checkpoints/nsf_centers_hard.pt \
    --csv logs/nsf_centers_hard.csv \
    --epochs 5 --passes-per-file 5 --warmup-epochs 2
```

| Argument            | Default                     | Description                                        |
|---------------------|-----------------------------|----------------------------------------------------|
| `--train-file`      | —                           | **Single** testcase JSON (recommended)             |
| `--train-dir`       | `data/train`                | Dir of `episode_*.json` (only if no `--train-file`) |
| `--checkpoint`      | `vgae_hrl_ql_checkpoint.pt` | Output checkpoint path                             |
| `--csv`             | `training_log.csv`          | Training log CSV                                   |
| `--log-dir`         | `runs`                      | TensorBoard log directory                          |
| `--epochs`          | 5                           | Training epochs                                    |
| `--passes-per-file` | 5                           | Passes over the file per epoch                     |
| `--warmup-epochs`   | 2                           | Epochs with ε=1.0 (pure exploration, no decay)     |
| `--pretrained-vgae` | —                           | Optional pre-trained VGAE checkpoint (see §9)      |
| `--archive-size`    | 200                         | Pareto archive max size                            |
| `--verbose`         | flag                        | Verbose per-episode debug                          |
| `--ablation`        | `HRL-VGAE`                  | Ablation mode (see §10)                            |

**Output:** checkpoint `checkpoints/<name>.pt`, CSV log, TensorBoard event files.
The checkpoint stores `vgae`, `hl_q`, `ll_q`, `w_accept`, `w_cost`.

> Keep a **separate checkpoint per testcase**. Overwriting one shared checkpoint mixes
> experiments.

**Monitor training:**

```bash
tensorboard --logdir runs
```

### 6.2 Generate 10 test episodes

Use the **same testcase JSON** as the topology source. With `--infer-cfg`, request
parameters (λ, deadlines, VNF lengths, bandwidth) are inferred from the original `R`
list, so difficulty/load stays consistent while requests are new.

```bash
python data/generate_episodes.py \
    --topology data/testcase/nsf_centers_hard.json \
    --num-episodes 10 --infer-cfg --seed 42 \
    --out-dir data/test/nsf/centers/hard
```

| Argument          | Default  | Description                                             |
|-------------------|----------|---------------------------------------------------------|
| `--topology`      | required | Topology/testcase JSON `{V,E,F[,R]}`                    |
| `--num-episodes`  | 50       | Number of episodes → use **10** for this pipeline       |
| `--sim-duration`  | 50       | Simulation duration (time units)                        |
| `--seed`          | 42       | Master seed (deterministic given same args)             |
| `--out-dir`       | required | Output directory                                        |
| `--infer-cfg`     | flag     | Infer request config from existing `R`                  |
| `--scheme`        | `easy`   | Baseline random_rate: easy=0, normal=0.5, hard=1.0      |
| `--arrival-rate`  | —        | Override λ                                              |
| `--segments`      | —        | Override VNF chain length                               |
| `--bw-min/--bw-max` | —      | Override bandwidth request range                        |
| `--delay-ratio-min/max` | 1.0 / 1.2 | Deadline ratio range                              |

Each episode shares topology `{V,E,F}` and gets a fresh `{R}`. Re-running with the same
`--topology` + `--seed` + `--num-episodes` + `--sim-duration` yields identical files.

### 6.3 Evaluate — two modes

#### Mode A — Full Pareto scan (`--pareto-scan`)

Runs the trained model over a **weight grid** `w_accept ∈ [0,1]` and records
`(acc_ratio, deploy_cost)` for every (weight × episode). Use this to build a Pareto front.

```bash
python evaluate.py \
    --pareto-scan \
    --checkpoint checkpoints/nsf_centers_hard.pt \
    --data-dir data/test/nsf/centers/hard \
    --pareto-points 11 \
    --pareto-out results/nsf_centers_hard/pareto_front.json \
    --compute-hv --hv-out results/nsf_centers_hard/hv_result.json
```

#### Mode B — AR/cost screening (`--ar-cost-only`)

Runs each episode **once** using the weights stored in the checkpoint (no weight grid).
Produces one `(acc_ratio, deploy_cost)` per episode — useful for ranking episodes fast.
This is what the automated runner uses to pick the best episode.

```bash
python evaluate.py \
    --ar-cost-only \
    --checkpoint checkpoints/nsf_centers_hard.pt \
    --data-dir data/test/nsf/centers/hard \
    --ar-cost-out results/nsf_centers_hard/screen_ar_cost.json
```

#### Mode C — HV only (reuse an existing scan)

```bash
python evaluate.py \
    --scan-json results/nsf_centers_hard/pareto_front.json \
    --out results/nsf_centers_hard/hv_result.json
```

#### evaluate.py arguments

| Argument           | Default                     | Description                                        |
|--------------------|-----------------------------|----------------------------------------------------|
| `--pareto-scan`    | flag                        | Mode A: full weight-grid scan                      |
| `--ar-cost-only`   | flag                        | Mode B: one run/episode with checkpoint weights    |
| `--checkpoint`     | `vgae_hrl_ql_checkpoint.pt` | Model checkpoint from training                     |
| `--data-dir`       | `data/dataset_01/test`      | Directory containing `episode_*.json`              |
| `--single-episode` | —                           | Evaluate only this one episode file                |
| `--recursive`      | flag                        | Search `*/*/*/episode_*.json` under `--data-dir`   |
| `--pareto-points`  | 11                          | Number of weights (uniform, `w_accept` 0→1)        |
| `--pareto-out`     | `pareto_front.json`         | Pareto scan output                                 |
| `--ar-cost-out`    | (same as pareto-out)        | Output for `--ar-cost-only`                        |
| `--compute-hv`     | flag                        | Compute HV + spacing + spread after scan           |
| `--hv-out`         | `hv_result.json`            | Metrics output file                                |
| `--hv-eps`         | 0.01                        | Epsilon margin for reference point                 |
| `--ablation`       | `HRL-VGAE`                  | Must match the mode used at train time (see §10)   |

---

## 7. Automated runner — run all experiments

`run_kaggle_experiments.py` is the **recommended** way to run the whole benchmark. For
each experiment it does, in order:

1. **Train** on `data/testcase/{topology}_{allocation}_{difficulty}.json`.
2. **Generate** `--num-test-episodes` (default 10) test episodes.
3. **Screen** all episodes with checkpoint weights (`--ar-cost-only`) → AR + cost.
4. **Select** the episode with the highest AR (tie-break: lower cost, then path).
5. **Pareto scan** (`--pareto-points`, default 11) **on that single best-AR episode**
   only, then compute Hypervolume / Spacing / Spread.
6. Write a `summary.json` linking screening + selected episode + metrics.

```bash
# Everything (all 36 testcases)
python run_kaggle_experiments.py

# One topology only
python run_kaggle_experiments.py --topology nsf

# One topology × allocation
python run_kaggle_experiments.py --topology nsf --allocation centers

# One specific experiment
python run_kaggle_experiments.py --topology nsf --allocation centers --difficulty hard
```

| Argument            | Default                | Description                                        |
|---------------------|------------------------|----------------------------------------------------|
| `--topology`        | all                    | Subset: `nsf` `cogent` `conus` (space-separated)   |
| `--allocation`      | all                    | Subset: `centers` `uniform` `rural` `urban`        |
| `--difficulty`      | all                    | Subset: `easy` `normal` `hard`                     |
| `--epochs`          | 10                     | Training epochs                                    |
| `--passes-per-file` | 1                      | Passes per file per epoch                          |
| `--warmup-epochs`   | 2                      | Warmup epochs                                      |
| `--num-test-episodes`| 10                    | Test episodes generated per experiment            |
| `--seed`            | 42                     | Master seed                                        |
| `--pareto-points`   | 11                     | Weight grid size (best-AR episode only)            |
| `--pretrained-vgae` | —                      | Pre-trained VGAE to feed to `train.py`             |
| `--skip-train`      | flag                   | Reuse existing checkpoints                         |
| `--skip-generate`   | flag                   | Reuse existing test episodes                       |
| `--skip-evaluate`   | flag                   | Training only                                      |
| `--dry-run`         | flag                   | Print commands without executing                   |
| `--stop-on-error`   | flag                   | Abort the run on first failure                     |
| `--testcase-dir`    | `data/testcase`        | Source testcases                                   |
| `--test-root`       | `data/test`            | Where test episodes go                             |
| `--results-root`    | `<root>/results`       | Where per-experiment results go                    |
| `--ckpt-root`       | `<root>/checkpoints`   | Where checkpoints go                               |
| `--log-root`        | `<root>/logs`          | Where CSV logs go                                  |

`<root>` is `/kaggle/working` on Kaggle, otherwise the current directory. So on a laptop
results land in `./results/{name}/`.

Per experiment the runner writes:

```
results/{topology}_{allocation}_{difficulty}/
    screen_ar_cost.json     # all tests, 1 weight (ranking input)
    pareto_front.json       # full weight scan on best-AR episode
    hv_result.json          # hypervolume / spacing / spread
    summary.json            # experiment summary (ranking + selected + metrics)
```

> **Ablation note:** `run_kaggle_experiments.py` does **not** expose `--ablation`; it
> always runs the default `HRL-VGAE`. To run ablations at scale, see §10 (loop the
> manual pipeline, or add `--ablation` passthrough to `train_one`/`evaluate_*`).

---

## 8. Running subsets

The fastest way to run a subset is `run_kaggle_experiments.py` with `--topology` /
`--allocation` / `--difficulty` (§7). If you prefer the manual pipeline, filter the file
list in the shell. `train.py` and `evaluate.py` have **no built-in**
`--topology`/`--allocation`/`--difficulty` filters.

Generic loop body (fill in train → generate → evaluate):

```bash
mkdir -p checkpoints logs results
for tc in data/testcase/*.json; do
  base=$(basename "$tc" .json)          # e.g. nsf_centers_hard
  topology=${base%%_*}                   # nsf
  rest=${base#*_}
  difficulty=${rest##*_}                # hard
  allocation=${rest%_*}                 # centers

  # 1) Train
  python train.py --train-file "$tc" \
      --checkpoint "checkpoints/${base}.pt" \
      --csv "logs/${base}.csv" \
      --epochs 5 --passes-per-file 5 --warmup-epochs 2

  # 2) Generate 10 tests
  out_test="data/test/${topology}/${allocation}/${difficulty}"
  python data/generate_episodes.py --topology "$tc" \
      --num-episodes 10 --infer-cfg --seed 42 --out-dir "$out_test"

  # 3) Evaluate
  mkdir -p "results/${base}"
  python evaluate.py --pareto-scan \
      --checkpoint "checkpoints/${base}.pt" --data-dir "$out_test" \
      --pareto-points 11 \
      --pareto-out "results/${base}/pareto_front.json" \
      --compute-hv --hv-out "results/${base}/hv_result.json"
done
```

Change the glob on the `for` line to select a subset:

```bash
# One topology
for tc in data/testcase/nsf_*.json; do ... ; done

# One allocation (note: files use 'centers', not 'center')
for tc in data/testcase/*_centers_*.json; do ... ; done

# One topology × allocation (covers easy/normal/hard)
for tc in data/testcase/nsf_centers_*.json; do ... ; done

# One difficulty only
for tc in data/testcase/*_hard.json; do ... ; done
```

---

## 9. Pre-train the VGAE

`pretrain_vgae.py` trains the **static GCN layers** of the VGAE on one topology
(reconstruction + KL). The temporal GRU is still trained during main training, so
pre-training mainly speeds up convergence and stabilizes the graph encoder.

```bash
python pretrain_vgae.py \
    --episode data/testcase/nsf_centers_hard.json \
    --epochs 100 \
    --lr 1e-3 --neg-ratio 1.0 --seed 42 \
    --out pretrained_vgae.pt
```

| Argument     | Default            | Description                              |
|--------------|--------------------|------------------------------------------|
| `--episode`  | required           | Topology/episode JSON to train on        |
| `--epochs`   | 100                | Pre-training epochs                      |
| `--lr`       | 1e-3               | Adam learning rate (static params only)  |
| `--neg-ratio`| 1.0                | Negative sampling ratio for recon loss   |
| `--seed`     | 42                 | RNG seed                                 |
| `--out`      | `pretrained_vgae.pt`| Output checkpoint                       |

The script keeps the **best-loss** epoch and saves a dict with `vgae` weights +
`pretrain_cfg`. Then feed it to training:

```bash
python train.py --train-file data/testcase/nsf_centers_hard.json \
    --pretrained-vgae pretrained_vgae.pt \
    --checkpoint checkpoints/nsf_centers_hard.pt --csv logs/nsf_centers_hard.csv
```

When `--pretrained-vgae` is given, `train.py` freezes the static layers
(`gcn0.`, `skip.`, `gcn_mu.`, `gcn_logvar.`) and only optimizes the `temporal_cell`
parameters during RL training.

---

## 10. Ablation study

The `--ablation` flag (available on **both** `train.py` and `evaluate.py`) swaps out
individual pipeline components with heuristics to isolate the DRL contribution at each
level. It is set via `config.train.ablation_mode`.

| Ablation         | Request selection (HL)     | VNF placement (LL)        | Routing                    |
|------------------|----------------------------|---------------------------|----------------------------|
| `DefaultBaseline`| By remaining time (EDF)    | Greedy, most-free CPU     | Basic Dijkstra (delay only)|
| `Random-Fit`     | Random request             | DRL low-level policy      | Weighted Dijkstra          |
| `GreedyFIFO`     | Arrival order (FIFO)       | DRL low-level policy      | Weighted Dijkstra          |
| `GreedyLowLevel` | DRL high-level policy      | Greedy, most-free CPU     | Weighted Dijkstra          |
| `HRL-VGAE`       | DRL high-level policy      | DRL low-level policy      | Weighted Dijkstra          |

`HRL-VGAE` is the **full model** and the default. The other four each replace one or two
components.

### What each mode trains

| Ablation         | `ll_agent.train_step()` | `hl_agent.train_step()` |
|------------------|-------------------------|-------------------------|
| `DefaultBaseline`| no                      | no                      |
| `Random-Fit`     | **yes**                 | no                      |
| `GreedyFIFO`     | **yes**                 | no                      |
| `GreedyLowLevel` | no                      | **yes**                 |
| `HRL-VGAE`       | **yes**                 | **yes**                 |

Agents bypass DQN selection **and** training whenever their component is replaced by a
greedy/random strategy.

### Where the swaps happen (code reference)

- **Request selection** — `agents/hl_agent.py` (`HLAgent.select_sfc`), runs **before**
  ε-greedy: `DefaultBaseline`→ min `deadline-t`; `Random-Fit`→ random; `GreedyFIFO`→
  min `arrival_time`.
- **VNF placement** — `agents/ll_agent.py` (`LLAgent.select_node`), also before ε-greedy:
  `DefaultBaseline`/`GreedyLowLevel`→ `max(valid_nodes, key=cpu_free)`.
- **Routing** — `train.py` passes `lambda_penalty = 0.0` for `DefaultBaseline`, so
  Dijkstra minimizes delay only ("Basic Dijkstra"). The hard constraint
  `bw_free ≥ req_bw` is kept in all modes; only the soft congestion penalty is disabled.

### Running an ablation

You **must** pass the same `--ablation` to `train.py` and `evaluate.py`. Keep ablation
checkpoints separate from the full-model checkpoint.

```bash
# Train an ablation variant
python train.py \
    --train-file data/testcase/nsf_centers_hard.json \
    --ablation GreedyLowLevel \
    --checkpoint checkpoints/nsf_centers_hard_GreedyLowLevel.pt \
    --csv logs/nsf_centers_hard_GreedyLowLevel.csv

# Evaluate with the SAME mode
python evaluate.py --pareto-scan --ablation GreedyLowLevel \
    --checkpoint checkpoints/nsf_centers_hard_GreedyLowLevel.pt \
    --data-dir data/test/nsf/centers/hard --pareto-points 11 \
    --pareto-out results/nsf_centers_hard_GreedyLowLevel/pareto_front.json \
    --compute-hv --hv-out results/nsf_centers_hard_GreedyLowLevel/hv_result.json
```

Loop over all five modes for one testcase:

```bash
for mode in DefaultBaseline Random-Fit GreedyFIFO GreedyLowLevel HRL-VGAE; do
  python train.py --train-file data/testcase/nsf_centers_hard.json \
      --ablation "$mode" --epochs 5 --passes-per-file 5 --warmup-epochs 2 \
      --checkpoint "checkpoints/nsf_centers_hard_${mode}.pt" \
      --csv "logs/nsf_centers_hard_${mode}.csv"
  python evaluate.py --pareto-scan --ablation "$mode" \
      --checkpoint "checkpoints/nsf_centers_hard_${mode}.pt" \
      --data-dir data/test/nsf/centers/hard --pareto-points 11 \
      --pareto-out "results/nsf_centers_hard_${mode}/pareto_front.json" \
      --compute-hv --hv-out "results/nsf_centers_hard_${mode}/hv_result.json"
done
```

**Comparing two fronts (Coverage C-metric):** coverage has **no CLI flag**. Call
`_compute_hv` from Python and pass a baseline `pareto_front.json`:

```python
from evaluate import _compute_hv
_compute_hv(
    scan_json="results/nsf_centers_hard_HRL-VGAE/pareto_front.json",
    out="results/nsf_centers_hard_HRL-VGAE/hv_result.json",
    eps=0.01,
    baseline_json="results/nsf_centers_hard_GreedyLowLevel/pareto_front.json",
)
# → coverage_ab : fraction of B's front dominated by A
# → coverage_ba : fraction of A's front dominated by B
```

---

## 11. Plots & graphs

`scripts/plot_results.py` reads the JSON/CSV artifacts and writes PNGs. It uses the
non-interactive `Agg` backend (safe on headless servers / Kaggle).

```bash
# Everything for one experiment
python scripts/plot_results.py \
    --csv logs/nsf_centers_hard.csv \
    --screen-json results/nsf_centers_hard/screen_ar_cost.json \
    --pareto-json results/nsf_centers_hard/pareto_front.json \
    --hv-json results/nsf_centers_hard/hv_result.json \
    --summary-json results/nsf_centers_hard/summary.json \
    --out plots/nsf_centers_hard

# Whole results/ tree (one subfolder per experiment), plus training CSVs
python scripts/plot_results.py \
    --results-root results --logs-root logs --out plots
```

| Argument         | Description                                             |
|------------------|---------------------------------------------------------|
| `--csv`          | Training log CSV (→ training-curve PNGs)                |
| `--screen-json`  | `screen_ar_cost.json` (→ screening bar/scatter PNGs)    |
| `--pareto-json`  | `pareto_front.json` (→ Pareto-front PNG)                |
| `--hv-json`      | `hv_result.json` (metrics overlay + metrics card)       |
| `--summary-json` | `summary.json` (selected-episode info in metrics card)  |
| `--results-root` | Plot **every** experiment folder under this directory   |
| `--logs-root`    | With `--results-root`: find `{name}.csv` here for curves|
| `--out`          | Output directory (default `plots`)                      |
| `--window`       | Smoothing window for training curves (default 20)       |

### Outputs produced

**Training curves** (from `--csv`, one PNG per metric + an `overview.png` grid):
`acceptance_ratio.png`, `revenue.png`, `deploy_cost.png`, `hl_reward.png`,
`ll_reward.png`, `vgae_loss.png`, `hl_loss.png`, `ll_loss.png`, `link_util.png`,
`cpu_util.png`, `epsilon.png`, `pareto_weight.png`, `overview.png`.

**Screening** (from `--screen-json`): `screen_ar.png` (AR per episode, selected one
highlighted), `screen_cost.png` (cost per episode), `screen_scatter.png` (AR vs cost).

**Pareto** (from `--pareto-json` [+ `--hv-json`]): `pareto_front.png` — scatter of
`(deploy_cost, acceptance_ratio)` colored by `w_accept`, the non-dominated front drawn,
and a text box with HV / Spacing / Spread / reference point when `--hv-json` is given.

**Metrics card** (from `--hv-json` / `--summary-json`): `metrics_card.png` — a table of
Hypervolume / Spacing / Spread / front size (+ coverage if present) and the selected
best-AR episode.

---

## 12. Output file formats

**`pareto_front.json`** — one row per (weight, episode) in the scan:

```json
[
  { "w_accept": 0.0, "w_cost": 1.0, "acc_ratio": 0.82,
    "deploy_cost": 145.3, "path": "data/test/nsf/centers/hard/episode_0000.json" }
]
```

**`screen_ar_cost.json`** — one row per episode (checkpoint weights):

```json
[
  { "w_accept": 0.5, "w_cost": 0.5, "acc_ratio": 0.81, "deploy_cost": 150.2,
    "accepted": 81, "rejected": 19,
    "path": "data/test/nsf/centers/hard/episode_0000.json" }
]
```

**`hv_result.json`** — front quality metrics:

```json
{
  "hypervolume": 0.1234,
  "spacing": 0.0456,
  "delta": 0.3210,
  "spread": 0.3210,
  "reference_point": { "acc_ratio": -0.01, "neg_deploy_cost": -145.31 },
  "pareto_front": [ { "acc_ratio": 0.9, "deploy_cost": 120.5 } ],
  "all_points": [ ... ]
}
```

When a baseline is supplied to `_compute_hv`, it also contains `coverage_ab`,
`coverage_ba`, and `coverage`.

**`summary.json`** (runner only) — experiment name, all screening rows, the episode
selected for Pareto, and the metrics block.

---

## 13. Configuration

All hyperparameters live in `config.py` (dataclasses). **There is no CLI override** for
config values — edit the file to change them.

| Config class      | Key parameters                                                                 |
|-------------------|--------------------------------------------------------------------------------|
| `NetworkConfig`   | `num_nodes=50`, `cpu_capacity_mips=8000`, `bw_min/max_mbps`, `delay_min/max_ms` |
| `SFCConfig`       | `arrival_rate=20`, `vnf_len_min/max=2/8`, `deadline_min/max=20/100`            |
| `VGAEConfig`      | `d_latent=32`, `d_hidden=64`, `temporal=True`, `temporal_hidden=64`            |
| `QNetConfig`      | `lr=5e-4`, `gamma=0.95`, `eps_start=1.0`, `eps_end=0.05`, `batch_size=64`      |
| `RewardConfig`    | `mu_cpu/mu_bw=0.001`, `omega_bw=100`, `lambda_penalty=1.0`, `r_fail=10`         |
| `DeployCostConfig`| `w_cpu/w_ram/w_storage/w_bw/w_init`, `init_cost_per_hop=0.5`                   |
| `ParetoConfig`    | `num_weight_bins=11`, `chebyshev_rho=0.05`, HV reference components `-25.0`     |
| `TrainConfig`     | `seed=42`, `train_interval_sfc=4`, `k_hop_explore=3`, `ablation_mode="HRL-VGAE"` |

---

## 14. Legacy / alternative scripts

- **`data/generate.py`** — older generator: `--train-file` → N test episodes in
  `--out-dir`, inferring request config from the train file's `R`. Superseded by
  `generate_episodes.py` for the independent-experiment pipeline.
- **`data/prepare_pipeline.py`** — builds a **shared-train** split under
  `data/all_tests/{topology}/{allocation}/{difficulty}/` (trains on 12 hard originals,
  generates per-difficulty tests). This does **not** match the required
  independent-experiment model in §4. Use it only if you intentionally want the
  shared-train design.

---

## 15. Limitations

Documented for transparency (this README does not modify source code):

1. **No CLI filters** for topology / allocation / difficulty on `train.py` /
   `evaluate.py`. Use `run_kaggle_experiments.py` or filter the file list in the shell.
2. **`generate_episodes.py` always writes `episode_NNNN.json`** (no rename option).
   Identify experiments by the directory hierarchy.
3. **Coverage metric** exists only as the `baseline_json=` argument of `_compute_hv()`;
   there is **no** CLI flag. Compare models by calling `_compute_hv` from Python (§10).
4. **`--pareto-scan` metrics are aggregated** over all episodes × weights in
   `--data-dir`, not reported per-episode. Use `--single-episode` (or the runner, which
   already scans a single best-AR episode) for per-episode fronts.
5. **`run_kaggle_experiments.py` now exposes `--ablation`** (shared by `train.py` and
   `evaluate.py`). Default `HRL-VGAE` keeps the original output paths; every other mode
   appends a `_<ablation>` suffix to checkpoint / csv / results so runs don't collide.

---

## 16. Command cheatsheet

```bash
# --- one experiment, end to end ---
python train.py --train-file data/testcase/nsf_centers_hard.json \
    --checkpoint checkpoints/nsf_centers_hard.pt --csv logs/nsf_centers_hard.csv \
    --epochs 5 --passes-per-file 5 --warmup-epochs 2

python data/generate_episodes.py --topology data/testcase/nsf_centers_hard.json \
    --num-episodes 10 --infer-cfg --seed 42 --out-dir data/test/nsf/centers/hard

python evaluate.py --pareto-scan --checkpoint checkpoints/nsf_centers_hard.pt \
    --data-dir data/test/nsf/centers/hard --pareto-points 11 \
    --pareto-out results/nsf_centers_hard/pareto_front.json \
    --compute-hv --hv-out results/nsf_centers_hard/hv_result.json

python scripts/plot_results.py --results-root results --logs-root logs --out plots

# --- all experiments (recommended) ---
python run_kaggle_experiments.py
python run_kaggle_experiments.py --topology nsf --allocation centers --difficulty hard

# --- run an ablation via the runner (train + evaluate share the mode) ---
python run_kaggle_experiments.py --ablation GreedyLowLevel
python run_kaggle_experiments.py --ablation Random-Fit --topology nsf

# --- optional: pre-train VGAE first ---
python pretrain_vgae.py --episode data/testcase/nsf_centers_hard.json \
    --epochs 100 --out pretrained_vgae.pt

# --- ablation (train + evaluate must share --ablation) ---
python train.py --train-file data/testcase/nsf_centers_hard.json --ablation GreedyLowLevel \
    --checkpoint checkpoints/x_GreedyLowLevel.pt --csv logs/x_GreedyLowLevel.csv
python evaluate.py --pareto-scan --ablation GreedyLowLevel \
    --checkpoint checkpoints/x_GreedyLowLevel.pt --data-dir data/test/nsf/centers/hard \
    --pareto-points 11 --pareto-out results/x_GreedyLowLevel/pareto_front.json \
    --compute-hv --hv-out results/x_GreedyLowLevel/hv_result.json

# --- TensorBoard ---
tensorboard --logdir runs
```
