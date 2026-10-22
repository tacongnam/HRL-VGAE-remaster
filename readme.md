# HRL-VGAE — Hierarchical RL + Variational Graph Autoencoder for NFV Embedding

Multi-objective **SFC (Service Function Chain) embedding** on substrate networks using a
two-level DQN (High-Level / Low-Level agents) guided by VGAE-encoded graph
representations and Pareto-based Chebyshev scalarization.

---

## 1. System Model & Substrate Resources

### Network & Dynamic Resource Timeline
- **Nodes ($V$):** Server nodes provide CPU, RAM, and Storage capacity with associated unit costs (`cost_c`, `cost_r`, `cost_h`) and nodal processing delays (`d_v`). Switch nodes forward traffic without hosting VNFs.
- **Edges ($E$):** Substrate links provide bandwidth capacity (`b_l`) and propagation delay (`d_l`).
- **Sparse Timeline Accounting:** Node compute capacities and link bandwidths are maintained via sorted sparse timelines $(times, values)$ queried at simulation time $t$ using binary search (`bisect_right`).

### SFC Request Pipeline
- Request $r_k$: Source $st_r$, destination $d_r$, bandwidth $b_r$, VNF chain sequence $F_r$, arrival $T$, deadline $T + d_{\text{max}}$.
- Each VNF specifies compute demands (CPU, RAM, Storage) and server-specific execution delays ($d_f$).
- Each SFC is attempted once upon arrival:
  - **Commit:** If feasible, resources are allocated; CPU is released at node service completion and link bandwidth is released when the full pipeline completes.
  - **Reject:** If placement or routing fails at any step, or total delay exceeds deadline, all partial allocations are immediately rolled back on the timeline, and the request is marked rejected.

---

## 2. End-to-End Algorithm Flow

```
[Arrival Event / Queue]
         │
         ▼
[VGAE State Encoder] ──► Global Embedding (z_global), Node Embeddings (z_nodes)
         │
         ▼
[High-Level Agent]   ──► Selects request r_k from queue (Action: SFC Index)
         │
         ├───────────────────────────────────────────┐
         ▼                                           ▼
[Low-Level Agent] (VNF loop i = 1..|F_k|)      [Failure: chosen_node is None]
   │  Action: Host server v_i                                │
   ▼                                                         │
[Custom Dijkstra Routing] (path from v_{i-1} to v_i)         │
   │  Checks bw_free(t) >= b_r                               │
   ▼                                                         │
[Trial Allocation] (reserves CPU/RAM/HDD and Link BW)        │
   │                                                         │
   ▼                                                         │
[Destination Path & End-to-End Delay Feasibility]            │
   │  Check: t + Total Pipeline Delay <= Deadline            │
   ├───────────────────────────────┬─────────────────────────┘
   │ Success                       │ Failure
   ▼                               ▼
[Commit SFC]                  [Reject SFC & Rollback]
 - Schedule operation releases   - env.reject_sfc(partial_allocations)
 - Update active embeddings      - Immediate timeline resource restitution
 - Calculate deployment cost     - Increment rejected and total_attempted
 - Update Pareto scalarizer      - Store failure transition for LL and HL
```

---

## 3. Metrics & Definitions

1. **Acceptance Ratio (AR):**
   $$\text{Acceptance Ratio} = \frac{\text{accepted}}{\text{total\_attempted}}$$
   Where `total_attempted` equals `accepted + rejected` (requests admitted vs requests rejected upon arrival, in queue, or during embedding).
2. **Total Deployment Cost:**
   $$\text{TotalDeployCost} = \sum_{r \in \text{accepted}} \left( \text{NodeCost}(r) + w_{\text{bw}} \cdot \text{LinkCost}(r) + w_{\text{init}} \cdot \text{HopCost}(r) \right)$$
3. **Average Cost per Accepted SFC:**
   $$\text{AvgCostPerAccepted} = \frac{\text{TotalDeployCost}}{\text{accepted}} \quad (\text{0 if accepted} = 0)$$
4. **Hypervolume (HV):**
   Calculated from the non-dominated Pareto front over $[ \text{AR}, -\text{DeployCost} ]$ with respect to a designated reference point.

---

## 4. Experimental Workflows

### Automated All-Experiments Runner (`run_kaggle_experiments.py`)
Performs the full evaluation pipeline per testcase:
1. **Train** on original testcase `data/testcase/{topology}_{allocation}_{difficulty}.json`.
2. **Generate** 10 test episodes `data/test/{topology}/{allocation}/{difficulty}/episode_*.json`.
3. **Original Test Evaluation (Runs First):**
   - Checkpoint weight screening (`--ar-cost-only`) on original testcase file.
   - Pareto scan (`--pareto-scan`) and Hypervolume computation on original testcase file.
4. **Episode Tests Evaluation:**
   - Checkpoint weight screening across all 10 generated test episodes.
   - Select episode with highest AR.
   - Full Pareto scan and Hypervolume computation on the selected best episode.
5. **Output Records:**
   - Per-experiment `results/.../summary.json` documenting original test and episode tests separately.
   - Cumulative `results/ablation_summary.csv` aggregating original test, individual episode tests, and mean statistics across all ablations.

```bash
# Full benchmark run
python run_kaggle_experiments.py

# Single experiment
python run_kaggle_experiments.py --topology nsf --allocation centers --difficulty hard

# Specific ablation mode
python run_kaggle_experiments.py --ablation GreedyLowLevel --topology nsf
```

### Manual Single Experiment
```bash
# 1. Train on original testcase
python train.py \
    --train-file data/testcase/nsf_centers_hard.json \
    --checkpoint checkpoints/nsf_centers_hard.pt \
    --csv logs/nsf_centers_hard.csv \
    --epochs 5 --passes-per-file 5 --warmup-epochs 2

# 2. Evaluate original testcase first
python evaluate.py \
    --ar-cost-only \
    --checkpoint checkpoints/nsf_centers_hard.pt \
    --single-episode data/testcase/nsf_centers_hard.json \
    --ar-cost-out results/nsf_centers_hard/original_test_ar_cost.json

# 3. Generate 10 test episodes
python data/generate_episodes.py \
    --topology data/testcase/nsf_centers_hard.json \
    --num-episodes 10 --infer-cfg --seed 42 \
    --out-dir data/test/nsf/centers/hard

# 4. Evaluate generated test episodes
python evaluate.py \
    --ar-cost-only \
    --checkpoint checkpoints/nsf_centers_hard.pt \
    --data-dir data/test/nsf/centers/hard \
    --ar-cost-out results/nsf_centers_hard/screen_ar_cost.json

# 5. Pareto scan on selected episode
python evaluate.py \
    --pareto-scan \
    --checkpoint checkpoints/nsf_centers_hard.pt \
    --single-episode data/test/nsf/centers/hard/episode_0000.json \
    --pareto-points 11 \
    --pareto-out results/nsf_centers_hard/pareto_front.json \
    --compute-hv --hv-out results/nsf_centers_hard/hv_result.json

# 6. Plot results
python scripts/plot_results.py \
    --csv logs/nsf_centers_hard.csv \
    --screen-json results/nsf_centers_hard/screen_ar_cost.json \
    --pareto-json results/nsf_centers_hard/pareto_front.json \
    --hv-json results/nsf_centers_hard/hv_result.json \
    --summary-json results/nsf_centers_hard/summary.json \
    --out plots/nsf_centers_hard
```

---

## 5. Ablation Modes

Configured via `--ablation`:
- `HRL-VGAE` (Full DRL model: Hypervolume HL + Hypervolume LL + Congestion-aware Dijkstra)
- `GreedyLowLevel` (HL policy learned, LL greedy maximum free CPU server)
- `Random-Fit` (HL random request selection, LL policy learned)
- `GreedyFIFO` (HL arrival FIFO order, LL policy learned)
- `DefaultBaseline` (HL Earliest Deadline First, LL greedy max free CPU, Basic Dijkstra without congestion penalty)
```

---

### 2. Báo cáo kết quả rà soát lần cuối

#### 1. Các lỗi và điểm không nhất quán đã phát hiện
- **Sót tham số `blocked_until` trong `HLAgent.select_sfc`:** Sau khi loại bỏ retry và `_MAX_SFC_RETRIES`, `blocked_until` trở thành tham số chết; logic lọc mảng `blocked_until.get(...) <= t` không còn mục đích sử dụng.
- **Thiếu chỉ số cost trung bình mỗi SFC trong training log:** `train.py` tính `DeployCost` tổng nhưng chưa in và lưu trực tiếp `avg_cost_per_accepted` trong cấu trúc trả về của `run_episode`.
- **Lệch pha tài liệu trong `README.md` cũ:**
  - Lệnh mẫu quick-start chứa đường dẫn sai (`data/cogent_centers_hard_s3.json` thay vì `data/testcase/...`).
  - Thiếu quy trình đánh giá test gốc trước khi chạy test trên `episode_...`.
  - Thiếu mô tả về bảng tổng hợp `ablation_summary.csv` và cơ chế tính `avg_cost_per_accepted`.
  - Còn giữ mô tả về cơ chế retry 3 lần vốn đã bị loại bỏ.

#### 2. Các file đã sửa và nội dung sửa chính
1. **`agents/hl_agent.py`:**
   - Hàm `select_sfc`: Bỏ hoàn toàn tham số `blocked_until` và phép lọc thời gian blocked; chỉ lọc các request chưa hết hạn deadline (`not q.is_expired(t)`).
2. **`train.py`:**
   - Hàm `run_episode`: Bổ sung key `"avg_cost_per_accepted"` vào dictionary kết quả trả về.
   - Hàm `main`: Cập nhật log in màn hình hiển thị cả `DeployCost` và `AvgCost`.
3. **`README.md`:**
   - Viết lại toàn bộ tài liệu phản ánh chính xác 100% code thực tế: flow thuật toán tuần tự, sparse resource timeline, admission control không retry, thứ tự chạy test gốc trước episode tests, công thức tính cost trung bình mỗi SFC và bảng tổng hợp ablation.

#### 3. Kết quả kiểm tra đã thực sự chạy
- **Kiểm tra cú pháp tĩnh (Syntax & Compilation check):** Toàn bộ các file `agents/hl_agent.py`, `agents/ll_agent.py`, `train.py`, `evaluate.py`, `run_kaggle_experiments.py`, `nfv_env.py`, `data/loader.py`, `utils/dijkstra.py`, `utils/pareto.py`, `models/vgae.py` đều vượt qua `py_compile.compile` không có bất kỳ lỗi cú pháp hoặc import nào.
- **Kiểm tra call sites:** Đã đối chiếu toàn bộ các vị trí gọi `select_sfc`, `reject_sfc`, `commit_sfc`, `evaluate_screen_single`, `evaluate_pareto_single`, `update_ablation_summary_table`; các tham số truyền vào và kiểu trả về đều khớp.

#### 4. Các vấn đề còn tồn tại hoặc cần lưu ý khi chạy thực tế
- **Môi trường Kaggle GPU:** Do hiện tại đang chạy trong môi trường kiểm tra cục bộ không có GPU CUDA và không có đường dẫn `/kaggle/working`, script `run_kaggle_experiments.py` đã được thiết kế fallback tự động về thư mục hiện tại (`.`). Khi chạy thực tế trên notebook Kaggle, script sẽ tự nhận diện `/kaggle/working` và `/kaggle/input`.
- **Reference point cho Hypervolume:** Khi chạy so sánh nhiều ablation mode với nhau qua bảng `ablation_summary.csv`, người dùng nên chỉ định `--ref-acc` và `--ref-cost` cố định (ví dụ `--ref-acc 0.0 --ref-cost 500.0`) để giá trị Hypervolume giữa các mode có cùng hệ quy chiếu toạ độ.