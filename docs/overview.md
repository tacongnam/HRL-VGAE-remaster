# Tổng quan hệ thống DRL-NFV

## Bài toán cần giải quyết

Hãy tưởng tượng một mạng viễn thông lớn gồm 50 máy chủ (node) được kết nối với nhau. Mỗi ngày có hàng nghìn yêu cầu dịch vụ mạng đến — ví dụ một cuộc gọi video cần đi qua lần lượt các phần mềm: mã hóa → nén → kiểm tra an ninh → giải mã. Chuỗi phần mềm như vậy gọi là **Service Function Chain (SFC)**.

Người vận hành mạng cần quyết định:

1. Có chấp nhận xử lý yêu cầu này không?
2. Nếu có, đặt từng phần mềm (VNF) lên máy chủ nào?
3. Truyền dữ liệu giữa các máy chủ theo đường đi nào?

Đây là bài toán **NFV Resource Management** (Quản lý tài nguyên mạng ảo hóa chức năng). Nó khó vì:

- Không gian quyết định cực lớn: 50 node × 2–8 VNF × hàng trăm yêu cầu đồng thời.
- Nhiều ràng buộc phải thỏa mãn cùng lúc: CPU, băng thông, độ trễ, thời hạn.
- Nhiều mục tiêu mâu thuẫn: chấp nhận nhiều yêu cầu nhất có thể nhưng chi phí triển khai phải thấp, tải mạng phải cân bằng, độ trễ phải nhỏ.
- Môi trường thay đổi liên tục: yêu cầu đến ngẫu nhiên, tài nguyên giải phóng và được chiếm dụng theo thời gian.

Hệ thống DRL-NFV giải quyết bài toán này bằng **học tăng cường sâu (Deep Reinforcement Learning)** kết hợp với **tối ưu đa mục tiêu Pareto**.

---

## Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│                    Môi trường mạng (NFV Env)                │
│   50 node, ~150 cạnh, hàng nghìn SFC request               │
└──────────────────────┬──────────────────────────────────────┘
                       │ node features [50 × 16]
                       │ edge_index
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  VGAE (Bộ mã hóa đồ thị)                   │
│   GCNConv × 2 + Skip → μ, σ → z_nodes [50 × 32]           │
│   GRUCell → z_global [64]  (trạng thái mạng tổng hợp)     │
└──────────────────────┬──────────────────────────────────────┘
                       │ z_nodes, z_global
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              HRL — Học tăng cường phân cấp                  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  HL Agent — Chọn SFC từ hàng đợi                    │  │
│  │  Input: z_global [64] + sfc_feat [7] + w [2]        │  │
│  │  Output: Q-set per SFC → HV selection               │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │ SFC được chọn                     │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  LL Agent — Chọn node đặt từng VNF (k = 1..K)       │  │
│  │  Input: z_global + z_prev + z_node + vnf + sfc + w  │  │
│  │  Output: Q-set per node → HV selection              │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │ chosen_node                       │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  Dijkstra Routing                                    │  │
│  │  Tìm đường từ prev_node → chosen_node               │  │
│  │  Trọng số: delay + penaty(băng thông bị tắc)        │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │ path                              │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  Commit / Rollback                                   │  │
│  │  Thành công → chiếm CPU/RAM/HDD + băng thông        │  │
│  │              và lên lịch giải phóng theo operation  │  │
│  │  Thất bại   → hoàn trả tài nguyên                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Luồng xử lý một SFC request

```
Bước 1  Tại mỗi timestep t:
         VGAE encode đồ thị mạng hiện tại
         → z_nodes [N×32], z_global [64]

Bước 2  HL Agent nhìn vào hàng đợi SFC
         Với mỗi SFC hợp lệ (chưa hết hạn):
           tính Q-set = {q_vec ∈ ℝ²}
         Chọn SFC có HyperVolume(Q-set) lớn nhất

Bước 3  LL Agent đặt từng VNF (k = 1, 2, ..., K):
         Với mỗi node còn đủ CPU (và đủ RAM/HDD nếu có yêu cầu):
           tính Q-set = {q_vec ∈ ℝ²}
         Chọn node có HyperVolume(Q-set) lớn nhất
         Chạy Dijkstra từ node trước → node vừa chọn

Bước 4  Sau khi đặt xong tất cả K VNF:
         Nếu thành công → commit tài nguyên (đưa sự kiện giải phóng
                          từng operation vào min-heap `_op_heap`), nhận reward
         Nếu thất bại (không đường đi, không đủ CPU/RAM/HDD)
           → rollback tất cả, nhận failure penalty [-10, -10]

Bước 5  Cập nhật replay buffer và huấn luyện các mạng
```

Lưu ý về đồng hồ: trong `run_episode` (`train.py`), timestep `t` **không** tự tiến sau mỗi SFC — nó chỉ tiến (`env.step_time()`) khi hàng đợi rỗng hoặc mọi SFC còn lại đang bị `blocked_until`. VGAE cũng không encode lại sau mỗi VNF; nó dùng lại cache `z_nodes`/`z_global` cho tới khi `graph_dirty=True` và `step_time()` được gọi (xem `docs/vgae.md` mục 9). Ngay khi `step_time()` chạy, `_release_expired_active_sfcs()` được gọi **trước tiên** để trả tài nguyên cho mọi operation đã hoàn thành — xem mục "Giải phóng tài nguyên theo từng operation" bên dưới.

---

## Các thành phần chính và file tương ứng

| Thành phần | File | Vai trò |
|---|---|---|
| Môi trường mạng | `env/nfv_env.py` | Quản lý tài nguyên node (CPU/RAM/HDD) và link (băng thông), commit/rollback, giải phóng theo operation qua min-heap `_op_heap` |
| Sinh đồ thị & SFC | `utils/graph_generator.py` | Barabási-Albert topology, SFC ngẫu nhiên |
| Bộ mã hóa đồ thị | `models/vgae.py` | GCN + VAE + GRU temporal |
| HL Q-network | `models/hl_scorer.py` | 2 output heads (`N_OBJ_HL = 2`), input dim 73 (`cfg.d_hl_input`) |
| LL Q-network | `models/ll_dqn.py` | 2 output heads (`N_OBJ = 2`), input dim 139 (`cfg.d_ll_input`) |
| HL Agent | `agents/hl_agent.py` | Q-set, HV selection, replay buffer |
| LL Agent | `agents/ll_agent.py` | Q-set, HV selection, replay buffer |
| Routing | `utils/dijkstra.py` | Custom Dijkstra với bandwidth penalty |
| Chi phí triển khai | `utils/deploy_cost.py` | CPU + RAM + storage + bandwidth |
| Pareto & HV | `utils/pareto.py` | dominance, HV, pruning, target Q-set |
| Cấu hình | `config.py` | Tất cả hyperparameter |
| Vòng lặp training | `train.py` | Episode loop, reward, logging |
| Đánh giá | `evaluate.py` | Inference, Pareto scan |
| Metrics/Logger | `utils/metrics.py`, `utils/logger.py` | Thu thập & ghi CSV/TensorBoard |

**Lưu ý về import path:** `train.py` và `evaluate.py` import các module bằng tên phẳng, ví dụ `from nfv_env import NFVEnvironment`, `from vgae import MNVGAE`, `from hl_agent import HLAgent`, `from ll_agent import LLAgent`, `from dijkstra import ...`, `from deploy_cost import ...`, `from pareto import ...`, `from ll_dqn import N_OBJ`, `from metrics import MetricsTracker`, `from logger import TrainingLogger` — **không** dùng path `env.`, `models.`, `utils.`, `agents.` dù file thực tế nằm trong các thư mục con đó (đã xác nhận qua nguồn `env/nfv_env.py`, `models/vgae.py`, `models/hl_scorer.py`, `models/ll_dqn.py`, `agents/hl_agent.py`, `agents/ll_agent.py`, `utils/dijkstra.py`, `utils/deploy_cost.py`, `utils/pareto.py`, `utils/metrics.py`, `utils/logger.py`). Điều này chỉ chạy được nếu các thư mục con nằm trong `sys.path` (mỗi thư mục có `__init__.py` rỗng hoặc thư mục được thêm trực tiếp vào `PYTHONPATH`) — `train.py` chỉ tự thêm `os.path.dirname(__file__)` (thư mục gốc) vào `sys.path`, không thêm từng thư mục con. Ngược lại, `evaluate.py` import đúng theo package con: `from env.nfv_env import NFVEnvironment`, `from models.vgae import MNVGAE`, `from agents.hl_agent import HLAgent`, `from agents.ll_agent import LLAgent`, `from utils.pareto import ...`, và gọi lại `from train import run_episode, set_seeds` — tức là **`train.py` và `evaluate.py` dùng hai kiểu import khác nhau cho cùng codebase**, đây là điểm không nhất quán cần lưu ý khi chạy thực tế (có thể do `train.py` được thiết kế chạy độc lập với các file phẳng ở thư mục gốc, trong khi `evaluate.py` chạy theo cấu trúc package).

---

## Các mục tiêu tối ưu (2 objectives)

**[CẬP NHẬT]** Hệ thống cùng lúc tối ưu **2 mục tiêu**, không phải 4, tất cả theo hướng **maximize** (giá trị càng cao càng tốt). Vector này được `train.py` dựng qua `_build_reward_vec_2obj(r_perf, r_cost)`:

| Index | Tên | Công thức thực tế trong `train.py` | Ý nghĩa |
|---|---|---|---|
| 0 | `r_acc_c` (HL) / `r_quality` (LL) | HL: `revenue + θ·urgency + λ_L·r̄_quality`, với `revenue = (μ_cpu·total_cpu + μ_bw·bw·(F_k+1))/F_k`; LL: `r_quality_k = −(α·path_cost + β·path_delay + γ_load·load_std)` | Maximize doanh thu + mức khẩn + chất lượng đường đi (giảm trễ/tắc nghẽn) |
| 1 | `r_cost_c` | `-cfg.reward.mu_deploy_cost * deploy_cost` (SFC thành công) hoặc `-μ_deploy_cost * cost_per_step` cho từng transition LL, với `cost_per_step = deploy_cost / (F_k + 1)` | Minimize chi phí triển khai |

Khi SFC/VNF thất bại, cả 2 thành phần dùng `cfg.pareto.failure_penalty_vector()` = **`[-10, -10]`** (mặc định `failure_penalty_cost = failure_penalty_success = 10.0`), không phải công thức trên.

**[CẬP NHẬT] Những gì đã thay đổi so với bản 4 mục tiêu:**

- Số head của mỗi Q-network giảm từ 4 xuống **2**: `N_OBJ_HL = 2` (`models/hl_scorer.py`), `N_OBJ = 2` (`models/ll_dqn.py`).
- `r̄_quality` **không còn** là mục tiêu độc lập — nó được gộp vào thành phần 0 qua `λ_L` ở tầng HL, và ở tầng LL nó **chính là** thành phần 0.
- `−load_std` (cân bằng tải) **không còn** là mục tiêu riêng; `load_std` chỉ ảnh hưởng gián tiếp qua `r_quality_seg`.
- Cờ `r_success = +1.0` **không còn** — tín hiệu "đặt thành công" chỉ còn mã hóa gián tiếp qua `r_acc_c > 0` và `done = 1.0`.
- `ParetoConfig.hv_ref_delay`, `hv_ref_balance`, `failure_penalty_delay`, `failure_penalty_balance` trở thành **tham số chết** — `hv_reference_point()` chỉ đọc `hv_ref_cost`/`hv_ref_success`.
- `compute_hypervolume`, `dominates`, `non_dominated_indices` trong `utils/pareto.py` giờ **chỉ đúng cho không gian 2 chiều** (hàm `_hv_wfg` phiên bản N-chiều đã bị xóa).

Không có "trả lời đúng duy nhất" cho 2 mục tiêu này vì chúng mâu thuẫn nhau (chấp nhận nhiều yêu cầu hơn thường kéo theo chi phí cao hơn). Hệ thống tìm **tập nghiệm Pareto** — tập các giải pháp mà không có giải pháp nào tốt hơn hoàn toàn một giải pháp khác. Việc chọn hành động (HL/LL) dùng **Hypervolume trên vector 2 chiều** này; riêng biến `r_H` (Chebyshev 2 chiều `[r_acc_c, r_cost_c]`) chỉ dùng để log/điều chỉnh `w_accept`, `w_cost` — không lan vào loss Q-network (chi tiết ở `docs/hrl.md` mục 7 và `docs/pareto.md`).

---

## Các chế độ ablation

**[BỔ SUNG]** `train.ablation_mode` (mặc định `"HRL-VGAE"`) chọn cách thay thế từng thành phần của pipeline bằng heuristic, để đo đóng góp riêng của DRL ở từng tầng. Chi tiết đầy đủ ở `docs/ABLATION.md`; bảng dưới đây là tổng quan kèm **căn cứ code** cho từng ô.

| Ablation | Request selection | VNF placement | Routing |
|---|---|---|---|
| `DefaultBaseline` | By remaining time | Greedy, least-loaded server | Basic Dijkstra |
| `Random-Fit` | Random request | DRL low-level policy | Weighted Dijkstra |
| `GreedyFIFO` | Arrival order | DRL low-level policy | Weighted Dijkstra |
| `GreedyLowLevel` | DRL high-level policy | Greedy, least-loaded server | Weighted Dijkstra |
| `HRL-VGAE` | DRL high-level policy | DRL low-level policy | Weighted Dijkstra |

**Căn cứ code cho từng cột:**

| Cột | Nơi cài đặt | Logic thực tế |
|---|---|---|
| **Request selection** | `agents/hl_agent.py:89-98` (`HLAgent.select_sfc`) | `DefaultBaseline`: `min(valid, key=lambda i: queue[i].deadline − t)` — chọn SFC có **ít thời gian còn lại nhất** (tương đương EDF — earliest deadline first, vì `deadline − t` nhỏ nhất cũng là `deadline` nhỏ nhất). `Random-Fit`: `random.choice(valid)`. `GreedyFIFO`: `min(valid, key=lambda i: queue[i].arrival_time)` — **đến sớm nhất**. `GreedyLowLevel` và `HRL-VGAE`: đi vào nhánh DRL (ε-greedy + Q-network + HV). |
| **VNF placement** | `agents/ll_agent.py:75-78` (`LLAgent.select_node`) | `DefaultBaseline` **và** `GreedyLowLevel`: `max(valid_nodes, key=lambda n: G.nodes[n].get("cpu_free", 0.0))` — node có **`cpu_free` lớn nhất** ("least-loaded" theo nghĩa tuyệt đối, không phải tỷ lệ %). Yêu cầu thêm `G is not None` (`train.py` luôn truyền `env.G`). Ba mode còn lại đi vào nhánh DRL. |
| **Routing** | `train.py:232` (đoạn tới node VNF) và `train.py:283` (đoạn cuối tới `dest`) | `lambda_penalty` được đặt thành `0.0` **chỉ** ở `DefaultBaseline`, ngược lại `cfg.reward.lambda_penalty = 1.0`. Với `λ = 0` trọng số Dijkstra rơi về `w = delay` — "Basic Dijkstra". Ràng buộc cứng `bw_free ≥ req_bw` **vẫn được áp dụng** ở mọi mode. |

**Ngoài ra, ablation còn tắt luôn việc huấn luyện của agent bị thay thế** (`train.py:410-420`):

```python
if sfc_total_count % cfg.train.train_interval_sfc == 0:
    if cfg.train.ablation_mode in ["HRL-VGAE", "Random-Fit", "GreedyFIFO"]:
        loss_ll = ll_agent.train_step()      # LL chỉ được train ở 3 mode này
    if cfg.train.ablation_mode in ["HRL-VGAE", "GreedyLowLevel"]:
        loss_hl = hl_agent.train_step()      # HL chỉ được train ở 2 mode này
```

Hệ quả: `DefaultBaseline` **không train cả HL lẫn LL**; `GreedyLowLevel` train HL nhưng không train LL; `Random-Fit`/`GreedyFIFO` train LL nhưng không train HL. Đây chính là nội dung `docs/ABLATION.md` mô tả là "Agents bypass DQN selection and training when their component is overridden by a greedy/random strategy".

Chọn mode khi chạy: `--ablation <mode>` (cả `train.py` lẫn `evaluate.py`, `evaluate.py` khai báo `choices=["DefaultBaseline", "Random-Fit", "GreedyFIFO", "GreedyLowLevel", "HRL-VGAE"]`, mặc định `HRL-VGAE`).

---

## Giải phóng tài nguyên theo từng operation

**[BỔ SUNG]** `env/nfv_env.py` giải phóng tài nguyên **theo từng operation** dựa trên **thời điểm hoàn thành thực**, chứ không phải khi SFC hết `deadline`. Cơ chế dùng một **min-heap** `_op_heap` chứa các tuple `(finish_time, counter, spec)` (`counter` để phá hoà và tránh `heapq` so sánh `spec`):

- **Khi commit** — `commit_sfc(...)` gọi `_schedule_operation_releases(allocations, t, sfc)` để đẩy sự kiện vào heap, với mốc `t` **nhân dần dọc theo embedding**:
  - `"link"`: `finish = t + Σ delay(link)` trên `path` → giải phóng `bw_free` cho **từng link** trên path (chỉ khi `len(path) > 1`).
  - `"cpu"`: `finish = t + proc_delay[node] + VNF.delay_at(node)` → giải phóng `cpu_free`, `ram_free`, `hdd_free` của node (chỉ khi `cpu_req > 0`; đoạn cuối tới `dest` có `cpu_req = 0` nên chỉ sinh sự kiện `"link"`).
- **Khi tiến thời gian** — `step_time()` gọi `_release_expired_active_sfcs()` **đầu tiên**, pop mọi phần tử có `finish_time <= t` và cộng trả tài nguyên tương ứng; nếu có thay đổi thì xóa `_load_std_cache`.

**[BỔ SUNG] Hệ quả:** `deadline` chỉ còn là hạn phải được phục vụ. Một SFC thường **giải phóng tài nguyên sớm hơn hẳn deadline** vì `finish_serve` chỉ là `t + Σ(link delay) + Σ(proc_delay + VNF delay)` (cỡ vài chục đơn vị thời gian) trong khi `deadline ∈ [20, 100]`. Chi tiết ở `docs/hrl.md` mục 2.1.1 và `docs/pareto.md`.

---

## Vòng lặp training

```
for epoch in 1..epochs (mặc định 10):
  for each episode file (topology + SFC trace), đã shuffle mỗi epoch:
    G, reqs, topo_id = parse_episode(file)
    for pass in 1..passes_per_file (mặc định 5):     # replay cùng episode
      env.reset(G, requests, topology_id=topo_id)
      encode_graph(vgae, env, device)                # z_nodes, z_global ban đầu

      while not done:
        HL chọn SFC  →  LL đặt từng VNF  →  Dijkstra route mỗi đoạn
        commit (thành công) hoặc rollback (thất bại)
        lưu transition LL (mỗi VNF) + gọi ll_agent.train_step() ngay sau mỗi lần lưu
        lưu transition HL (mỗi SFC) + gọi hl_agent.train_step() ngay sau mỗi lần lưu
        train VGAE mỗi cfg.vgae.train_every_steps (=20) SFC thành công
        nếu hàng đợi rỗng: step_time(); encode lại nếu graph_dirty

      cuối episode (nếu train và không fixed_weight): adapt_weights() cập nhật w_accept/w_cost
      epsilon decay (theo lịch mũ, chỉ sau warmup)
      log metrics (CSV + TensorBoard)

lưu checkpoint sau khi hết toàn bộ epoch
```

**Warm-up:** `args.warmup_epochs` epoch đầu (mặc định 2) dùng `hl_agent.epsilon = ll_agent.epsilon = 1.0` (chọn ngẫu nhiên hoàn toàn), nhưng `train_step()` của cả hai agent **vẫn được gọi** ngay khi buffer đủ mẫu tối thiểu — không phải chỉ dùng warmup để lấp buffer thụ động (xem `docs/hrl.md` mục 8).

---

## Tham số quan trọng (config.py)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `network.num_nodes` | 50 | Số node trong mạng |
| `vgae.d_latent` | 32 | Chiều không gian latent mỗi node |
| `vgae.d_hidden` | 64 | Chiều hidden layer GCN |
| `qnet.gamma` | 0.95 | Discount factor cho Bellman |
| `qnet.batch_size` | 64 | Kích thước batch training |
| `qnet.max_q_vectors_per_action` | 10 | Số vector tối đa trong Q-set |
| `qnet.hl_buffer_size` | 15 000 | Replay buffer HL |
| `qnet.ll_buffer_size` | 72 000 | Replay buffer LL |
| `pareto.hv_ref_cost` / `pareto.hv_ref_success` | -25.0 | Hai thành phần của điểm tham chiếu HV 2 chiều (`hv_reference_point()`); `ref` của LL còn được thích nghi thêm qua `ll_ref_momentum`/`ll_ref_margin` |
| `train.ablation_mode` | `"HRL-VGAE"` | Chế độ ablation (`DefaultBaseline`, `Random-Fit`, `GreedyFIFO`, `GreedyLowLevel`, `HRL-VGAE`) — xem mục "Các chế độ ablation" |
| `train.k_hop_explore` | 3 | Bán kính giới hạn node khi LL chọn ngẫu nhiên trong ε-greedy |
| `sfc.arrival_rate` | 20.0 | Tốc độ đến trung bình (Poisson), dùng ở chế độ ngẫu nhiên |
| `sfc.arrival_interval` | 100 | Số timestep giữa các đợt arrival ở chế độ ngẫu nhiên |
| `sfc.vnf_len_min/max` | 2–8 | Độ dài chuỗi VNF |
| `train.episode_horizon` | 500 | Số timestep tối đa mỗi episode ở chế độ ngẫu nhiên (không áp dụng ở dataset mode) |

Lưu ý: `sfc_quota_per_timestep` (property trong `Config`) được định nghĩa nhưng **không được `train.py`/`nfv_env.py` sử dụng ở đâu cả** — số SFC xử lý mỗi timestep chỉ bị giới hạn bởi kích thước hàng đợi thực tế và `blocked_until`.

---

## Cách chạy

```bash
# Training
python train.py \
  --train-dir data/train \
  --epochs 10 \
  --passes-per-file 5 \
  --warmup-epochs 2 \
  --verbose

# Chọn chế độ ablation (mặc định HRL-VGAE)
python train.py --ablation GreedyFIFO
python train.py --ablation DefaultBaseline

# Evaluation
python evaluate.py \
  --checkpoint vgae_hrl_ql_checkpoint.pt \
  --data-dir data/test1

# Đánh giá từng chế độ ablation
python evaluate.py --ablation GreedyLowLevel

# Pareto front scan (11 weight points)
python evaluate.py --pareto-scan --pareto-points 11
```

**[CẬP NHẬT] `pareto_test.py` không còn tồn tại trong repo** — lệnh `python pareto_test.py` trong phiên bản trước của tài liệu này là nội dung lỗi thời (xác minh: `find . -name 'pareto_test.py'` không trả về kết quả). Không có bộ unit test nào cho `utils/pareto.py` hiện tại.

`evaluate.py` mặc định đọc `--data-dir data/dataset_01/test` (không phải `data/test1` như ví dụ trên) và `--checkpoint vgae_hrl_ql_checkpoint.pt`; nếu thư mục dataset không tồn tại, nó tự chuyển sang chạy `args.episodes` (mặc định 20) episode ngẫu nhiên thay vì báo lỗi.