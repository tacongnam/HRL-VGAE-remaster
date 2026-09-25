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
│  │  Thành công → chiếm CPU, băng thông, ghi nhận       │  │
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
           tính Q-set = {q_vec ∈ ℝ⁴}
         Chọn SFC có HyperVolume(Q-set) lớn nhất

Bước 3  LL Agent đặt từng VNF (k = 1, 2, ..., K):
         Với mỗi node còn đủ CPU:
           tính Q-set = {q_vec ∈ ℝ⁴}
         Chọn node có HyperVolume(Q-set) lớn nhất
         Chạy Dijkstra từ node trước → node vừa chọn

Bước 4  Sau khi đặt xong tất cả K VNF:
         Nếu thành công → commit tài nguyên, nhận reward
         Nếu thất bại (không đường đi, không đủ CPU)
           → rollback tất cả, nhận failure penalty

Bước 5  Cập nhật replay buffer và huấn luyện các mạng
```

Lưu ý về đồng hồ: trong `run_episode` (`train.py`), timestep `t` **không** tự tiến sau mỗi SFC — nó chỉ tiến (`env.step_time()`) khi hàng đợi rỗng hoặc mọi SFC còn lại đang bị `blocked_until`. VGAE cũng không encode lại sau mỗi VNF; nó dùng lại cache `z_nodes`/`z_global` cho tới khi `graph_dirty=True` và `step_time()` được gọi (xem `docs/vgae.md` mục 9).

---

## Các thành phần chính và file tương ứng

| Thành phần | File | Vai trò |
|---|---|---|
| Môi trường mạng | `env/nfv_env.py` | Quản lý tài nguyên node/link, commit/rollback |
| Sinh đồ thị & SFC | `utils/graph_generator.py` | Barabási-Albert topology, SFC ngẫu nhiên |
| Bộ mã hóa đồ thị | `models/vgae.py` | GCN + VAE + GRU temporal |
| HL Q-network | `models/hl_scorer.py` | 4 output heads, input dim 73 (`cfg.d_hl_input`) |
| LL Q-network | `models/ll_dqn.py` | 4 output heads, input dim 139 (`cfg.d_ll_input`) |
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

## Các mục tiêu tối ưu (4 objectives)

Hệ thống không tối ưu một chỉ số duy nhất mà cùng lúc tối ưu 4 mục tiêu, tất cả được định nghĩa theo hướng **maximize** (giá trị càng cao càng tốt). Vector này được `train.py` dựng qua `_build_reward_vec_4obj(r_cost, r_delay, r_balance, r_success)`:

| Index | Tên | Công thức thực tế trong `train.py` | Ý nghĩa |
|---|---|---|---|
| 0 | r_cost | `-cfg.reward.mu_deploy_cost * deploy_cost` (SFC thành công) hoặc thành phần tương ứng trong `cost_per_step` cho từng transition LL | Minimize chi phí triển khai |
| 1 | r_delay | `r_bar_quality` = trung bình `-(α·path_cost + β·path_delay + γ_load·load_std)` trên các đoạn định tuyến | Minimize độ trễ và tắc nghẽn |
| 2 | r_resource_balance | `-load_std` (độ lệch chuẩn mức sử dụng link toàn mạng) | Cân bằng tải giữa các link |
| 3 | r_success | `1.0` nếu thành công, `0.0`/`-10` nếu không (transition trung gian LL = 0, thất bại = failure_vec) | Maximize tỷ lệ đặt thành công |

Khi SFC/VNF thất bại, cả 4 thành phần dùng `cfg.pareto.failure_penalty_vector()` = `[-10, -10, -10, -10]` (mặc định `failure_penalty_* = 10.0`), không phải công thức trên.

Không có một "trả lời đúng duy nhất" cho 4 mục tiêu này vì chúng mâu thuẫn nhau (chấp nhận nhiều yêu cầu hơn thường kéo theo chi phí cao hơn). Hệ thống tìm **tập nghiệm Pareto** — tập các giải pháp mà không có giải pháp nào tốt hơn hoàn toàn một giải pháp khác. Việc chọn hành động (HL/LL) dùng **Hypervolume trên vector 4 chiều** này; riêng biến `r_H` (Chebyshev 2 chiều `[r_accept, r_cost]`) chỉ dùng để log/điều chỉnh `w_accept`, `w_cost` — không lan vào loss Q-network (chi tiết ở `docs/hrl.md` mục 7 và `docs/pareto.md`).

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
| `pareto.hv_ref_*` | -1.0 (×4) | Reference point cho HV |
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

# Evaluation
python evaluate.py \
  --checkpoint vgae_hrl_ql_checkpoint.pt \
  --data-dir data/test1

# Pareto front scan (11 weight points)
python evaluate.py --pareto-scan --pareto-points 11

# Unit tests (không cần GPU)
python pareto_test.py
```

`evaluate.py` mặc định đọc `--data-dir data/dataset_01/test` (không phải `data/test1` như ví dụ trên) và `--checkpoint vgae_hrl_ql_checkpoint.pt`; nếu thư mục dataset không tồn tại, nó tự chuyển sang chạy `args.episodes` (mặc định 20) episode ngẫu nhiên thay vì báo lỗi.