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

---

## Các thành phần chính và file tương ứng

| Thành phần | File | Vai trò |
|---|---|---|
| Môi trường mạng | `nfv_env.py` | Quản lý tài nguyên node/link, commit/rollback |
| Sinh đồ thị & SFC | `graph_generator.py` | Barabási-Albert topology, SFC ngẫu nhiên |
| Bộ mã hóa đồ thị | `vgae.py` | GCN + VAE + GRU temporal |
| HL Q-network | `hl_scorer.py` | 4 output heads, input dim 73 |
| LL Q-network | `ll_dqn.py` | 4 output heads, input dim 202 |
| HL Agent | `hl_agent.py` | Q-set, HV selection, replay buffer |
| LL Agent | `ll_agent.py` | Q-set, HV selection, replay buffer |
| Routing | `dijkstra.py` | Custom Dijkstra với bandwidth penalty |
| Chi phí triển khai | `deploy_cost.py` | CPU + RAM + storage + bandwidth |
| Pareto & HV | `pareto.py` | dominance, HV, pruning, target Q-set |
| Cấu hình | `config.py` | Tất cả hyperparameter |
| Vòng lặp training | `train.py` | Episode loop, reward, logging |
| Đánh giá | `evaluate.py` | Inference, Pareto scan |

---

## Các mục tiêu tối ưu (4 objectives)

Hệ thống không tối ưu một chỉ số duy nhất mà cùng lúc tối ưu 4 mục tiêu, tất cả được định nghĩa theo hướng **maximize** (giá trị càng cao càng tốt):

| Index | Tên | Công thức | Ý nghĩa |
|---|---|---|---|
| 0 | r_cost | `-μ × deploy_cost` | Minimize chi phí triển khai |
| 1 | r_delay | `-(α×path_cost + β×path_delay + γ×load_std)` | Minimize độ trễ và tắc nghẽn |
| 2 | r_resource_balance | `-load_std` | Cân bằng tải giữa các link |
| 3 | r_success | `+1.0` nếu thành công | Maximize tỷ lệ đặt thành công |

Không có một "trả lời đúng duy nhất" cho 4 mục tiêu này vì chúng mâu thuẫn nhau (chấp nhận nhiều yêu cầu hơn thường kéo theo chi phí cao hơn). Hệ thống tìm **tập nghiệm Pareto** — tập các giải pháp mà không có giải pháp nào tốt hơn hoàn toàn một giải pháp khác.

---

## Vòng lặp training

```
for epoch in 1..10:
  for each episode file (topology + SFC trace):
    for pass in 1..5:                    # replay cùng episode 5 lần
      env.reset(G, requests)
      VGAE.encode(graph)
      
      while not done:
        HL chọn SFC  →  LL đặt VNF  →  Dijkstra route
        compute reward vector [r_cost, r_delay, r_balance, r_success]
        store to replay buffer
        train HL network (Q-set Bellman update)
        train LL network (Q-set Bellman update)
        train VGAE (every 20 SFC processed)
      
      epsilon decay (exploration giảm dần)
      log metrics
```

**Warm-up:** 2 epoch đầu dùng `epsilon = 1.0` (chọn ngẫu nhiên hoàn toàn) để lấp đầy replay buffer trước khi bắt đầu học.

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
| `sfc.arrival_rate` | 20.0 | Tốc độ đến trung bình (Poisson) |
| `sfc.vnf_len_min/max` | 2–8 | Độ dài chuỗi VNF |

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
