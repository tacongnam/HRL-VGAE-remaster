# Học tăng cường phân cấp (HRL) cho đặt SFC

## 1. Vì sao cần phân cấp?

Với mỗi yêu cầu SFC, hệ thống phải đưa ra chuỗi quyết định lồng nhau:

1. **Chọn SFC nào** trong hàng đợi để xử lý trước.
2. Với SFC đó, **chọn node** cho từng VNF (k = 1..K).
3. **Chọn đường đi** giữa các node liên tiếp.

Nếu gộp tất cả vào một agent phẳng, không gian hành động là tích Descartes `|queue| × N^K` (ví dụ 20 SFC × 50⁵ ≈ 6·10⁹ cho K = 5) — không thể học trực tiếp. HRL tách bài toán thành các tầng, mỗi tầng có không gian hành động nhỏ và một khung thời gian riêng:

| Tầng | Quyết định | Không gian hành động | Tần suất | Được học? |
|---|---|---|---|---|
| **HL Agent** | Chọn SFC từ hàng đợi | `\|queue\|` (thường 1–vài chục) | 1 lần / SFC | Có (Q-network) |
| **LL Agent** | Chọn node đặt VNF thứ k | ≤ `N` (chỉ node đủ CPU) | K lần / SFC | Có (Q-network) |
| **Routing** | Đường đi giữa hai node | Đồ thị con còn đủ băng thông | K + 1 lần / SFC | Không (Dijkstra) |

Ý tưởng then chốt: phần **định tuyến** có thuật toán cổ điển tốt (Dijkstra với trọng số tắc nghẽn) nên không cần học; sức mạnh của DRL dành cho hai quyết định khó mà heuristic đơn giản xử lý kém: *ưu tiên SFC nào* và *đặt VNF ở đâu*.

```
          ┌─────────────────────────────────────────────┐
          │              Hàng đợi SFC (queue)           │
          └───────────────────┬─────────────────────────┘
                              │  state: z_global, đặc trưng SFC, w
                              ▼
                     ┌─────────────────┐
                     │    HL Agent     │  chọn SFC ★
                     └────────┬────────┘
                              │ SFC ★ = (src, dst, bw, [VNF₁..VNF_K], deadline)
                              ▼
          ┌───────────────────────────────────────────┐
          │  for k = 1..K:                            │
          │     LL Agent: chọn node vₖ (mask CPU)     │
          │     Dijkstra: v_{k-1} → vₖ                │
          │     allocate tạm thời CPU + băng thông    │
          │  Dijkstra: v_K → dst                      │
          └───────────────────┬───────────────────────┘
                              ▼
                   thành công? ── có ──► commit
                        │
                       không ─────────► rollback + failure penalty
```

---

## 2. Mô hình hóa bài toán

### 2.1. Thời gian và hàng đợi (`nfv_env.py`)

Môi trường rời rạc theo timestep `t`. Có hai chế độ:

| Chế độ | Nguồn SFC | Điều kiện kết thúc |
|---|---|---|
| **Dataset** (`reset(G, requests, topology_id)`) | Trace từ file JSON: mỗi request có `arrival_time`, `deadline = arrival + d_max` | Đã phát hết request **và** hàng đợi rỗng |
| **Ngẫu nhiên** (`reset()`) | Sinh Barabási-Albert + Poisson arrival | `t ≥ episode_horizon` (500) |

Các quy tắc quan trọng:

- Một SFC **hết hạn** khi `deadline ≤ t` (`SFCRequest.is_expired`). SFC hết hạn còn trong hàng đợi bị đếm là `rejected`.
- **Deadline đóng hai vai trò** trong code: là hạn phải được phục vụ, đồng thời là thời điểm tài nguyên của SFC đã commit được giải phóng (`_release_expired_active_sfcs` trả lại CPU/băng thông khi `deadline ≤ t`). Nói cách khác, SFC "sống" trên mạng đến hết deadline của nó.
- `acceptance_ratio = accepted / total_attempted`, với `total_attempted = accepted + rejected`.

**Đồng hồ chỉ tiến khi cần.** Trong `run_episode`, môi trường **không** tăng `t` sau mỗi SFC. Time chỉ tiến (`env.step_time()`) khi hàng đợi rỗng, hoặc mọi SFC còn lại đều đang bị chặn (xem 2.4). Vì vậy trong một timestep, agent có thể xử lý toàn bộ hàng đợi — và tài nguyên bị chiếm bởi SFC vừa commit ảnh hưởng ngay tới SFC kế tiếp trong cùng timestep.

### 2.2. Đặc trưng SFC — `to_feature_vector` (7 chiều)

```python
[source, dest, bandwidth, F_k, deadline - t, urgency, total_cpu_req]
```

với `urgency = 1 / max(1, deadline - t)`. Đây là vector đầu vào `sfc_feat` dùng chung cho cả HL và LL (hằng `d_sfc = 7`). Các chiều **chưa được chuẩn hóa** (`source`/`dest` là id node 0–49, `total_cpu_req` có thể lên hàng nghìn) nên mạng học phải tự thích nghi với thang đo khác nhau; chuẩn hóa các chiều này là một hướng cải thiện rõ ràng.

Đặc trưng VNF (`build_vnf_feature`, 2 chiều): `[cpu_req / cpu_capacity, 1.0]` (chiều thứ hai là hằng số bias).

### 2.3. Vector phần thưởng 4 mục tiêu

Mọi Q-network dự đoán vector 4 chiều, tất cả theo hướng **maximize**:

| Index | Tên | Giá trị khi SFC thành công | Giá trị khi thất bại |
|---|---|---|---|
| 0 | `r_cost` | `-μ_deploy × deploy_cost` | −10 |
| 1 | `r_delay` | `r̄_quality` (xem dưới) | −10 |
| 2 | `r_balance` | `-load_std` | −10 |
| 3 | `r_success` | `+1.0` | −10 |

Thất bại dùng `failure_penalty_vector()` = `[-10, -10, -10, -10]` (`ParetoConfig.failure_penalty_*`).

**Thành phần chất lượng đường đi.** Với mỗi đoạn định tuyến (K đoạn tới các node đặt VNF + 1 đoạn cuối tới đích):

```
r_quality_seg = -( α × path_cost + β × path_delay + γ_load × load_std )
```

trong đó, tính **sau khi đã allocate** đoạn đó:

- `path_cost = Σ_{link ∈ path} (1 − bw_free/bw_total)` — tổng độ sử dụng các link trên đường đi;
- `path_delay = Σ delay_link` — chỉ cộng trễ truyền dẫn (`d_l`) của link, không gồm `proc_delay` của node;
- `load_std` — độ lệch chuẩn (ddof = 1) của độ sử dụng **toàn bộ** link trong mạng, tính **một lần sau khi commit** rồi dùng chung cho mọi đoạn của SFC.

Lấy trung bình trên `n_steps = F_k + 1` đoạn: `r̄_quality = (Σ r_quality_seg) / (F_k + 1)`. Đây chính là thành phần `r_delay` trong vector thưởng.

**Chi phí triển khai** (`deploy_cost.py`), cộng theo từng allocation `(node, path, cpu_req, bw)`:

```
Nếu cpu_req > 0:
   w_cpu × cost_cpu(node) × cpu_req
 + w_ram × cost_ram(node) × (cpu_req × ram_per_cpu_unit)
 + w_storage × cost_stor(node) × storage_per_vnf
Luôn luôn:
 + w_bw   × bw × path_len
 + w_init × init_cost_per_hop × path_len
```

Allocation cho đoạn cuối tới `dest` có `cpu_req = 0` nên chỉ mang chi phí băng thông + khởi tạo. `path_len = len(path) − 1`; hai VNF liên tiếp đặt trên cùng một node cho `path = [node]`, độ dài 0, không tốn băng thông.

### 2.4. Vòng đời một SFC: thành công, thất bại, chặn tạm thời

```
                        ┌────────────── embedding_failed ──────────────┐
                        │  (a) không node nào đủ CPU (mask rỗng)       │
                        │  (b) Dijkstra không tìm thấy đường           │
                        │  (c) đường tới dest không tồn tại            │
                        ▼                                              │
  rollback toàn bộ allocation tạm                                      │
  reward = failure_vec                                                 │
  SFC còn hạn ở t+1?  ── có ──► đưa lại vào queue, blocked_until = t+1 │
                      └─ không ─► rejected += 1                        │
```

`blocked_until[sfc_id] = t + 1` ngăn HL agent chọn lại chính SFC vừa thất bại trong cùng timestep (nếu không nó sẽ bị chọn đi chọn lại vô hạn trong khi đồng hồ không tiến). Sang timestep sau, tài nguyên có thể đã được giải phóng nên SFC được thử lại — đến khi hết hạn.

---

## 3. HL Agent — chọn SFC

**File:** `hl_agent.py` (agent), `hl_scorer.py` (mạng).

### 3.1. Mạng Q kiểu "scorer"

Số SFC trong hàng đợi thay đổi theo thời gian, nên mạng **không** có một đầu ra cho mỗi hành động (như DQN Atari). Thay vào đó, mạng chấm điểm một cặp *(trạng thái, SFC ứng viên)* rồi lặp cho từng ứng viên:

```
                 z_global (64) ┐
   đặc trưng SFC ứng viên (7) ├─ concat → 73 ─► Linear(73→256) ─ReLU
             pareto_w (2)      ┘                   Linear(256→128) ─ReLU
                                                        │
                        ┌────────────┬────────────┬─────┴──────┬────────────┐
                     head_cost   head_delay   head_balance  head_success
                        Linear(128→1) × 4
                                                        │
                                       Q(s, a) = [q_cost, q_delay, q_bal, q_succ] ∈ ℝ⁴
```

```python
x = torch.cat([z_global, sfc_features, pareto_w], dim=-1)
h = self.trunk(x)
return torch.cat([head(h) for head in self.q_heads], dim=-1)   # [batch, 4]
```

Toàn bộ `m` SFC hợp lệ được chấm trong **một batch duy nhất** (`z_global` và `pareto_w` được `expand` thành `m` hàng). 4 head chia sẻ chung thân (trunk) nhưng mỗi head ước lượng tổng phần thưởng kỳ vọng của một mục tiêu riêng. Tổng tham số ≈ 52K.

### 3.2. Chọn hành động (`select_sfc`)

```
valid = [i | SFC i chưa hết hạn  VÀ  blocked_until[i] ≤ t]
nếu valid rỗng → trả về None  (→ step_time)

với xác suất ε:      chọn ngẫu nhiên trong valid          (khám phá)
ngược lại:
    Q_i = q_net(z_global, sfc_i, w)              # vector 4 chiều cho mỗi SFC
    Q-set_i = prune(non_dominated({Q_i}))        # ở đây luôn là tập 1 phần tử
    chọn SFC có Hypervolume(Q-set_i) lớn nhất    # hòa → tie-break
```

Vì mạng cho **đúng một vector** mỗi ứng viên, "Q-set" của một hành động lúc chọn luôn có một phần tử, và hypervolume của nó là tích `Π (qᵢ − refᵢ)` (chi tiết ở tài liệu *DRL + Pareto + Hypervolume*). Bước `non_dominated` và `prune_by_hypervolume` giữ nguyên API để mở rộng sang Q-set nhiều phần tử về sau.

### 3.3. Replay buffer và bộ dữ liệu chuyển tiếp

Mỗi lần HL xử lý một SFC (thành công hoặc thất bại) sinh đúng **một** transition:

| Trường | Giá trị trong `train.py` |
|---|---|
| `z_global` | Trạng thái toàn cục lúc chọn |
| `sfc_feat` | Đặc trưng của SFC **đã chọn** |
| `pareto_w` | Vector trọng số của episode |
| `action_idx` | Chỉ số SFC trong hàng đợi (chỉ để tham khảo, loss không dùng) |
| `reward_vec` | `[r_cost, r_delay, r_balance, 1.0]` nếu thành công; `failure_vec` nếu thất bại |
| `next_z_global` | **Cùng `z_global`** (không encode lại trước khi lưu) |
| `next_sfc_feats` | Đặc trưng các SFC còn hạn trong hàng đợi sau bước này |
| `done` | `1.0` nếu hàng đợi rỗng và đã hết request trong trace |
| `valid_action_mask` | Toàn `True` (kích thước `len(next_sfc_feats)`) |

Buffer: `deque(maxlen = 15 000)`.

### 3.4. Cập nhật (`train_step`)

Được gọi **một lần sau mỗi transition HL** khi buffer đủ `min(batch_size, buffer_size//10) = 64` mẫu.

```
với mỗi mẫu trong batch (64 mẫu):
    Q_hiện_tại = q_net(z_global, sfc_feat, w)                    # [4]
    nếu done hoặc không còn SFC kế tiếp:
        target_set = { r }
    ngược lại:
        cho từng SFC kế tiếp j:  q_j = target_net(z', sfc_j, w')   # [4]
        ND      = non_dominated({q_j})
        targets = non_dominated({ r + γ·q  |  q ∈ ND })
        nếu |targets| > 10: prune bằng hypervolume contribution
        target_set = targets
    target = mean(target_set)                                   # [4]

loss = MSE(Q_hiện_tại, target)
Adam step, clip_grad_norm(10)
mỗi 50 bước: target_net ← q_net
```

Điểm cần lưu ý khi đọc thuật toán:

- Đây là Bellman **dạng vector**: `Q(s,a) ← r + γ · (Q của hành động kế tiếp)`, nhưng thay vì `max` (không xác định với vector), hành động kế tiếp được đại diện bởi **tập Pareto** của các Q-vector ứng viên, rồi nén thành một vector bằng **trung bình** để có target hồi quy được bằng MSE.
- `γ = 0.95`, `lr = 5e-4`, `target_update_freq = 50`.

---

## 4. LL Agent — chọn node đặt VNF

**File:** `ll_agent.py` (agent), `ll_dqn.py` (mạng).

### 4.1. Trạng thái và hành động

Với VNF thứ `k` của SFC đã chọn:

| Thành phần | Kích thước | Ý nghĩa |
|---|---|---|
| `z_global` | 64 | Trạng thái toàn mạng |
| `z_prev` | 32 | Embedding của node hiện tại (`current_source`: source của SFC với VNF đầu, hoặc node của VNF trước) |
| `z_cand` | 32 | Embedding của node **ứng viên** (hành động) |
| `vnf_feat` | 2 | `[cpu_req/capacity, 1.0]` |
| `sfc_feat` | 7 | Đặc trưng SFC đang xử lý |
| `pareto_w` | 2 | Vector trọng số |
| **Tổng** | **139** | `d_ll_input` |

Hành động = "đặt VNF lên node `i`". **Mask hợp lệ** (`env.build_ll_mask`): node là function node **và** `cpu_free ≥ cpu_req`. Mask không kiểm tra đường đi tới node đó, nên vẫn có thể chọn phải node không định tuyến được (→ thất bại loại (b) ở mục 2.4).

Mạng `LLNodeScorer` cùng kiểu scorer với HL (thân `139→256→128`, 4 head, ≈ 69K tham số). Toàn bộ `N` node được chấm trong một batch; node không hợp lệ bị bỏ qua bằng mask.

### 4.2. Chọn hành động (`select_node`)

```
valid_nodes = {i | cpu_mask[i]}       nếu rỗng → None (SFC thất bại)
với xác suất ε: chọn ngẫu nhiên trong valid_nodes
ngược lại:
    Q_i = q_net(z_global, z_prev, z_i, vnf_feat, sfc_feat, w)  ∀ i
    Q-set_i = prune(non_dominated({Q_i}))   nếu i hợp lệ, ngược lại ∅
    chọn node có HV(Q-set_i) lớn nhất (mask loại node không hợp lệ)
```

### 4.3. Transition và target

Sau mỗi SFC **thành công**, K transition LL được lưu (một cho mỗi VNF). Với VNF thứ `k`:

| Trường | Giá trị |
|---|---|
| state | `(z_global, z_prev, z_cand=z_nodes[node đã chọn], vnf_feat, sfc_feat, w)` |
| reward | `[ -μ × cost/(F_k+1),  r_quality_k,  −load_std,  1.0 nếu k cuối else 0.0 ]` |
| next state | `None` nếu k là VNF cuối; ngược lại `(z_global, z_prev'=embedding node vừa chọn, toàn bộ z_nodes, vnf_feat kế, sfc_feat, mask kế, w)` |
| done | `1.0` nếu k cuối |

Chú ý cách chia thưởng: chi phí triển khai của cả SFC được chia đều cho `n_steps = F_k + 1` (kể cả đoạn tới `dest` không có transition riêng), nên tổng thưởng chi phí trên K transition là `F_k/(F_k+1)` tổng chi phí. `r_success = 1.0` chỉ xuất hiện ở transition cuối — tín hiệu "cả chuỗi đã thành công".

Khi SFC **thất bại**, mọi transition đã tạo trước điểm thất bại được lưu với `reward = failure_vec / J` (J = số VNF đã đặt tạm), `done = 1`, `next = None`. Hai hệ quả thiết kế:

1. Các quyết định "tốt" trước điểm thất bại cũng bị phạt (credit assignment thô).
2. **Nếu thất bại xảy ra ngay tại VNF đầu tiên** (không node nào đủ CPU hoặc không có đường), `ll_records` rỗng nên không có transition nào được lưu — chính quyết định dẫn tới thất bại chưa nhận phản hồi âm trực tiếp ở LL (HL vẫn nhận `failure_vec`).

Cập nhật (`train_step`) giống HL: MSE giữa `Q(s,a)` và trung bình của tập target, với target tính từ `target_net` trên các node hợp lệ của trạng thái kế tiếp. LL gọi `train_step` **sau mỗi transition** được lưu; buffer `deque(maxlen = 72 000)`.

---

## 5. Định tuyến — Dijkstra có phạt tắc nghẽn

**File:** `dijkstra.py`, hàm `custom_dijkstra(G, source, target, req_bw, omega_bw, lambda_penalty)`.

Chỉ xét các link còn đủ băng thông (`bw_free ≥ req_bw`). Trọng số của link:

```
ρ = 1 − bw_free / bw_total                    # độ sử dụng, kẹp vào [0, 1 − 1e-9]
w = delay + λ × ( exp(ρ · ln Ω) − 1 ) = delay + λ × ( Ω^ρ − 1 )
```

với `Ω = omega_bw = 100`, `λ = lambda_penalty = 1`. Với `delay` 2–5 ms:

| Độ sử dụng ρ | Phạt `Ω^ρ − 1` |
|---|---|
| 0.0 | 0 |
| 0.3 | ≈ 2.98 |
| 0.5 | 9 |
| 0.8 | ≈ 38.8 |
| 0.95 | ≈ 78.4 |

Nên khi link trống, đường đi ngắn nhất theo trễ; khi link gần đầy, phạt tăng theo hàm mũ và Dijkstra tự động "đi vòng" — một cách cân bằng tải mà không cần học. Trường hợp đặc biệt: `source == target` trả về `[source]` ngay.

Dijkstra được gọi `K + 1` lần mỗi SFC, và **mỗi đoạn được `allocate` ngay** trước khi tính đoạn kế tiếp, nên các đoạn của cùng một SFC "nhìn thấy" băng thông mà các đoạn trước đã chiếm. (Lớp `PathCache` có trong file nhưng `train.py` hiện không dùng.)

---

## 6. Commit / Rollback

`NFVEnvironment` giữ hai thao tác đối xứng:

| Thao tác | Tác dụng |
|---|---|
| `allocate(node, path, cpu, bw)` | Trừ `cpu_free` của node, trừ `bw_free` của mọi link trên path |
| `_rollback_resources(allocations)` | Cộng trả đúng các giá trị trên |
| `commit_sfc(sfc, allocations, deploy_cost)` | Ghi vào `active_embeddings` (kèm deadline để giải phóng sau), `accepted += 1`, cộng dồn `total_deploy_cost` |

Cơ chế "allocate tạm rồi rollback nếu thất bại" đảm bảo tính **nguyên tử**: một SFC hoặc được đặt trọn vẹn, hoặc không thay đổi gì trên mạng. `partial_allocations` gồm cả K allocation VNF và allocation cuối tới `dest`.

---

## 7. Điều khiển đa mục tiêu: trọng số `w` và scalarizer

Ngoài vector 4 chiều dùng cho lựa chọn hành động, `train.py` còn duy trì một tầng **scalarization 2 mục tiêu** cho phía "admission" (chấp nhận / chi phí):

- `r_accept_component = revenue + θ × urgency + λ_L × r̄_quality`, với `revenue = (μ_cpu × total_cpu + μ_bw × bw × (F_k+1)) / F_k`;
- `r_cost_component = −μ_deploy × deploy_cost`.

`ParetoScalarizer.scalarize` biến cặp này thành một số qua **Chebyshev tăng cường** (augmented Chebyshev):

```
d_a = w_accept × (utopia_accept − r_accept) / range_accept
d_c = w_cost   × (utopia_cost   − r_cost)   / range_cost
r_H = −( max(d_a, d_c) + ρ × (d_a + d_c) )          ρ = 0.05
```

`utopia` (điểm lý tưởng) và `nadir` (điểm tệ nhất) được cập nhật động bằng `update_utopia` (trung bình trượt với `utopia_momentum = 0.98`).

**Vai trò của `r_H` trong code hiện tại:** được cộng dồn thành `hl_reward` của episode (chỉ số theo dõi trong log/TensorBoard) và cập nhật utopia/nadir. **Nó không tham gia loss của Q-network** — Q-network học từ vector 4 chiều, không phải từ `r_H`.

**Trọng số ưu tiên `w = (w_accept, w_cost)`:**

| Giai đoạn | Cách chọn |
|---|---|
| Train | Mỗi episode lấy ngẫu nhiên một trong `num_weight_bins = 11` điểm `{0, 0.1, …, 1.0}` (`sample_weight`) |
| Evaluate (mặc định) | `scalarizer.w_accept/w_cost` lưu trong checkpoint |
| Evaluate (`--pareto-scan`) | Quét tuần tự 11 điểm, mỗi điểm cố định cho cả episode |

`w` được ghép vào đầu vào của cả hai Q-network (2 chiều cuối), tạo điều kiện để một mạng duy nhất phục vụ nhiều sở thích. Cuối mỗi episode, `adapt_weights` điều chỉnh `w_accept` theo khoảng cách giữa tỷ lệ chấp nhận đạt được và mục tiêu 0.9, rồi làm mượt với trọng số của episode vừa chạy (`0.9 × w_episode + 0.1 × w_scalarizer`).

Kết quả episode cũng được thêm vào `ParetoArchive` (điểm 2 chiều `[avg_r_accept, avg_r_cost]`), dùng để theo dõi mặt Pareto xấp xỉ theo thời gian (in mỗi `front_log_every = 25` episode).

---

## 8. Khám phá và lịch huấn luyện

**Epsilon-greedy** áp dụng độc lập cho HL và LL. Lịch (`train.py`, `main`):

```
warmup_episodes   = warmup_epochs × số_file × passes_per_file
decay_episodes    = 0.75 × (tổng_episode − warmup_episodes)
eps_decay_per_ep  = (eps_end / eps_start) ^ (1 / decay_episodes)

warm-up:      ε = 1.0 cố định (chọn hoàn toàn ngẫu nhiên)
sau warm-up:  ε ← max(0.05, ε × eps_decay_per_ep)   # sau mỗi episode
```

Khác với lời mô tả "chỉ lấp buffer", trong thực tế các `train_step` vẫn chạy trong warm-up ngay khi mỗi buffer đủ 64 mẫu — mạng học từ dữ liệu ngẫu nhiên ở giai đoạn này. Tham số `qnet.eps_decay = 0.995` trong config **không** được dùng; lịch thực tế là hàm mũ tính ở trên.

Cấu trúc vòng lặp:

```
for epoch in 1..epochs:
    for file in shuffle(train_files):           # mỗi file = topology + trace SFC
        G, reqs, _, topo_id = parse_episode(file)
        for pass in 1..passes_per_file:         # phát lại cùng episode nhiều lần
            env.reset(G, reqs, topo_id)
            stats = run_episode(...)            # xem mục 9
            decay ε (nếu hết warm-up)
            log
lưu checkpoint: vgae, hl_q, ll_q, epsilon, w_accept, w_cost, hv_ref_point, pareto_archive
```

---

## 9. Chạy thử tay một SFC (K = 3)

Giả sử `t = 40`, hàng đợi có 3 SFC hợp lệ, `w = (0.7, 0.3)`.

1. **Encode.** `z_nodes [50×32]`, `z_global [64]` lấy từ cache.
2. **HL.** Chấm 3 SFC bằng 3 vector Q ∈ ℝ⁴, tính HV từng cái; chọn SFC có HV lớn nhất, gọi là **S**. (Có xác suất ε chọn ngẫu nhiên.)
3. **VNF₁.** `current_source = S.source`. `cpu_mask` cho biết 14 node đủ CPU. LL chấm 14 node → chọn node 17. Dijkstra `S.source → 17`. `allocate` CPU node 17 và băng thông đường đi. Ghi record: trạng thái, `path_cost`, `path_delay`.
4. **VNF₂, VNF₃.** Lặp tương tự; `current_source` cập nhật thành node vừa chọn (VNF₂ có thể được chọn cùng node 17 → `path = [17]`, độ dài 0).
5. **Đoạn cuối.** Dijkstra `v₃ → S.dest`, `allocate` với `cpu = 0`.
6. **Thành công** → tính `deploy_cost`, `load_std`; `commit_sfc`; dựng vector thưởng HL `[−μ·cost, r̄_quality, −load_std, 1]`; lưu 3 transition LL (transition cuối `done = 1`); mỗi transition lưu xong gọi `ll_agent.train_step`. Đặt `graph_dirty = True`.
   **Thất bại** (ví dụ Dijkstra ở VNF₃ không có đường) → `rollback` mọi allocation tạm; HL nhận `failure_vec`; 2 transition LL đã có được lưu với thưởng `failure_vec/2`; SFC bị chặn tới `t + 1`.
7. **HL transition** được lưu và `hl_agent.train_step()` chạy một lần.
8. **Chuyển timestep** khi hàng đợi rỗng/mọi SFC bị chặn: `step_time()` giải phóng SFC hết hạn, nhận SFC mới, encode lại nếu `graph_dirty`.

---

## 10. Hạn chế đã biết của triển khai hiện tại

1. **`w` chỉ là đầu vào, không đi vào target.** Vector thưởng 4 chiều không phụ thuộc `w`, nên mạng không có tín hiệu học rõ ràng về việc `w` thay đổi Q như thế nào; `w` chủ yếu ảnh hưởng qua `r_H` (chỉ log) và qua sự phân bố dữ liệu. Muốn `w` thực sự điều hướng chính sách cần đưa `w` vào cách chọn hành động hoặc vào phần thưởng.
2. **Trạng thái kế tiếp của HL không được encode lại.** `next_z_global` bằng `z_global` hiện tại, dù mạng đã đổi sau commit — target HL xấp xỉ.
3. **`valid_action_mask` của HL luôn toàn `True`**, kể cả SFC đang bị chặn; do đó target có thể tính max trên ứng viên mà thực tế chưa được chọn.
4. **Thất bại tại VNF đầu tiên không tạo transition LL** (xem 4.3).
5. **Thang đo đặc trưng SFC chưa chuẩn hóa** (xem 2.2).
6. **Ngưỡng tham chiếu HV cố định = −1** trong khi thang đo các mục tiêu có thể vượt −1 — ảnh hưởng trực tiếp tới việc chọn hành động (chi tiết ở tài liệu *DRL + Pareto + Hypervolume*).
7. **`sfc_quota_per_timestep` trong config không được sử dụng**: số SFC xử lý mỗi timestep chỉ bị chặn bởi kích thước hàng đợi và cơ chế `blocked_until`.

---

## 11. Bảng tra cứu code

| Chức năng | File | Hàm / lớp |
|---|---|---|
| Vòng lặp episode | `train.py` | `run_episode` |
| Lịch ε, epoch, checkpoint | `train.py` | `main` |
| Chọn SFC | `hl_agent.py` | `HLAgent.select_sfc` |
| Cập nhật Q của HL | `hl_agent.py` | `HLAgent.train_step` |
| Chọn node | `ll_agent.py` | `LLAgent.select_node` |
| Cập nhật Q của LL | `ll_agent.py` | `LLAgent.train_step` |
| Mạng HL / LL | `hl_scorer.py` / `ll_dqn.py` | `HLSharedQScorer` / `LLNodeScorer` |
| Mask node hợp lệ | `nfv_env.py` | `build_ll_mask` |
| Allocate / rollback / commit | `nfv_env.py` | `allocate`, `_rollback_resources`, `commit_sfc` |
| Đồng hồ và hàng đợi | `nfv_env.py` | `step_time`, `_flush_arrivals`, `_purge_expired_queue` |
| Định tuyến | `dijkstra.py` | `custom_dijkstra` |
| Chất lượng đường đi, `load_std` | `dijkstra.py` | `compute_path_cost_delay`, `compute_load_std` |
| Chi phí triển khai | `deploy_cost.py` | `total_deploy_cost` |
| Scalarizer, archive | `pareto.py` | `ParetoScalarizer`, `ParetoArchive` |