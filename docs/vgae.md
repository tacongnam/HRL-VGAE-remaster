# Mã hóa đồ thị mạng với VGAE

## 1. Vấn đề: Làm sao mô tả trạng thái mạng?

Trước khi agent có thể ra quyết định, nó cần "nhìn thấy" mạng. Nhưng mạng là một đồ thị — 50 node, ~150 cạnh — không phải một vector số đơn giản mà các mạng nơ-ron thông thường có thể ăn vào.

Thách thức đặt ra:

- **Cấu trúc đồ thị**: node kết nối với node, thông tin lan truyền qua các cạnh.
- **Kích thước thay đổi**: các episode dùng topology khác nhau.
- **Thay đổi liên tục**: sau mỗi SFC được đặt, CPU và băng thông thay đổi.
- **Cần biểu diễn compact**: agent không thể xử lý ma trận kề 50×50 mỗi bước.

Giải pháp: **VGAE** (Variational Graph Autoencoder) — một mạng nơ-ron học cách "nén" đồ thị thành các vector ý nghĩa (latent representation).

---

## 2. Nền tảng lý thuyết

### 2.1. Graph Convolutional Network (GCN) — Mạng tích chập đồ thị

**Ý tưởng gốc:** Trong ảnh, một pixel được tích chập với các pixel lân cận. Trong đồ thị, một node được "tổng hợp" thông tin từ các node hàng xóm.

Công thức một lớp GCN:

```
H^(l+1) = σ( D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l) )
```

Trong đó:
- `H^(l)` — ma trận đặc trưng ở lớp l, shape `[N, d_l]`
- `Ã = A + I` — ma trận kề có thêm self-loop (để mỗi node cũng giữ thông tin của chính nó)
- `D̃` — ma trận bậc (degree matrix) của Ã, dùng để chuẩn hóa
- `W^(l)` — ma trận trọng số có thể học được
- `σ` — hàm kích hoạt (ReLU)

**Hiệu ứng thực tế:** Sau mỗi lớp GCN, mỗi node "biết" trạng thái của các node cách 1 hop. Sau 2 lớp, nó biết đến 2 hop. Điều này quan trọng vì khi chọn node đặt VNF, agent cần biết không chỉ tình trạng node đó mà cả vùng lân cận.

Chuẩn hóa đối xứng `D̃^(-1/2) Ã D̃^(-1/2)` có tác dụng: node có nhiều hàng xóm (bậc cao) không bị "át" giá trị của các node bậc thấp khi cộng gộp. Với topology Barabási-Albert (có vài hub bậc rất cao), điều này đặc biệt quan trọng.

**Trong code** (`vgae.py`):
```python
self.gcn0 = GCNConv(d_in, d_h)   # d_in=16 → d_h=64
```

`GCNConv` của PyTorch Geometric tự thêm self-loop và tự tính chuẩn hóa từ `edge_index`, nên code không phải dựng ma trận kề.

### 2.2. Variational Autoencoder (VAE) — Học phân phối latent

**Vấn đề với encoder thông thường:** Nếu chỉ học một điểm latent `z` cho mỗi node, mô hình dễ overfit — chỉ ghi nhớ dữ liệu training chứ không học được cấu trúc tổng quát.

**Giải pháp VAE:** Thay vì học một điểm `z`, học một phân phối Gaussian `q(z|x) = N(μ, σ²)`:

```
Encoder: x → μ,  log(σ²)
Sampling: z = μ + σ × ε,  ε ~ N(0, 1)   [reparameterization trick]
```

**Reparameterization trick** là kỹ thuật quan trọng: thay vì lấy mẫu trực tiếp từ `N(μ, σ²)` (không thể backprop qua), ta lấy mẫu `ε ~ N(0,1)` (hằng số ngẫu nhiên) rồi tính `z = μ + σε`. Gradient chạy qua μ và σ bình thường.

**Loss function VAE:**

```
L = Reconstruction Loss + KL Divergence

KL = -0.5 × mean(1 + log(σ²) - μ² - σ²)
```

KL divergence ép phân phối `q(z|x)` gần với prior `N(0,1)`, tránh mô hình "gian lận" bằng cách dùng σ→0 để ép z = μ.

**Trong code:**
```python
# Hai GCNConv song song, mỗi cái output một chiều khác nhau
self.gcn_mu     = GCNConv(d_h, d_z)   # học μ
self.gcn_logvar = GCNConv(d_h, d_z)   # học log(σ²)

def reparameterize(self, mu, logvar):
    if self.training:
        std = torch.exp(0.5 * logvar)   # σ = exp(0.5 × log σ²)
        eps = torch.randn_like(std)      # ε ~ N(0,1)
        return mu + std * eps            # z = μ + σε
    return mu                            # inference: dùng luôn μ (ổn định)

def kl_loss(self, mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
```

Lưu ý: `kl_loss` dùng `mean` trên toàn bộ `N × d_z` phần tử (bản gốc của Kipf & Welling dùng tổng rồi chia cho N). Hệ quả là KL có trọng số khá nhỏ so với reconstruction loss, mô hình gần với một graph autoencoder thường hơn là một VAE "chặt".

### 2.3. Reconstruction Loss — Tái tạo cấu trúc đồ thị

VGAE được huấn luyện để tái tạo lại đồ thị từ z: nếu node u và node v kết nối, tích vô hướng `z_u · z_v` phải lớn; nếu không kết nối, phải nhỏ.

```
Reconstruction: P(A_uv = 1) = sigmoid(z_u · z_v)
```

Đây là một **inner-product decoder** — decoder không có tham số học được, mọi "kiến thức" nằm trong encoder.

Trong training, lấy ngẫu nhiên số cạnh âm (không tồn tại) bằng số cạnh dương (tồn tại) để tránh class imbalance (mạng thưa: ~150 cạnh trên 1225 cặp node khả dĩ).

**Trong code** (`train.py`, hàm `_compute_recon_loss_sampled`):
```python
pos_score = (z_nodes[edge_index[0]] * z_nodes[edge_index[1]]).sum(dim=-1)
neg_score = (z_nodes[neg_src] * z_nodes[neg_dst]).sum(dim=-1)
loss = BCE_with_logits(pos_score, ones) + BCE_with_logits(neg_score, zeros)
```

Chi tiết:

- Dùng `binary_cross_entropy_with_logits` nên **không** áp `sigmoid` thủ công — hàm này tự áp sigmoid ở bên trong, ổn định số học hơn.
- `edge_index` từ `NFVEnvironment._build_edge_index()` đã chứa cả hai chiều (u→v và v→u), nên `num_pos = 2 × số cạnh`.
- Cạnh âm được lấy ngẫu nhiên có hoàn lại từ `randint(0, N)` cho cả hai đầu. Cách này rẻ (O(E)) nhưng không loại trừ cạnh thật hay cặp `u == v`; với đồ thị thưa, xác suất bốc trúng cạnh thật khoảng 12% (300/2500 cặp có hướng) và được chấp nhận như một xấp xỉ.
- Số cạnh âm = `num_pos × neg_sample_ratio` (mặc định 1.0).

---

## 3. Kiến trúc MNVGAE trong dự án

```
Input: x [N=50, d_in=16]  +  edge_index [2, E]
                │
                ▼
┌────────────────────────────────────────────────┐
│  Layer 1: Skip connection + GCNConv           │
│                                               │
│  h1 = ReLU( GCNConv_0(x, edge_index) )       │
│       + Linear_skip(x)                        │
│  Skip giúp gradient chảy thẳng, giữ được     │
│  đặc trưng gốc của từng node                 │
│  h1: [N, d_h=64]                             │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│  Layer 2: Dual-head VAE encoder               │
│                                               │
│  μ       = GCNConv_mu(h1, edge_index)        │  [N, d_z=32]
│  log(σ²) = GCNConv_logvar(h1, edge_index)   │  [N, d_z=32]
│                                               │
│  z_nodes = μ + σ × ε  (training)            │
│          = μ           (inference)           │
│  z_nodes: [N=50, d_z=32]                    │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│  Global pooling → z_global_static             │
│                                               │
│  z_mean = mean(z_nodes, dim=0)  [d_z=32]    │
│  z_max  = max(z_nodes,  dim=0)  [d_z=32]    │
│  z_global_static = cat([z_mean, z_max])      │
│                  → [d_g = 64]               │
│                                               │
│  Dùng cả mean (thông tin trung bình)         │
│  và max (thông tin đỉnh điểm) để tóm tắt   │
│  toàn bộ đồ thị trong 64 số                 │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│  Temporal GRU (nếu temporal=True)             │
│                                               │
│  Mỗi topology có một "memory" riêng          │
│                                               │
│  prev_state = graph_states.get(graph_id)     │
│  new_state = GRUCell(z_global_static,        │
│                      prev_state)             │
│  z_global = new_state  [64]                  │
│                                               │
│  GRU ghi nhớ lịch sử thay đổi của mạng      │
│  (VNF đặt trước, tài nguyên đã tiêu thụ)    │
└────────────────────────────────────────────────┘
                │
         z_nodes [50, 32]
         z_global [64]
```

Số tham số (với cấu hình mặc định `16 → 64 → 32`):

| Module | Shape | Tham số |
|---|---|---|
| `gcn0` | 16 → 64 (+bias) | 1 088 |
| `skip` | 16 → 64 (không bias) | 1 024 |
| `gcn_mu` | 64 → 32 (+bias) | 2 080 |
| `gcn_logvar` | 64 → 32 (+bias) | 2 080 |
| `temporal_cell` (GRUCell 64→64) | 3 cổng | 24 960 |
| **Tổng** | | **31 232** |

Đây là encoder rất nhỏ so với hai Q-network (~52K và ~69K tham số), phù hợp với việc gọi lại nhiều lần trong một episode.

---

## 4. Node features — 16 chiều đầu vào

Mỗi node được mô tả bởi 16 con số (`cfg.vgae.d_in = 16`), trong đó **8 chiều đầu có ý nghĩa**, 8 chiều sau là padding `0.0` dành cho mở rộng:

| Index | Tên | Công thức | Ý nghĩa |
|---|---|---|---|
| 0 | cpu_util | `1 - cpu_free / cpu_total` | Tỷ lệ CPU đang dùng |
| 1 | cpu_free_norm | `cpu_free / cpu_capacity` | CPU còn trống (chuẩn hóa) |
| 2 | cpu_total_norm | `cpu_total / cpu_capacity` | CPU tổng (thường = 1.0) |
| 3 | is_function_node | `0.0` hoặc `1.0` | Có thể đặt VNF không? |
| 4 | avg_bw_util | trung bình `1 - bw_free/bw_total` các cạnh kề | Tắc nghẽn băng thông lân cận |
| 5 | degree_norm | `degree / (N-1)` | Mức độ kết nối |
| 6 | proc_delay_norm | `proc_delay / 5.0` | Độ trễ xử lý tại node |
| 7 | cost_cpu_norm | `cost_cpu / 2.0` | Đơn giá CPU tại node |
| 8–15 | (padding) | `0.0` | Dành cho mở rộng sau |

30% số node được đánh dấu là **function node** (`is_function_node = True`) — chỉ những node này mới có thể đặt VNF. Điều này phản ánh thực tế: không phải thiết bị mạng nào cũng có khả năng chạy phần mềm ảo hóa. Với node không phải function node, `cpu_total = 0`, nên `cpu_util = 0` và `cpu_free_norm = 0`.

Có **hai đường tính feature** tùy chế độ chạy của môi trường (`NFVEnvironment._compute_node_features`):

| Chế độ | Hàm | Chuẩn hóa CPU |
|---|---|---|
| Ngẫu nhiên (`reset()` không tham số) | `graph_generator.get_node_features` | chia cho `cfg.network.cpu_capacity_mips` |
| Dataset (`reset(G, requests, ...)`) | `loader.get_node_features_from_graph` | chia cho `max(cpu_total)` của chính đồ thị đó |

Cách thứ hai cần thiết vì topology từ dataset có thể có dung lượng CPU khác nhau giữa các node và giữa các file; chia cho max giúp `cpu_free_norm`, `cpu_total_norm` luôn nằm trong `[0, 1]`.

**Trong code** (`graph_generator.py`):
```python
feat = [cpu_util, cpu_free_norm, cpu_total_norm, is_fn, avg_bw_util,
        degree_norm, proc_delay_norm, cost_cpu_norm]
pad = cfg.vgae.d_in - len(feat)   # padding zeros đến 16
```

---

## 5. Tại sao cần Skip Connection?

```python
h1 = F.relu(self.gcn0(x, edge_index)) + self.skip(x)
```

Không dùng skip: GCN có thể "quên" đặc trưng gốc của node sau khi trộn với hàng xóm. Node cuối cùng chỉ còn thông tin từ hàng xóm mà thiếu thông tin về chính nó (hiện tượng over-smoothing — đặc biệt rõ ở các hub của đồ thị Barabási-Albert, nơi hầu hết node đều cách nhau vài hop).

Dùng skip: `Linear_skip(x)` giữ lại thông tin ban đầu của node. Kết quả là `h1` chứa cả: thông tin từ hàng xóm (GCN) + thông tin bản thân (skip).

Lưu ý thứ tự trong code: skip được cộng **sau** ReLU (`relu(gcn0(x)) + skip(x)`), nên `h1` có thể âm ở những chiều mà nhánh skip âm — đây là lựa chọn có chủ đích để nhánh skip không bị cắt bởi ReLU.

---

## 6. Temporal GRU — Ghi nhớ lịch sử

Tại sao cần GRU? Xét một kịch bản: trong cùng một episode, 10 VNF đã được đặt trước đó lên node A. Nếu encode lại đồ thị lúc này, node A sẽ có `cpu_util` cao hơn — VGAE "nhìn thấy" điều này qua node features.

Nhưng **xu hướng thay đổi** (đang bắt đầu tắc nghẽn hay đang giải phóng tài nguyên?) không nằm trong snapshot hiện tại. GRU ghi nhớ điều này:

```python
prev_state = self._graph_states.get(graph_id)   # trạng thái từ bước trước
new_state = GRUCell(z_global_static, prev_state) # cập nhật bộ nhớ
z_global = new_state
```

### 6.1. Quản lý trạng thái theo `graph_id`

`MNVGAE` giữ một dictionary `_graph_states: {graph_id → tensor[64]}`. Mỗi topology có một `graph_id` riêng → bộ nhớ riêng.

`graph_id` là `topology_id` do `loader.compute_topology_id` sinh ra: một mã MD5 (16 ký tự đầu) của **danh sách node đã sắp xếp + danh sách cạnh đã sắp xếp**. Hai file episode có cùng cấu trúc mạng (dù khác SFC trace) sẽ dùng chung một bộ nhớ; hai topology khác nhau không bao giờ lẫn state.

Vòng đời của state:

| Tình huống | Hành vi |
|---|---|
| Lần đầu gặp `graph_id` | `prev = z_global_static.detach()` — khởi tạo từ chính quan sát hiện tại |
| Các lần encode tiếp theo | GRU nhận `(z_global_static, prev)` và sinh state mới |
| `update_temporal=True` | State mới được ghi lại (`.detach()`) |
| `update_temporal=False` | Chỉ tính `z_global`, **không** ghi state |
| `graph_id=None` (chế độ ngẫu nhiên) | Bỏ qua GRU, `z_global = z_global_static` |
| `reset_temporal_state(graph_id)` | Xóa state của một topology (hiện chưa được gọi trong `train.py`) |

Khi episode kết thúc, bộ nhớ không bị xóa ngay. Vì `--passes-per-file 5` replay cùng một file (cùng topology_id) nhiều lần liên tiếp, state của pass sau kế thừa từ pass trước — GRU có "ngữ cảnh dài hạn" nhưng cũng đồng nghĩa lần encode đầu tiên của một episode mới không bắt đầu từ trạng thái mạng "sạch".

### 6.2. Khi nào state được cập nhật?

State tiến lên một bước **mỗi lần gọi `encode_graph(..., update_temporal=True)`** (mặc định), tức là:

1. Đầu episode.
2. Sau mỗi lần `step_time()` nếu đồ thị đã thay đổi (`graph_dirty`, do có SFC được commit) hoặc còn request chưa đến (dataset mode).
3. Ngay sau mỗi lần train VGAE (mỗi 20 SFC thành công).

Nghĩa là "một bước thời gian" của GRU không đồng nghĩa với một timestep của môi trường mà là một lần mạng được quan sát lại.

---

## 7. Training VGAE

VGAE được huấn luyện **trong vòng lặp training chính**, không phải pre-train riêng:

```python
# Cứ sau 20 SFC được xử lý thành công (cfg.vgae.train_every_steps = 20):
vgae.train()
z_nodes_tr, _, mu_tr, logvar_tr = vgae(x, ei, update_temporal=False)
kl = vgae.kl_loss(mu_tr, logvar_tr)
recon = reconstruction_loss(z_nodes_tr, edge_index, num_neg)
loss = recon + kl
loss.backward()
optimizer.step()

# Sau đó update z_nodes, z_global dùng cho agent
vgae.eval()
z_nodes, z_global, _, _ = vgae(x, ei, update_temporal=True)
```

Lưu ý `update_temporal=False` khi training để tránh GRU state bị cập nhật bởi gradient pass.

Chi tiết cần biết:

- **Mỗi lần update chỉ là một gradient step trên một đồ thị** (đồ thị hiện tại của episode). Không có mini-batch nhiều đồ thị; tổng số update = `số SFC thành công / 20` mỗi episode.
- **Optimizer:** `Adam(vgae.parameters(), lr=1e-3)`, tách biệt với optimizer của HL/LL.
- **Chế độ train/eval:** `vgae.train()` bật lấy mẫu `μ + σε`; ngay sau bước update code gọi `vgae.eval()` để agent dùng `z = μ` (xác định, ổn định). Khi khởi tạo, module ở chế độ `train` mặc định của `nn.Module`, nên các lần encode **trước** lần train VGAE đầu tiên vẫn dùng z có nhiễu; từ lần update đầu trở đi mới chuyển sang `μ`. `evaluate.py` gọi `vgae.eval()` tường minh nên suy luận luôn xác định.
- **Cô lập gradient:** `encode_graph` bọc trong `torch.no_grad()` và trả về `z_nodes.detach()`, `z_global.detach()`. Loss của HL/LL DQN **không** lan ngược vào VGAE — VGAE chỉ học từ reconstruction + KL.

---

## 8. Output của VGAE và cách dùng

| Output | Shape | Dùng ở đâu |
|---|---|---|
| `z_nodes` | `[50, 32]` | LL Agent: `z_nodes[node_i]` là embedding của node i |
| `z_global` | `[64]` | Cả HL và LL Agent: trạng thái toàn cục mạng |
| `mu` | `[50, 32]` | KL loss khi training VGAE |
| `logvar` | `[50, 32]` | KL loss khi training VGAE |

**HL Agent** nhận `z_global` (64 chiều) để biết trạng thái chung của mạng, kết hợp với đặc trưng SFC để quyết định chọn yêu cầu nào.

**LL Agent** nhận cả `z_global`, embedding của node hiện tại `z_prev` (node đặt VNF trước đó, hoặc source của SFC với VNF đầu tiên) và embedding của node ứng viên `z_cand`, để biết vừa trạng thái toàn cục vừa vị trí tương đối giữa node trước và node ứng viên.

```
HL input: [z_global(64) | sfc_feat(7) | pareto_w(2)] = 73 chiều
LL input: [z_global(64) | z_prev(32) | z_cand(32) | vnf_feat(2) | sfc_feat(7) | pareto_w(2)] = 139 chiều
```

Thực tế `d_ll_input = d_global + 2*d_latent + d_vnf + d_sfc + 2 = 64 + 64 + 2 + 7 + 2 = 139`.

---

## 9. VGAE trong vòng lặp `run_episode`

VGAE không được gọi ở mỗi quyết định mà được **cache** và chỉ encode lại khi cần, vì mỗi lần encode tốn một forward pass GCN + GRU:

```
Bắt đầu episode
  └─ encode_graph()                      → z_nodes, z_global (cache)

Trong vòng lặp:
  ├─ HL chọn SFC      ┐
  ├─ LL đặt K VNF     ├─ dùng z_nodes / z_global từ cache (KHÔNG encode lại
  └─ commit/rollback  ┘   giữa các VNF của cùng một SFC)
        │
        └─ nếu commit thành công: graph_dirty = True

  Khi hàng đợi rỗng hoặc hết SFC xử lý trong timestep:
  └─ step_time() → giải phóng SFC hết hạn, nhận SFC mới
       └─ nếu graph_dirty (hoặc còn request sắp đến ở dataset mode):
            encode_graph() lại           → làm mới cache

  Mỗi 20 SFC thành công:
  └─ 1 bước train VGAE, rồi encode lại
```

Hệ quả thiết kế đáng lưu ý:

- **Trong lúc đặt K VNF của một SFC, `z_nodes` là "ảnh chụp" trước khi đặt VNF đầu tiên.** CPU và băng thông đã tiêu thụ bởi các VNF trước đó chưa được phản ánh trong embedding. Thông tin này được bù bằng `z_prev` (vị trí đã chọn) và `cpu_mask` (ràng buộc CPU thực tế, luôn được tính từ đồ thị hiện tại).
- **SFC thất bại không làm `graph_dirty = True`** vì tài nguyên đã được rollback, đồ thị không đổi nên không cần encode lại.
- Vì `x` (node features) được tính lại từ `env.G` mỗi khi gọi `get_node_features_tensor`, đồ thị dùng cho reconstruction loss luôn khớp với trạng thái mạng lúc encode.

---

## 10. Lưu ý triển khai và hạn chế

1. **GRU chưa được huấn luyện bởi bất kỳ loss nào.** Loss của VGAE (`recon + KL`) chỉ phụ thuộc `z_nodes`, `mu`, `logvar`; `z_global` (đầu ra của GRU) không tham gia loss nào, và các Q-network nhận `z_global.detach()`. Do đó trọng số `temporal_cell` giữ nguyên giá trị khởi tạo ngẫu nhiên, GRU hoạt động như một bộ nhớ tái diễn ngẫu nhiên cố định (giống reservoir). Nếu muốn GRU học được, cần cho gradient của Q-loss chảy vào `z_global` hoặc thêm loss phụ trợ (ví dụ dự đoán feature bước kế tiếp).
2. **Reconstruction loss chỉ nhìn vào cạnh**, không yêu cầu `z` giữ lại thông tin CPU/băng thông. Thông tin tài nguyên đi vào `z` chỉ qua `x` → GCN và không bị ràng buộc phải được bảo toàn. Để `z` mang thông tin tài nguyên rõ hơn có thể thêm một head tái tạo feature (`x̂`) vào loss.
3. **Phụ thuộc `torch_geometric`.** `GCNConv` là dependency ngoài PyTorch; cần cài đặt phiên bản tương thích với PyTorch đang dùng.
4. **State GRU sống trong `dict` Python, không nằm trong `state_dict`.** `torch.save` checkpoint chỉ lưu trọng số; `_graph_states` không được lưu. Khi `evaluate.py` load checkpoint, mỗi topology bắt đầu với state khởi tạo từ `z_global_static` — có thể khác phân phối so với lúc train (khi state đã tích lũy qua nhiều episode).
5. **Kích thước `N` cố định theo config trong chế độ ngẫu nhiên** (50 node), nhưng bản thân GCN không phụ thuộc `N`, nên cùng trọng số dùng được cho topology có số node khác nhau trong dataset mode.

---

## 11. Tóm tắt hyperparameter và ánh xạ code

| Tham số (`cfg.vgae`) | Giá trị | Ý nghĩa |
|---|---|---|
| `d_in` | 16 | Chiều feature mỗi node (8 dùng + 8 padding) |
| `d_hidden` | 64 | Chiều lớp GCN đầu |
| `d_latent` | 32 | Chiều `z` mỗi node |
| `lr` | 1e-3 | Learning rate của Adam cho VGAE |
| `neg_sample_ratio` | 1.0 | Số cạnh âm / số cạnh dương |
| `train_every_steps` | 20 | Số SFC thành công giữa hai lần train VGAE |
| `temporal` | True | Bật GRU |
| `temporal_hidden` | 64 | Khai báo trong config nhưng **không dùng** — chiều GRU thực tế là `2 × d_latent = 64` |

| Chức năng | File / hàm |
|---|---|
| Encoder, pooling, GRU, KL | `vgae.py` — `MNVGAE` |
| Tính node features (ngẫu nhiên) | `graph_generator.py` — `get_node_features` |
| Tính node features (dataset) | `loader.py` — `get_node_features_from_graph` |
| `topology_id` | `loader.py` — `compute_topology_id` |
| Xây `edge_index` (hai chiều) | `nfv_env.py` — `_build_edge_index` |
| Encode + cache | `train.py` — `encode_graph` |
| Reconstruction loss | `train.py` — `_compute_recon_loss_sampled` |
| Bước train VGAE | `train.py` — `run_episode` |