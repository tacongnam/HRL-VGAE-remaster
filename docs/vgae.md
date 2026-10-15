> **Ghi chú xác thực nguồn:** Nội dung `models/vgae.py` cung cấp cho lần rà soát này thực chất là bản sao của chính tài liệu này, không phải mã Python thực thi — do đó các chi tiết cài đặt nội bộ của `MNVGAE` (cấu trúc `GCNConv`, `reparameterize`, `GRUCell`, quản lý `_graph_states`...) trong tài liệu dưới đây **chưa được đối chiếu trực tiếp với source code** và được giữ nguyên như bản gốc. Các phần **đã xác minh được** qua cách `train.py` gọi module (`encode_graph`, `vgae(x, ei, graph_id=env.topology_id, update_temporal=...)`, chu kỳ `vgae.train()/eval()`, `train_every_steps`, `z_nodes.detach()/z_global.detach()`) là khớp với mô tả ở mục 6, 7, 9. Nếu có bản `models/vgae.py` thật, cần đối chiếu lại toàn bộ tài liệu này ở checkpoint kế tiếp.

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

**Hiệu ứng thực tế:** Sau mỗi lớp GCN, mỗi node "biết" trạng thái của các node cách 1 hop. Sau 2 lớp, nó biết đến 2 hop.

Chuẩn hóa đối xứng `D̃^(-1/2) Ã D̃^(-1/2)` giúp node bậc cao không "át" giá trị node bậc thấp khi cộng gộp. Với topology Barabási-Albert (có vài hub bậc rất cao), điều này đặc biệt quan trọng.

`GCNConv` của PyTorch Geometric tự thêm self-loop và tự tính chuẩn hóa từ `edge_index`, nên code không phải dựng ma trận kề.

### 2.2. Variational Autoencoder (VAE) — Học phân phối latent

Thay vì học một điểm latent `z`, học một phân phối Gaussian `q(z|x) = N(μ, σ²)`:

```
Encoder: x → μ,  log(σ²)
Sampling: z = μ + σ × ε,  ε ~ N(0, 1)   [reparameterization trick]
```

**Loss function VAE:**

```
L = Reconstruction Loss + KL Divergence
KL = -0.5 × mean(1 + log(σ²) - μ² - σ²)
```

Lưu ý: `kl_loss` (theo tài liệu gốc) dùng `mean` trên toàn bộ `N × d_z` phần tử (bản gốc Kipf & Welling dùng tổng rồi chia cho N) — nếu đúng, KL có trọng số nhỏ hơn so với reconstruction loss so với công thức gốc. *(Chưa đối chiếu được với source thật — xem ghi chú đầu file.)*

### 2.3. Reconstruction Loss — Tái tạo cấu trúc đồ thị (đã xác minh qua `train.py`)

```python
pos_score = (z_nodes[edge_index[0]] * z_nodes[edge_index[1]]).sum(dim=-1)
neg_score = (z_nodes[neg_src] * z_nodes[neg_dst]).sum(dim=-1)
loss = BCE_with_logits(pos_score, ones) + BCE_with_logits(neg_score, zeros)
```

Xác nhận trực tiếp từ `train.py::_compute_recon_loss_sampled`:
- `num_pos = edge_index.shape[1]` — `edge_index` từ `NFVEnvironment._build_edge_index()` chứa cả hai chiều (u→v và v→u), nên `num_pos = 2 × số cạnh`.
- Cạnh âm lấy ngẫu nhiên có hoàn lại bằng `torch.randint(0, num_nodes, ...)` cho cả hai đầu — không loại trừ cạnh thật hay `u == v`.
- `num_neg = max(1, int(num_pos * neg_ratio))`, `neg_ratio = cfg.vgae.neg_sample_ratio` (mặc định 1.0).
- Dùng `binary_cross_entropy_with_logits` nên không áp `sigmoid` thủ công.

---

## 3. Kiến trúc MNVGAE trong dự án

*(Chi tiết forward pass dưới đây chưa đối chiếu với source `models/vgae.py` thật — xem ghi chú đầu file.)*

```
Input: x [N=50, d_in=16]  +  edge_index [2, E]
                │
                ▼
Layer 1: Skip connection + GCNConv
  h1 = ReLU( GCNConv_0(x, edge_index) ) + Linear_skip(x)
  h1: [N, d_h=64]
                │
                ▼
Layer 2: Dual-head VAE encoder
  μ = GCNConv_mu(h1, edge_index)         [N, d_z=32]
  log(σ²) = GCNConv_logvar(h1, edge_index) [N, d_z=32]
  z_nodes = μ + σ·ε (training) / μ (inference)
                │
                ▼
Global pooling → z_global_static
  z_mean = mean(z_nodes, dim=0), z_max = max(z_nodes, dim=0)
  z_global_static = cat([z_mean, z_max])  → [d_g = 64]
                │
                ▼
Temporal GRU (nếu temporal=True)
  prev_state = graph_states.get(graph_id)
  z_global = GRUCell(z_global_static, prev_state)  [64]
```

Đã xác minh: `cfg.d_global = 2 * cfg.vgae.d_latent = 64` (property trong `config.py`), khớp `d_g = 64` ở trên. `cfg.d_ll_input = d_global + 2*d_latent + d_vnf + d_sfc + 2 = 64+64+2+7+2 = 139` và `cfg.d_hl_input = d_global + d_sfc + 2 = 64+7+2 = 73` — khớp input dim của `HLSharedQScorer`/`LLNodeScorer` (`models/hl_scorer.py`, `models/ll_dqn.py`).

Số tham số (bảng dưới **chưa xác minh**, giữ nguyên theo tài liệu gốc):

| Module | Shape | Tham số |
|---|---|---|
| `gcn0` | 16 → 64 (+bias) | 1 088 |
| `skip` | 16 → 64 (không bias) | 1 024 |
| `gcn_mu` | 64 → 32 (+bias) | 2 080 |
| `gcn_logvar` | 64 → 32 (+bias) | 2 080 |
| `temporal_cell` (GRUCell 64→64) | 3 cổng | 24 960 |
| **Tổng** | | **31 232** |

---

## 4. Node features — 16 chiều đầu vào (đã xác minh qua `utils/graph_generator.py`, `env/nfv_env.py`)

Mỗi node được mô tả bởi 16 con số (`cfg.vgae.d_in = 16`), 8 chiều đầu có ý nghĩa, 8 chiều sau là padding `0.0`:

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

Xác nhận trực tiếp từ `utils/graph_generator.py::get_node_features` — khớp chính xác 8 công thức trên, kể cả padding `cfg.vgae.d_in - len(feat)`.

30% số node được đánh dấu **function node** (`nc.function_node_ratio = 0.30`, xác nhận trong `build_substrate_network`) — chỉ node này có thể đặt VNF. Với node không phải function node, `cpu_total = cpu_free = nc.cpu_capacity_mips` vẫn được gán như node thường trong `build_substrate_network` (không phải `cpu_total = 0` như tài liệu gốc từng nói) — `is_function_node` mới là cờ quyết định việc có được chọn đặt VNF hay không (`env.build_ll_mask` kiểm tra `nd.get('is_function_node', True) and nd['cpu_free'] >= cpu_req`), CPU vẫn được cấp đều cho mọi node trong chế độ sinh ngẫu nhiên.

Hai đường tính feature tùy chế độ chạy (`NFVEnvironment._compute_node_features`, đã xác minh trong `env/nfv_env.py`):

| Chế độ | Hàm | Chuẩn hóa CPU |
|---|---|---|
| Ngẫu nhiên (`self._dataset_mode == False`) | `utils.graph_generator.get_node_features` | chia cho `cfg.network.cpu_capacity_mips` |
| Dataset (`self._dataset_mode == True`) | `data.loader.get_node_features_from_graph` | chia cho `max(cpu_total)` của chính đồ thị đó (chưa có source `data/loader.py` để xác minh chi tiết) |

---

## 5. Skip Connection (chưa xác minh trực tiếp — xem ghi chú đầu file)

```python
h1 = F.relu(self.gcn0(x, edge_index)) + self.skip(x)
```

Mục đích: tránh over-smoothing, giữ đặc trưng gốc của node không bị "hòa tan" hoàn toàn vào hàng xóm sau khi tích chập.

---

## 6. Temporal GRU — Ghi nhớ lịch sử

### 6.1. Quản lý trạng thái theo `graph_id` (đã xác minh cách gọi qua `train.py`)

`train.py::encode_graph` gọi `vgae(x, ei, graph_id=env.topology_id, update_temporal=update_temporal)`. `env.topology_id` được set trong `NFVEnvironment.reset()`:
- Ở chế độ dataset: `self.topology_id = topology_id` (tham số truyền vào, do `data/loader.py` tính).
- Ở chế độ ngẫu nhiên: `self.topology_id = None` — tài liệu gốc mô tả trường hợp này VGAE bỏ qua GRU (`z_global = z_global_static`), phù hợp logic vì không có `graph_id` để tra `_graph_states`.

Bảng vòng đời state (giữ nguyên theo tài liệu gốc, chưa xác minh trực tiếp nội bộ `MNVGAE`):

| Tình huống | Hành vi |
|---|---|
| Lần đầu gặp `graph_id` | `prev = z_global_static.detach()` |
| Các lần encode tiếp theo | GRU nhận `(z_global_static, prev)` sinh state mới |
| `update_temporal=True` | State mới được ghi lại (`.detach()`) |
| `update_temporal=False` | Chỉ tính `z_global`, không ghi state |
| `graph_id=None` | Bỏ qua GRU |

### 6.2. Khi nào state được cập nhật? (đã xác minh qua `train.py::run_episode`)

Các điểm gọi `encode_graph(..., update_temporal=True)` (mặc định của hàm) trong `run_episode`:
1. Đầu episode (trước vòng lặp `while not done`).
2. Sau `step_time()` nếu `graph_dirty` hoặc (dataset mode và còn request chưa tới).
3. Ngay sau mỗi lần train VGAE (trong khối train VGAE, gọi `vgae(x, ei, graph_id=..., update_temporal=True)` trực tiếp, không qua `encode_graph`).

Điểm gọi `update_temporal=False` duy nhất: bên trong khối train VGAE, lần forward để tính loss (`vgae(x, ei, graph_id=env.topology_id, update_temporal=False)`) — đúng như tài liệu mô tả, tránh GRU state bị ảnh hưởng bởi gradient pass.

---

## 7. Training VGAE (đã xác minh qua `train.py::run_episode`)

```python
# Khi sfc_processed_since_vgae_train >= cfg.vgae.train_every_steps (mặc định 20):
vgae_optimizer.zero_grad()
vgae.train()
z_nodes_tr, _, mu_tr, logvar_tr = vgae(x, ei, graph_id=env.topology_id, update_temporal=False)
kl = vgae.kl_loss(mu_tr, logvar_tr)
recon_loss = _compute_recon_loss_sampled(z_nodes_tr, ei, env.num_nodes, cfg.vgae.neg_sample_ratio, device)
v_loss = recon_loss + kl
v_loss.backward()
vgae_optimizer.step()
vgae.eval()
with torch.no_grad():
    z_nodes, z_global, mu, logvar = vgae(x, ei, graph_id=env.topology_id, update_temporal=True)
z_nodes = z_nodes.detach(); z_global = z_global.detach()
sfc_processed_since_vgae_train = 0
```

Chi tiết xác minh được:
- `sfc_processed_since_vgae_train` chỉ tăng khi một SFC **thành công** (`sfc_processed_since_vgae_train += 1` nằm trong nhánh `else` sau `commit_sfc`) — SFC thất bại không tính vào bộ đếm này.
- Optimizer: `optim.Adam(vgae.parameters(), lr=cfg.vgae.lr)` (`lr=1e-3` mặc định), tách biệt hoàn toàn với optimizer của HL (`hl_agent.optimizer`) và LL (`ll_agent.optimizer`).
- `encode_graph()` (dùng ở mọi nơi khác ngoài khối train) luôn bọc trong `torch.no_grad()` và trả `z_nodes.detach()`, `z_global.detach()` — cô lập gradient khỏi HL/LL DQN, xác nhận đúng tài liệu gốc.
- Trạng thái `train()/eval()` khi khởi tạo module: đúng như tài liệu gốc mô tả (`nn.Module` mặc định ở chế độ `train`), các lần `encode_graph` gọi **trước** lần train VGAE đầu tiên trong episode đầu tiên của toàn bộ quá trình training sẽ dùng `z = μ + σε` (có nhiễu) vì `vgae.eval()` chưa từng được gọi — sau lần train đầu tiên (bất kỳ episode nào) mới chuyển hẳn sang `eval()` cho đến lần train VGAE kế tiếp (lúc đó lại `vgae.train()` tạm thời rồi `eval()` lại ngay). `evaluate.py` gọi `vgae.eval()` tường minh ngay sau khi load checkpoint nên suy luận luôn xác định.

---

## 8. Output của VGAE và cách dùng (đã xác minh qua `train.py`, `config.py`, `agents/*.py`)

| Output | Shape | Dùng ở đâu |
|---|---|---|
| `z_nodes` | `[N, 32]` | LL Agent: `z_nodes[node_i]` là embedding của node i (`select_node`, `z_prev`/`z_cand`) |
| `z_global` | `[64]` | Cả HL và LL Agent |
| `mu`, `logvar` | `[N, 32]` | Chỉ dùng cho `kl_loss` khi train VGAE |

```
HL input: [z_global(64) | sfc_feat(7) | pareto_w(2)] = 73 chiều   (khớp models/hl_scorer.py)
LL input: [z_global(64) | z_prev(32) | z_cand(32) | vnf_feat(2) | sfc_feat(7) | pareto_w(2)] = 139 chiều  (khớp models/ll_dqn.py)
```

---

## 9. VGAE trong vòng lặp `run_episode` (đã xác minh qua `train.py`)

```
Đầu episode: encode_graph()  → z_nodes, z_global (cache), graph_dirty = False

Trong vòng lặp while not done:
  nếu queue rỗng: step_time(); nếu graph_dirty (hoặc dataset mode còn request tới): encode lại
  nếu đủ điều kiện train VGAE: train 1 bước, encode lại ngay
  HL chọn SFC (dùng z_global cache)
  LL đặt từng VNF của SFC đó (dùng z_nodes/z_global cache — KHÔNG encode lại giữa các VNF)
  commit (thành công): graph_dirty = True, sfc_processed_since_vgae_train += 1
  rollback (thất bại): graph_dirty KHÔNG đổi
  nếu queue vẫn còn: tiếp tục vòng lặp (không step_time)
  nếu queue rỗng: step_time(); encode lại nếu graph_dirty
```

Hệ quả xác nhận đúng từ code: trong lúc đặt K VNF của một SFC, `z_nodes` là ảnh chụp trước VNF đầu tiên của SFC đó (không tính các VNF trước đó **trong cùng SFC**); ràng buộc CPU thực tế vẫn đúng vì `cpu_mask` được tính lại mỗi VNF qua `env.build_ll_mask()` đọc trực tiếp từ `env.G` (đã cập nhật bởi `env.allocate()` sau mỗi VNF).

---

## 10. Lưu ý triển khai và hạn chế

Các điểm 1–5 dưới đây **giữ nguyên theo tài liệu gốc, chưa xác minh với source `models/vgae.py` thật**:

1. GRU chưa được huấn luyện bởi loss nào (loss VGAE chỉ phụ thuộc `z_nodes`, `mu`, `logvar`).
2. Reconstruction loss chỉ nhìn cạnh, không ràng buộc `z` giữ thông tin CPU/băng thông.
3. Phụ thuộc `torch_geometric` (`GCNConv`).
4. State GRU sống trong `dict` Python, không nằm trong `state_dict` khi `torch.save`.
5. Kích thước `N` cố định 50 node ở chế độ ngẫu nhiên; GCN không phụ thuộc `N` nên dùng được cho dataset mode với số node khác.

---

## 11. Tóm tắt hyperparameter và ánh xạ code

| Tham số (`cfg.vgae`) | Giá trị | Ý nghĩa |
|---|---|---|
| `d_in` | 16 | Chiều feature mỗi node (8 dùng + 8 padding) — xác minh |
| `d_hidden` | 64 | Chiều lớp GCN đầu |
| `d_latent` | 32 | Chiều `z` mỗi node — xác minh qua `cfg.d_latent` |
| `lr` | 1e-3 | Learning rate Adam cho VGAE — xác minh qua `train.py` |
| `neg_sample_ratio` | 1.0 | Số cạnh âm / số cạnh dương — xác minh |
| `train_every_steps` | 20 | Số SFC thành công giữa hai lần train VGAE — xác minh |
| `temporal` | True | Bật GRU (chưa xác minh cách dùng nội bộ) |
| `temporal_hidden` | 64 | Khai báo trong config nhưng theo tài liệu gốc **không dùng** — chiều GRU thực tế = `2×d_latent=64` (khớp `cfg.d_global`, nhưng chưa xác minh trực tiếp trong `MNVGAE`) |

| Chức năng | File / hàm | Trạng thái xác minh |
|---|---|---|
| Encoder, pooling, GRU, KL | `models/vgae.py` — `MNVGAE` | Chưa (không có source thật) |
| Tính node features (ngẫu nhiên) | `utils/graph_generator.py` — `get_node_features` | Đã xác minh |
| Tính node features (dataset) | `data/loader.py` — `get_node_features_from_graph` | Chưa có source |
| `topology_id` | `data/loader.py` — `compute_topology_id` | Chưa có source |
| Xây `edge_index` (hai chiều) | `env/nfv_env.py` — `_build_edge_index` | Đã xác minh |
| Encode + cache | `train.py` — `encode_graph` | Đã xác minh |
| Reconstruction loss | `train.py` — `_compute_recon_loss_sampled` | Đã xác minh |
| Bước train VGAE | `train.py` — `run_episode` | Đã xác minh |