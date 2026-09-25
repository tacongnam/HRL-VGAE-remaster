# DRL đa mục tiêu: Q-learning vector, Pareto và Hypervolume

> **Ghi chú xác thực (checkpoint rà soát):** Toàn bộ nội dung dưới đây đã được đối chiếu với source thật `utils/pareto.py`, `train.py`, `agents/hl_agent.py`, `agents/ll_agent.py`, `models/hl_scorer.py`, `models/ll_dqn.py`, `utils/dijkstra.py`, `utils/deploy_cost.py`, `env/nfv_env.py`. Các công thức, bug `_hv_wfg`, và mô tả luồng transition đều khớp code thực tế; các sửa/bổ sung được đánh dấu **[SỬA]** hoặc **[BỔ SUNG]** inline.

Tài liệu này giải thích ba lớp kiến thức xếp chồng lên nhau trong dự án, đi từ nền tảng đến chi tiết cài đặt:

1. **DRL (Deep Q-Learning)** — cách agent học giá trị của hành động bằng mạng nơ-ron.
2. **Tối ưu đa mục tiêu Pareto** — khi phần thưởng là một *vector*, "tốt hơn" nghĩa là gì.
3. **Hypervolume (HV)** — cách biến "độ tốt" của một vector/tập vector thành một số để chọn hành động.

Các mảng này cùng nằm trong `agents/hl_agent.py`, `agents/ll_agent.py`, `models/hl_scorer.py`, `models/ll_dqn.py` và `utils/pareto.py`.

---

## 1. Nền tảng: từ Q-learning đến DQN

### 1.1. Q-function và phương trình Bellman

Agent tương tác với môi trường theo chu kỳ: quan sát trạng thái `s`, chọn hành động `a`, nhận phần thưởng `r` và chuyển sang `s'`. Mục tiêu là cực đại hóa tổng thưởng chiết khấu:

```
G = r₀ + γ·r₁ + γ²·r₂ + ...          (0 < γ < 1)
```

**Q-function** `Q(s, a)` là giá trị kỳ vọng của `G` nếu làm `a` ở `s` rồi tiếp tục theo chính sách tối ưu. Nó thỏa phương trình Bellman:

```
Q(s, a) = r + γ · max_{a'} Q(s', a')
```

Q-learning cập nhật Q về phía vế phải (gọi là **target**). Khi không gian trạng thái quá lớn để lưu bảng, dùng mạng nơ-ron `Q_θ(s, a)` xấp xỉ và huấn luyện bằng hồi quy:

```
loss = ( Q_θ(s, a) − target )²
```

### 1.2. Ba "mẹo" giúp DQN ổn định

| Kỹ thuật | Vấn đề giải quyết | Trong code |
|---|---|---|
| **Replay buffer** | Các mẫu liên tiếp tương quan mạnh; học trực tiếp từ chúng làm mạng dao động | `HLReplayBuffer`, `LLReplayBuffer` (`agents/hl_agent.py`, `agents/ll_agent.py`): `deque(maxlen)`, lấy mẫu `random.sample(batch_size = 64)` |
| **Target network** | Target phụ thuộc chính tham số đang học → "đuổi theo cái đuôi" | `target_net`, sao chép từ `q_net` mỗi `target_update_freq = 50` lần cập nhật |
| **ε-greedy** | Cân bằng khám phá / khai thác | `select_sfc`, `select_node`: xác suất ε chọn ngẫu nhiên trong tập hợp lệ |

Ngoài ra code dùng `clip_grad_norm_(…, 10.0)` để chặn gradient bùng nổ, `Adam(lr = 5e-4)`, `γ = 0.95` — tất cả xác nhận đúng trong `QNetConfig` (`config.py`) và `train_step()` của cả hai agent.

### 1.3. Mạng Q kiểu "scorer"

DQN chuẩn có một đầu ra cho mỗi hành động. Ở đây tập hành động thay đổi liên tục (số SFC trong hàng đợi, số node còn đủ CPU), nên **hành động được đưa vào đầu vào** của mạng:

```
DQN chuẩn:      Q_θ(s)      → [Q(s,a₁), Q(s,a₂), …, Q(s,a_n)]     (n cố định)
Scorer (dự án): Q_θ(s, a)   → Q(s,a)                              (a là input, gọi n lần)
```

Khi chọn hành động, tất cả `m` ứng viên được chấm trong **một batch** (`z_global`, `pareto_w` được `expand` thành `m` hàng, chỉ đặc trưng ứng viên khác nhau — xác nhận qua `HLAgent.select_sfc` và `LLAgent.select_node`). Khi học, hành động đã thực hiện được lưu dưới dạng *đặc trưng của nó* (`sfc_feat` với HL; `z_cand = z_nodes[node]` với LL) nên `current_q = q_net(state, a_đã_chọn)` là một phép truyền xuôi thông thường — không cần `gather` theo chỉ số.

Ưu điểm: dùng được cho hành động có số lượng biến thiên, và tự động tổng quát hóa qua các node/SFC cùng loại (chia sẻ trọng số). Nhược điểm: chi phí suy luận tỷ lệ với số ứng viên.

---

## 2. Mở rộng sang đa mục tiêu: Q-vector

### 2.1. Vì sao không gộp thành một số?

Bốn mục tiêu của hệ thống (chi phí, trễ/tắc nghẽn, cân bằng tải, thành công) **mâu thuẫn nhau**, và thang đo khác nhau. Nếu cộng có trọng số `Σ wᵢ·rᵢ` ngay từ đầu:

- Phải chọn `w` trước, mà không ai biết `w` nào là đúng.
- Weighted sum chỉ tìm được các nghiệm nằm trên phần **lồi** của mặt Pareto.
- Mỗi lần đổi ưu tiên lại phải huấn luyện lại.

**Hướng tiếp cận của dự án:** giữ nguyên **vector** phần thưởng `r ∈ ℝ⁴` và học **vector** giá trị:

```
Q(s, a) = [ Q_cost, Q_delay, Q_balance, Q_success ] ∈ ℝ⁴
```

Mạng có **một thân chung và 4 head tuyến tính** (mỗi head ước lượng một mục tiêu) — xác nhận đúng `models/hl_scorer.py::HLSharedQScorer` và `models/ll_dqn.py::LLNodeScorer`:

```python
h = self.trunk(x)                                           # [batch, 128]
return torch.cat([head(h) for head in self.q_heads], -1)    # [batch, 4]
```

Bellman dạng vector (cộng vector, nhân vô hướng `γ`) vẫn hợp lệ theo từng thành phần:

```
Q(s, a) = r + γ · Q(s', a*)         với r, Q ∈ ℝ⁴
```

Nhưng phép `max_{a'}` **không xác định** với vector — không có thứ tự toàn phần trên ℝ⁴. Hai câu hỏi cần trả lời:

1. Khi *chọn hành động*, so sánh `Q(s, a₁)` với `Q(s, a₂)` thế nào? → **Pareto dominance + Hypervolume** (mục 3–5).
2. Khi *tính target*, "hành động tốt nhất kế tiếp" là gì? → **tập Pareto các Q-vector kế tiếp** (mục 6).

---

## 3. Pareto dominance

### 3.1. Định nghĩa

Vector `a` **dominates** (trội hơn) `b` nếu `a` không tệ hơn `b` ở **mọi** mục tiêu và tốt hơn hẳn ở **ít nhất một** mục tiêu (với quy ước maximize):

```
a ≻ b   ⇔   ∀i: aᵢ ≥ bᵢ   VÀ   ∃j: aⱼ > bⱼ
```

Hai vector mà không cái nào trội hơn cái kia là **không so sánh được** (incomparable): mỗi cái tốt hơn ở một số mục tiêu. **Tập không bị trội** (non-dominated set) của một tập vector là các phần tử không bị phần tử nào khác trong tập trội hơn; khi áp dụng lên toàn bộ không gian nghiệm gọi là **mặt Pareto** (Pareto front).

### 3.2. Cài đặt (`utils/pareto.py`)

```python
def dominates(a, b):
    return bool(np.all(a >= b) and np.any(a > b))

def non_dominated_indices(points):           # O(n² · d)
    keep = []
    for i in range(n):
        dominated = any(j != i and dominates(points[j], points[i]) for j in range(n))
        if not dominated:
            keep.append(i)
    return keep
```

Điểm cần lưu ý:

- Độ phức tạp `O(n²·d)`; với `n` ≤ vài chục ứng viên là chấp nhận được.
- Hai vector **bằng nhau** không trội nhau (cần `np.any(a > b)`), nên **cả hai đều được giữ** trong tập không bị trội.
- Có thêm hàm phụ: `pareto_rank` (đếm số vector trội mình), `pareto_front` (cho danh sách điểm 2D) và `select_by_pareto_dominance` (lựa chọn cũ dựa trên tích vô hướng với `w`, giữ lại để tương thích ngược — **không được `train.py` hay hai agent gọi tới ở đường chạy chính**, chỉ tồn tại như API dự phòng).

---

## 4. Hypervolume

### 4.1. Ý tưởng

Dominance chỉ cho biết "tốt hơn / không so sánh được". Để **xếp hạng** các hành động ta cần một thang vô hướng nhưng vẫn tôn trọng thứ tự Pareto. **Hypervolume** (HV) là thước đo chuẩn: thể tích vùng của không gian mục tiêu bị các điểm "chiếm" tính từ một **điểm tham chiếu** (reference point) `ref`.

```
HV(P, ref) = thể tích của hợp các hộp  [ref, p]  với p ∈ P
```

Tính chất khiến HV lý tưởng cho việc này:

- **Đơn điệu Pareto:** nếu `a ≻ b` thì `HV({a}) ≥ HV({b})`; thêm một điểm không bao giờ làm HV giảm.
- **Nhạy với chất lượng lẫn độ phủ:** cho phép so sánh cả *tập* vector, không chỉ một vector.
- **Không cần trọng số** giữa các mục tiêu (nhưng có phụ thuộc vào `ref`, xem 4.5).

### 4.2. Các trường hợp cài đặt (`compute_hypervolume`)

```python
valid = np.all(points > ref, axis=1)     # điểm không vượt hoàn toàn ref đóng góp 0 → bỏ
if n_obj == 1: return max(points) − ref
if n_obj == 2: return _hv_2d(points, ref)
else:          return _hv_wfg(points, ref)
```

**Trường hợp một điểm** (quan trọng nhất trong dự án, xem mục 5.1): HV là thể tích một hộp

```
HV({q}) = Π_i (qᵢ − refᵢ)          nếu qᵢ > refᵢ ∀i,  ngược lại 0
```

**Hai mục tiêu — quét (sweep)** `_hv_2d`: sắp theo `x` giảm dần, duy trì `prev_y` là chiều cao đã được phủ; mỗi điểm có `y > prev_y` cộng thêm dải `(x − ref_x) × (y − prev_y)` — khớp code `_hv_2d`.

**Ba mục tiêu trở lên — cắt lát:** đệ quy theo từng chiều qua `_hv_wfg`.

### 4.3. ⚠ Lỗi trong hàm `_hv_wfg` — đã xác nhận đúng bằng code thật

Tên hàm gợi tới thuật toán WFG nhưng thực tế cài đặt kiểu cắt lát, và **độ rộng lát bị tính lệch chỉ số**. Code thật (`utils/pareto.py`):

```python
def _hv_wfg(points: np.ndarray, ref: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    if points.shape[1] == 2:
        return _hv_2d(points, ref)
    nd_idx = non_dominated_indices(points)
    points = points[nd_idx]
    if len(points) == 1:
        return float(np.prod(points[0] - ref))
    order = np.argsort(points[:, 0])[::-1]
    points = points[order]
    hv = 0.0
    for i, p in enumerate(points):
        x_width = p[0] - (points[i - 1][0] if i > 0 else ref[0])
        hv += abs(x_width) * _hv_wfg(points[:i + 1][:, 1:], ref[1:])
    return float(hv)
```

Dùng `p_i.x − p_{i−1}.x` (với `i = 0` dùng `p₀.x − ref`) thay vì `p_i.x − p_{i+1}.x`, và lát cắt dùng `points[:i+1]` (tích lũy từ đầu, tức các điểm có `x` **lớn hơn hoặc bằng** `p_i`) thay vì phần còn lại phía sau. Kết quả là **sai khi có ≥ 2 điểm không bị trội và ≥ 3 mục tiêu**. Ví dụ phản chứng:

```
P = {(2,1,1), (1,2,2)},  ref = (0,0,0)
  Đúng (bao hàm-loại trừ):  2 + 4 − 1 = 5
  Code hiện tại:                                6
```

**[BỔ SUNG]** Có `n_obj == 1` được xử lý riêng ở `compute_hypervolume` (không đi vào `_hv_wfg`), nhưng bản thân `_hv_wfg` **không** có nhánh xử lý `points.shape[1] == 1` — nếu bị gọi đệ quy tới khi còn 1 chiều mục tiêu (trường hợp 4 mục tiêu ban đầu, đệ quy 3 lần), nó rơi vào nhánh chung `order = argsort(...)`, `points[:i+1][:, 1:]` sẽ có shape `(i+1, 0)` ở lần đệ quy cuối — `_hv_wfg` gọi tiếp với `points` rỗng theo trục cột, trả về từ nhánh `len(points) == 0`? Không: `len(points)` đếm theo hàng nên vẫn `> 0`, dẫn tới đệ quy xuống `points.shape[1] == 0`, lúc đó `np.argsort(points[:, 0])` sẽ lỗi vì không có cột 0. Đây là một rủi ro runtime tiềm ẩn (`IndexError`) với đầu vào ≥ 2 điểm không bị trội ở 4 mục tiêu khi đệ quy chạm đáy — **cần kiểm tra thực nghiệm ở checkpoint sửa logic**, chưa chắc có xảy ra vì `non_dominated_indices` có thể rút gọn `points` về 1 hàng trước khi hết chiều.

Ảnh hưởng thực tế tới dự án được giới hạn vì:

- Chọn hành động (HL/LL) chỉ tính HV cho tập **một điểm** (`hv_score_for_action` gọi `compute_hypervolume` trên `nd_pts` sau `non_dominated_indices`, nhưng trong luồng chọn hành động mỗi `q_sets[i]` luôn có đúng 1 phần tử vì mạng chỉ trả 1 Q-vector/ứng viên — xác nhận qua `HLAgent._get_q_sets` và `LLAgent.select_node`) → công thức tích, **đúng**.
- Nhánh sai chỉ chạy khi `prune_by_hypervolume` phải loại bớt vector khỏi tập target có > `max_q_vectors_per_action = 10` phần tử (mục 6): hệ quả là vài vector bị loại "không tối ưu", không làm hỏng chọn hành động trực tiếp — nhưng **có ảnh hưởng gián tiếp tới target Bellman** vì `build_target_q_set` gọi `prune_by_hypervolume` khi `len(targets) > max_size`.

Phiên bản đã sửa (giữ nguyên đề xuất từ tài liệu gốc, **chưa áp dụng vào code** — việc sửa logic để ở checkpoint sau):

```python
def _hv_wfg(points: np.ndarray, ref: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    if points.shape[1] == 1:
        return float(points[:, 0].max() - ref[0])
    if points.shape[1] == 2:
        return _hv_2d(points, ref)
    points = points[non_dominated_indices(points)]
    if len(points) == 1:
        return float(np.prod(points[0] - ref))
    points = points[np.argsort(points[:, 0])[::-1]]      # x giảm dần
    n, hv = len(points), 0.0
    for i in range(n):
        x_next = points[i + 1][0] if i + 1 < n else ref[0]
        width = points[i][0] - x_next
        if width > 0:
            hv += width * _hv_wfg(points[:i + 1, 1:], ref[1:])
    return float(hv)
```

### 4.4. Hypervolume contribution và cắt tỉa

**Đóng góp** của điểm `pᵢ` trong tập `P` là phần thể tích mất đi nếu bỏ nó:

```
contrib(pᵢ) = HV(P) − HV(P \ {pᵢ})
```

`prune_by_hypervolume(vectors, max_size, ref)` — khớp code `utils/pareto.py`:

```
vectors ← non_dominated(vectors)
while |vectors| > max_size:
    xóa vector có contribution nhỏ nhất
```

Hàm ném `ValueError` nếu `max_size ≤ 0` và trả `[]` cho tập rỗng — xác nhận đúng code.

### 4.5. Điểm tham chiếu (reference point)

`ref` là "điểm tệ nhất chấp nhận được", đặt bởi `ParetoConfig.hv_ref_*` = `(−1, −1, −1, −1)` (`hv_reference_point()`, xác nhận trong `config.py`). Bất kỳ vector nào có **một thành phần ≤ ref** có HV = 0.

---

## 5. Chọn hành động bằng Hypervolume

### 5.1. Thuật toán (`select_by_hypervolume`, `utils/pareto.py`) — xác nhận khớp code

```
Đầu vào: q_sets[i] — Q-set của hành động i;  valid_mask;  ref

1. valid = { i | valid_mask[i] };  nếu rỗng → ValueError; nếu 1 phần tử → trả về luôn.
2. hv[i] = HV( non_dominated(q_sets[i]), ref )   ∀ i ∈ valid
3. tied  = { i | hv[i] ≥ max(hv) − 1e-12 }        # dung sai số học
4. nếu |tied| = 1 → trả về.
5. Tie-break 1: với mỗi i ∈ tied, tính trung bình thành phần 0 (r_cost) của q_sets[i];
                chọn nhóm có giá trị lớn nhất (r_cost lớn hơn = chi phí thấp hơn),
                sau đó thu hẹp `tied` xuống `cost_winners` — các phần tử có mean_cost_obj
                trong sai số 1e-12 so với giá trị lớn nhất đó.
6. nếu |cost_winners| = 1 → trả về.
7. Tie-break 2: chọn ngẫu nhiên bằng RNG có seed (HL: `cfg.train.seed`, LL: `cfg.train.seed + 1`)
                để tái lập được.
```

Trong dự án mỗi hành động chỉ có **một** Q-vector (một lần truyền xuôi), nên:

```
HV(q_sets[i]) = Π_k ( q_i,k − ref_k )         (nếu mọi q_i,k > ref_k, ngược lại 0)
```

Hệ quả cần hiểu rõ:

- **Chọn theo HV ≡ argmax của tích các "biên độ" so với ref.** Tương đương cực đại hóa `Σ_k log(q_k − ref_k)` — ưu tiên vector **cân bằng** giữa các mục tiêu hơn vector cực đoan ở một chiều và sát ref ở chiều khác.
- Không cần trọng số `w` để so sánh hành động.

### 5.2. Code trong hai agent — xác nhận khớp

```python
q_sets = [prune(non_dominated([q_np[j]]), max_q_vecs, ref) for j in valid]   # mỗi tập 1 phần tử
best   = select_by_hypervolume(q_sets, valid_mask, ref, tie_break_rng=self._tie_rng)
```

### 5.3. ⚠ Nhạy cảm với thang đo của reference point

Vì `ref = −1` cố định và HV = 0 khi bất kỳ thành phần nào ≤ ref, **thang đo phần thưởng phải nằm trong tầm** của `ref`. Với cấu hình mặc định (xác nhận qua `config.py`):

- `r_cost = −0.001 × deploy_cost` (`cfg.reward.mu_deploy_cost = 0.001`): mỗi VNF (CPU 100–500) tốn cỡ vài trăm đơn vị cho CPU + RAM (trọng số `w_cpu=w_ram=w_storage=w_bw=w_init=1.0` mặc định), cả SFC cỡ hàng nghìn → `r_cost` mỗi SFC cỡ **−1 đến −5**.
- `r_delay = −(α·path_cost + β·path_delay + γ·load_std)` với `α=1.0, β=0.5, γ_load=0.5` và trễ link 2–5 ms → mỗi hop đã đóng góp đáng kể.
- Q là tổng chiết khấu của nhiều bước (`γ = 0.95`) nên còn âm hơn phần thưởng một bước.

Do đó **rất có thể Q_cost và Q_delay nằm dưới −1** ở nhiều (thậm chí mọi) ứng viên → `HV = 0` cho tất cả → tie-break 1 (chọn theo `Q_cost` lớn nhất) quyết định → chính sách có thể suy biến thành "tham lam theo chi phí" mà không còn cân nhắc đa mục tiêu.

Cách kiểm tra: chạy với `verbose=True` (tham số của `run_episode`, `select_sfc`, `select_node`) và quan sát `hv_scores` in ra — nếu đồng loạt `0.0000` là dấu hiệu suy biến. **[BỔ SUNG]** `train.py::main()` không truyền `--verbose` xuống `run_episode` theo per-step mà chỉ bật `verbose_this = args.verbose and (global_ep <= 2)` — tức chỉ 2 episode đầu tiên in chi tiết dù `--verbose` được bật cho cả quá trình training.

---

## 6. Target Bellman dạng tập (`build_target_q_set`, `utils/pareto.py`) — xác nhận khớp code

Trong DQN vô hướng: `target = r + γ · max_{a'} Q_target(s', a')`. Bản đa mục tiêu thay `max` bằng **tập Pareto**:

```
1. Với mỗi hành động kế tiếp a', lấy Q-vector từ target_net:   Q_target(s', a')  ∈ ℝ⁴
2. ND      = non_dominated( { Q_target(s', a') } )
3. targets = non_dominated( { r + γ·q  |  q ∈ ND } )
4. nếu |targets| > max_q_vectors_per_action (10):  targets = prune_by_hypervolume(targets)
```

Các trường hợp đầu cuối trả `[r]`: `done = True`, hoặc `next_q_sets` rỗng/toàn tập rỗng.

**Từ tập về hồi quy.** Mạng chỉ xuất **một** vector nên không thể khớp cả tập; cả hai agent nén tập target thành **trung bình** — xác nhận đúng trong `HLAgent.train_step` (`rep = pts.mean(axis=0)`) và `LLAgent.train_step` (`rep = np.mean(target_sets, axis=0)`):

```python
rep = np.mean(target_sets, axis=0)              # [4]
loss = MSE(q_net(state, action), rep)
```

---

## 7. Hai tầng "Pareto" trong dự án

| | Tầng A: Q-vector 4 chiều + HV | Tầng B: Scalarizer / Archive 2 chiều |
|---|---|---|
| Mục tiêu | `[cost, delay, balance, success]` | `[r_accept, r_cost]` |
| Dùng để | **Chọn hành động** (SFC, node) và tạo target học | **Điều khiển & theo dõi** (weights, utopia/nadir, mặt Pareto theo episode) |
| Cơ chế | Dominance + Hypervolume | Chebyshev tăng cường + `ParetoArchive` |
| Ảnh hưởng tới loss Q | Có | Không (chỉ `r_H` để log và cập nhật trọng số) |
| Code | `utils/pareto.py`: `dominates`, `compute_hypervolume`, `select_by_hypervolume`, `build_target_q_set` | `ParetoScalarizer`, `ParetoArchive` (cùng file `utils/pareto.py`) |

### 7.1. `ParetoScalarizer` — xác nhận khớp `utils/pareto.py`

- **Chebyshev tăng cường**:
  ```
  d_k = w_k · (utopia_k − r_k) / (utopia_k − nadir_k)
  r_H = −( max(d_accept, d_cost) + ρ · (d_accept + d_cost) ),   ρ = cfg.pareto.chebyshev_rho = 0.05
  ```
- `update_utopia`: utopia tăng ngay khi có thưởng tốt hơn, còn lại trượt theo trung bình mũ (`momentum = cfg.pareto.utopia_momentum = 0.98`); nadir chỉ giảm (`min`).
- `sample_weight`: chọn `w_accept ∈ {0, 0.1, …, 1}` (`cfg.pareto.num_weight_bins = 11`), `w_cost = 1 − w_accept`.
- `adapt_weights`: tăng `w_accept` khi tỷ lệ chấp nhận thấp hơn `target_accept=0.9`, tốc độ thích nghi tăng theo `gap = |acc − target|` (`adaptive_rate = rate × (1 + 5×gap)`), giảm nửa tốc độ khi vượt.

**[BỔ SUNG]** Sau `adapt_weights()`, `train.py::run_episode` (cuối hàm, chỉ khi `train=True` và không `fixed_weight`) còn làm mượt thêm một lần: `scalarizer.w_accept = w_accept * 0.9 + scalarizer.w_accept * 0.1` (trộn `w_accept` của episode vừa chạy với `w_accept` mới do `adapt_weights` tính, tỷ lệ 90/10 nghiêng về giá trị episode) — đây là bước làm mượt bổ sung mà tài liệu gốc chưa nêu.

### 7.2. `ParetoArchive` — xác nhận khớp

Lưu các điểm `[avg_r_accept, avg_r_cost]` của từng episode kèm metadata. Logic `add`, crowding-distance-based `_prune_by_crowding`, `best_by_weight`, `summary` khớp code thật trong `utils/pareto.py`.

**[BỔ SUNG]** `pareto_archive.add(...)` chỉ được gọi trong `run_episode` khi `pareto_archive is not None and env.accepted > 0` — nếu một episode/pass không có SFC nào thành công thì episode đó **không** đóng góp điểm vào archive.

### 7.3. Quét mặt Pareto khi đánh giá (`evaluate.py --pareto-scan`)

```
for k in 0..num_points-1:  w_accept = k / (num_points-1), w_cost = 1 - w_accept   # cố định cho cả episode
    chạy toàn bộ file eval với fixed_weight = w (ε = 0, do evaluate.py set hl_agent.epsilon=ll_agent.epsilon=0.0)
    ghi lại (acc_ratio trung bình, deploy_cost trung bình trên các eval_paths)
front = pareto_front([(acc_ratio, −deploy_cost) for r in results])
```

Kết quả lưu vào `--pareto-out` (mặc định `pareto_front.json`) — xác nhận đúng `evaluate.py::scan_pareto_front`.

---

## 8. Kiểm thử (`pareto_test.py`)

Chạy bằng `python pareto_test.py` (không cần GPU). Các nhóm kiểm tra tương ứng các phần đã xác nhận ở trên: `dominates`/`non_dominated`, `compute_hypervolume`/`hypervolume_contribution`, `prune_by_hypervolume`, `select_by_hypervolume` (tie-break theo seed và theo `r_cost`), `build_target_q_set`, `ParetoArchive`, `Config` (ref point, failure vector, `N_OBJ=4` ở cả `hl_scorer.py` và `ll_dqn.py`), replay buffer HL (vector thưởng + mask), `epsilon=0` khi suy luận, `pareto_front` helper.

**Điểm chưa được kiểm thử:** giá trị chính xác của HV khi ≥ 3 mục tiêu và ≥ 2 điểm không bị trội (đây là lý do lỗi `_hv_wfg` ở mục 4.3 không bị phát hiện bởi `pareto_test.py` hiện tại — file test chỉ kiểm HV 4D "dương", không đối chiếu giá trị đúng).

---

## 9. Tham số liên quan (đã xác nhận qua `config.py`)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `qnet.gamma` | 0.95 | Chiết khấu trong Bellman |
| `qnet.lr` | 5e-4 | Learning rate Adam (HL & LL) |
| `qnet.batch_size` | 64 | Kích thước batch |
| `qnet.target_update_freq` | 50 | Số bước train giữa hai lần sao chép target |
| `qnet.max_q_vectors_per_action` | 10 | Cỡ tối đa của tập target sau khi cắt tỉa HV |
| `qnet.hl_buffer_size` / `ll_buffer_size` | 15 000 / 72 000 | Dung lượng replay |
| `pareto.hv_ref_{cost,delay,balance,success}` | −1.0 | Điểm tham chiếu HV |
| `pareto.failure_penalty_*` | 10.0 | Thưởng −10 cho từng mục tiêu khi thất bại |
| `pareto.num_weight_bins` | 11 | Số điểm trọng số `w` |
| `pareto.chebyshev_rho` | 0.05 | Hệ số tăng cường Chebyshev |
| `pareto.utopia_momentum` | 0.98 | Trượt utopia |
| `pareto.weight_adapt_rate` | 0.01 | Tốc độ thích nghi `w_accept` |
| `pareto.front_log_every` | 25 | Chu kỳ in tóm tắt archive |

---

## 10. Bảng tra cứu code

| Khái niệm | File | Hàm / lớp |
|---|---|---|
| Trội, lọc không bị trội | `utils/pareto.py` | `dominates`, `non_dominated_indices`, `non_dominated` |
| Hypervolume | `utils/pareto.py` | `compute_hypervolume`, `_hv_2d`, `_hv_wfg` |
| Đóng góp HV, cắt tỉa | `utils/pareto.py` | `hypervolume_contribution`, `prune_by_hypervolume` |
| Chọn hành động | `utils/pareto.py` | `hv_score_for_action`, `select_by_hypervolume` |
| Target Bellman dạng tập | `utils/pareto.py` | `build_target_q_set` |
| Scalarizer, archive | `utils/pareto.py` | `ParetoScalarizer`, `ParetoArchive` |
| Q-network 4 head | `models/hl_scorer.py`, `models/ll_dqn.py` | `HLSharedQScorer`, `LLNodeScorer` |
| Học Q (HL / LL) | `agents/hl_agent.py`, `agents/ll_agent.py` | `train_step` |
| Quét mặt Pareto | `evaluate.py` | `scan_pareto_front` |
| Kiểm thử | `pareto_test.py` | — |

---

---

# Học tăng cường phân cấp (HRL) cho đặt SFC

> **Ghi chú xác thực:** Đối chiếu với `env/nfv_env.py`, `train.py`, `agents/hl_agent.py`, `agents/ll_agent.py`, `models/hl_scorer.py`, `models/ll_dqn.py`, `utils/dijkstra.py`, `utils/deploy_cost.py`, `config.py`. Tất cả công thức reward, vòng đời SFC, điều kiện `done`, và bảng transition đều khớp code thực tế; điểm sửa được đánh dấu **[SỬA]**.

## 1. Vì sao cần phân cấp?

Với mỗi yêu cầu SFC, hệ thống phải đưa ra chuỗi quyết định lồng nhau:

1. **Chọn SFC nào** trong hàng đợi để xử lý trước.
2. Với SFC đó, **chọn node** cho từng VNF (k = 1..K).
3. **Chọn đường đi** giữa các node liên tiếp.

Nếu gộp tất cả vào một agent phẳng, không gian hành động là tích Descartes `|queue| × N^K` — không thể học trực tiếp. HRL tách bài toán thành các tầng:

| Tầng | Quyết định | Không gian hành động | Tần suất | Được học? |
|---|---|---|---|---|
| **HL Agent** | Chọn SFC từ hàng đợi | `\|queue\|` | 1 lần / SFC | Có (Q-network) |
| **LL Agent** | Chọn node đặt VNF thứ k | ≤ `N` (chỉ node đủ CPU) | K lần / SFC | Có (Q-network) |
| **Routing** | Đường đi giữa hai node | Đồ thị con còn đủ băng thông | K + 1 lần / SFC | Không (Dijkstra) |

Phần **định tuyến** dùng Dijkstra cổ điển (trọng số tắc nghẽn); DRL dành cho hai quyết định khó: *ưu tiên SFC nào* và *đặt VNF ở đâu*.

```
Hàng đợi SFC (queue)
  → HL Agent chọn SFC ★
    → for k = 1..K: LL Agent chọn node vₖ (mask CPU) → Dijkstra v_{k-1}→vₖ → allocate tạm
    → Dijkstra v_K → dst
  → thành công? commit : rollback + failure penalty
```

---

## 2. Mô hình hóa bài toán

### 2.1. Thời gian và hàng đợi (`env/nfv_env.py`)

Môi trường rời rạc theo timestep `t`. Có hai chế độ:

| Chế độ | Nguồn SFC | Điều kiện kết thúc |
|---|---|---|
| **Dataset** (`reset(G, requests, topology_id)`) | Trace từ file: mỗi request có `arrival_time`, `deadline` | `_req_cursor >= len(_all_requests)` **và** hàng đợi rỗng |
| **Ngẫu nhiên** (`reset()`) | Sinh Barabási-Albert + Poisson arrival | `t ≥ cfg.train.episode_horizon` (500) |

Các quy tắc quan trọng (đã xác nhận trong `env/nfv_env.py`):

- Một SFC **hết hạn** khi `deadline ≤ t` (`SFCRequest.is_expired`). SFC hết hạn còn trong hàng đợi bị đếm là `rejected` (qua `_purge_expired_queue`/`_flush_arrivals`).
- **Deadline đóng hai vai trò**: hạn phải được phục vụ, đồng thời thời điểm tài nguyên đã commit được giải phóng (`_release_expired_active_sfcs` trả CPU/băng thông khi `deadline ≤ t`).
- `acceptance_ratio = accepted / total_attempted`, với `total_attempted = accepted + rejected` (property `acceptance_ratio()`).

**Đồng hồ chỉ tiến khi cần.** Trong `run_episode`, `t` **không** tăng sau mỗi SFC. Time chỉ tiến (`env.step_time()`) khi hàng đợi rỗng, hoặc HL không tìm được SFC hợp lệ nào (`hl_result is None` — mọi SFC còn lại bị `blocked_until` hoặc đã hết hạn).

### 2.2. Đặc trưng SFC — `to_feature_vector` (7 chiều, khớp `utils/graph_generator.py::SFCRequest`)

```python
[source, dest, bandwidth, F_k, deadline - t, urgency, total_cpu_req]
```

với `urgency = 1 / max(1, deadline - t)`. Các chiều **chưa được chuẩn hóa** (`source`/`dest` id 0–49, `total_cpu_req` có thể lên hàng nghìn).

Đặc trưng VNF (`train.py::build_vnf_feature`, 2 chiều): `[vnf.cpu_req / cfg.network.cpu_capacity_mips, 1.0]`.

### 2.3. Vector phần thưởng 4 mục tiêu (xác nhận khớp `train.py`)

| Index | Tên | Giá trị khi SFC thành công | Giá trị khi thất bại |
|---|---|---|---|
| 0 | `r_cost` | `-mu_deploy_cost × deploy_cost` | −10 |
| 1 | `r_delay` | `r̄_quality` | −10 |
| 2 | `r_balance` | `-load_std` | −10 |
| 3 | `r_success` | `+1.0` | −10 |

Thất bại dùng `cfg.pareto.failure_penalty_vector()` = `[-10, -10, -10, -10]`.

**Thành phần chất lượng đường đi:**

```
r_quality_seg = -( α × path_cost + β × path_delay + γ_load × load_std )
```

tính **sau khi đã allocate** đoạn đó — xác nhận đúng thứ tự trong `train.py` (`env.allocate(...)` gọi trước `compute_path_cost_delay(env.G, path)`):

- `path_cost = Σ_{link ∈ path} (1 − bw_free/bw_total)` (`utils/dijkstra.py::compute_path_cost_delay`);
- `path_delay = Σ delay_link`;
- `load_std` (`env.get_load_std()`, có cache `_load_std_cache`) — tính **một lần sau khi commit** rồi dùng chung cho mọi đoạn của SFC (xác nhận: `load_std = env.get_load_std()` gọi 1 lần trước vòng `for rec in ll_records`).

`r̄_quality = (Σ r_quality_seg) / (F_k + 1)`.

**Chi phí triển khai** (`utils/deploy_cost.py::total_deploy_cost`), cộng theo từng allocation `(node, path, cpu_req, bw)` — công thức khớp code, xác nhận `path_len = max(0, len(path) - 1)`.

### 2.4. Vòng đời một SFC: thành công, thất bại, chặn tạm thời

```
embedding_failed khi:
  (a) LL không tìm được node hợp lệ (cpu_mask rỗng) tại VNF nào đó
  (b) Dijkstra không tìm thấy đường đến node LL vừa chọn, hoặc node đó không còn đủ CPU
      (kiểm tra kép: path is None OR env.G.nodes[chosen_node]['cpu_free'] < vnf.cpu_req)
  (c) Dijkstra không tìm được đường tới dest

→ rollback toàn bộ allocation tạm (env._rollback_resources)
→ reward = failure_vec
→ SFC còn hạn ở t+1? có: đưa lại vào queue (nếu chưa có), blocked_until[sfc_id] = t+1
                     không: rejected += 1, total_attempted += 1, blocked_until.pop(sfc_id)
```

`blocked_until[sfc_id] = t + 1` ngăn HL chọn lại chính SFC vừa thất bại trong cùng timestep. Sang timestep sau, tài nguyên có thể đã giải phóng nên SFC được thử lại — đến khi hết hạn.

**[SỬA]** Điều kiện (b) trong code thật kiểm tra **cả hai**: `path is None` (Dijkstra không tìm được đường) **hoặc** node đã chọn không còn đủ CPU tại thời điểm kiểm tra (`env.G.nodes[chosen_node]['cpu_free'] < vnf.cpu_req`) — trường hợp thứ hai có thể xảy ra dù `cpu_mask` đã lọc trước đó, vì `cpu_mask` được tính tại đầu vòng lặp VNF nhưng LL chọn dựa trên toàn bộ node hợp lệ trong mask; thực tế race-condition này khó xảy ra trong vòng lặp đơn luồng vì không có gì thay đổi `cpu_free` giữa lúc tính mask và lúc kiểm tra lại — đây là một điều kiện kiểm tra phòng thủ (defensive check) chứ không phải lỗ hổng logic đang thực sự kích hoạt.

---

## 3. HL Agent — chọn SFC

**File:** `agents/hl_agent.py` (agent), `models/hl_scorer.py` (mạng).

### 3.1. Mạng Q kiểu "scorer"

```python
x = torch.cat([z_global, sfc_features, pareto_w], dim=-1)
h = self.trunk(x)
return torch.cat([head(h) for head in self.q_heads], dim=-1)   # [batch, 4]
```

Toàn bộ `m` SFC hợp lệ được chấm trong **một batch duy nhất**. Tổng tham số: `d_in=73 → 256 → 128` + 4 head `128→1` ≈ 73×256 + 256×128 + 4×128 + biases ≈ **52K** (ước tính, chưa có công cụ đếm tham số chính xác trong phiên làm việc này).

### 3.2. Chọn hành động (`select_sfc`) — khớp `agents/hl_agent.py`

```
valid = [i | SFC i chưa hết hạn  VÀ  blocked_until.get(sfc_id, t) ≤ t]
nếu valid rỗng → trả về None  (→ step_time)

với xác suất ε:      chọn ngẫu nhiên trong valid
ngược lại:
    Q_i = q_net(z_global, sfc_i, w)              # vector 4 chiều cho mỗi SFC
    Q-set_i = prune(non_dominated({Q_i}))         # luôn là tập 1 phần tử
    chọn SFC có Hypervolume(Q-set_i) lớn nhất     # hòa → tie-break
```

### 3.3. Replay buffer và bộ dữ liệu chuyển tiếp — khớp `train.py`

Mỗi lần HL xử lý một SFC (thành công hoặc thất bại) sinh đúng **một** transition:

| Trường | Giá trị trong `train.py` |
|---|---|
| `z_global` | Trạng thái toàn cục lúc chọn |
| `sfc_feat` | Đặc trưng của SFC **đã chọn** |
| `pareto_w` | Vector trọng số của episode |
| `action_idx` | Chỉ số SFC trong hàng đợi (chỉ tham khảo) |
| `reward_vec` | `[r_cost, r_delay, r_balance, 1.0]` nếu thành công; `failure_vec` nếu thất bại |
| `next_z_global` | **Cùng `z_global`** hiện tại (`z_global.cpu()` — không encode lại trước khi lưu) |
| `next_sfc_feats` | Đặc trưng SFC còn hạn (`not q.is_expired(env.t)`) trong hàng đợi sau bước này — **không lọc `blocked_until`** |
| `done` | `1.0` nếu `not env.queue` và `env._req_cursor >= len(env._all_requests)` |
| `valid_action_mask` | Toàn `True` (`np.ones(len(next_sfc_feats), dtype=bool)`) |

Buffer: `deque(maxlen = cfg.qnet.hl_buffer_size = 15 000)`.

**[SỬA]** `done` được tính bằng `env._req_cursor >= len(env._all_requests)` — biểu thức này luôn `True` ở chế độ **ngẫu nhiên** vì `_all_requests = []` được gán trong `reset()` khi không dùng dataset, nên `hl_done` ở chế độ ngẫu nhiên thực chất chỉ phụ thuộc `not env.queue` (hàng đợi rỗng), không phụ thuộc `t ≥ episode_horizon`. Điều kiện dừng episode dựa trên `episode_horizon` chỉ được `env.step_time()` kiểm tra ở tầng vòng lặp ngoài (`done = env.step_time()` trong `run_episode`), không phải ở `hl_done` dùng cho transition.

### 3.4. Cập nhật (`train_step`) — khớp `agents/hl_agent.py`

```
với mỗi mẫu trong batch (64 mẫu):
    Q_hiện_tại = q_net(z_global, sfc_feat, w)
    nếu done hoặc không còn SFC kế tiếp:
        target_set = { r }
    ngược lại:
        cho từng SFC kế tiếp j:  q_j = target_net(z', sfc_j, w')
        ND = non_dominated({q_j}); targets = non_dominated({r + γ·q | q ∈ ND})
        nếu |targets| > 10: prune bằng hypervolume contribution
        target_set = targets
    target = mean(target_set)

loss = MSE(Q_hiện_tại, target)
Adam step, clip_grad_norm(10)
mỗi cfg.qnet.target_update_freq (=50) bước: target_net ← q_net
```

`min_buf = min(batch_size, hl_buffer_size // 10) = min(64, 1500) = 64` — `train_step` chỉ chạy khi buffer có ≥ 64 mẫu.

---

## 4. LL Agent — chọn node đặt VNF

**File:** `agents/ll_agent.py` (agent), `models/ll_dqn.py` (mạng).

### 4.1. Trạng thái và hành động

| Thành phần | Kích thước | Ý nghĩa |
|---|---|---|
| `z_global` | 64 | Trạng thái toàn mạng |
| `z_prev` | 32 | Embedding node hiện tại (`current_source`) |
| `z_cand` | 32 | Embedding node ứng viên |
| `vnf_feat` | 2 | `[cpu_req/capacity, 1.0]` |
| `sfc_feat` | 7 | Đặc trưng SFC đang xử lý |
| `pareto_w` | 2 | Vector trọng số |
| **Tổng** | **139** | `cfg.d_ll_input` |

Hành động = "đặt VNF lên node `i`". Mask hợp lệ (`env.build_ll_mask`): `nd.get('is_function_node', True) and nd['cpu_free'] >= cpu_req` — mask **không kiểm tra đường đi**, có thể chọn phải node không định tuyến được (→ thất bại loại (b)/(c) mục 2.4).

### 4.2. Chọn hành động (`select_node`) — khớp `agents/ll_agent.py`

```
valid_nodes = {i | cpu_mask[i]}       rỗng → None (SFC thất bại)
với xác suất ε: chọn ngẫu nhiên trong valid_nodes
ngược lại:
    Q_i = q_net(z_global, z_prev, z_i, vnf_feat, sfc_feat, w)  ∀ i
    Q-set_i = prune(non_dominated({Q_i})) nếu i hợp lệ, ngược lại ∅
    chọn node có HV(Q-set_i) lớn nhất (mask loại node không hợp lệ)
```

### 4.3. Transition và target — khớp `train.py`

Sau mỗi SFC **thành công**, K transition LL được lưu:

| Trường | Giá trị |
|---|---|
| state | `(z_global, z_prev, z_cand, vnf_feat, sfc_feat, w)` |
| reward | `[ -mu_deploy_cost × cost_per_step, r_quality_k, -load_std, 1.0 nếu k cuối else 0.0 ]` với `cost_per_step = deploy_cost / n_steps`, `n_steps = F_k + 1` |
| next state | `None` nếu k cuối; ngược lại `(z_global, z_prev'=z_nodes[node vừa chọn], toàn bộ z_nodes, vnf_feat kế, sfc_feat, mask kế, w)` |
| done | `1.0` nếu k cuối |

Chi phí triển khai chia đều cho `n_steps = F_k + 1` (kể cả đoạn tới `dest` không có transition riêng), nên tổng thưởng chi phí trên K transition là `F_k/(F_k+1)` tổng chi phí.

Khi SFC **thất bại**: mọi transition đã tạo được lưu với `reward = failure_vec / max(1, J)` (J = số VNF đã đặt tạm trước điểm fail), `done = 1`, `next = None`.

**Nếu thất bại ngay tại VNF đầu tiên** (`ll_records` rỗng khi break tại vòng đầu), **không có transition LL nào được lưu** — chỉ HL nhận `failure_vec`.

`train_step` được LL gọi **sau mỗi transition** được lưu (không đợi hết cả SFC). `min_buf = min(64, ll_buffer_size // 10) = min(64, 7200) = 64`.

---

## 5. Định tuyến — Dijkstra có phạt tắc nghẽn

**File:** `utils/dijkstra.py`, hàm `custom_dijkstra(G, source, target, req_bw, omega_bw, lambda_penalty)`.

Chỉ xét link còn đủ băng thông (`bw_free ≥ req_bw`). Trọng số link — khớp code thật:

```
ρ = 1 − bw_free / bw_total, kẹp vào [0, 1 − 1e-9]
w = delay + λ × ( exp(ρ · ln Ω) − 1 ) = delay + λ × ( Ω^ρ − 1 )
```

với `Ω = cfg.reward.omega_bw = 100`, `λ = cfg.reward.lambda_penalty = 1`.

`source == target` trả về `[source]` ngay (không có cạnh, `path_len=0`).

Dijkstra gọi `K + 1` lần mỗi SFC; mỗi đoạn được `env.allocate()` ngay trước khi tính đoạn kế tiếp, nên các đoạn cùng SFC "nhìn thấy" băng thông các đoạn trước đã chiếm. `PathCache` tồn tại trong `utils/dijkstra.py` nhưng **`train.py` không sử dụng** (xác nhận: không có lời gọi `PathCache` nào trong `train.py`).

---

## 6. Commit / Rollback

`NFVEnvironment` (`env/nfv_env.py`) giữ hai thao tác đối xứng:

| Thao tác | Tác dụng |
|---|---|
| `allocate(node, path, cpu, bw)` | Trừ `cpu_free` node, trừ `bw_free` mọi link trên path; xóa `_load_std_cache` nếu `len(path)>1` |
| `_rollback_resources(allocations)` | Cộng trả đúng giá trị trên; xóa `_load_std_cache` nếu có allocation |
| `commit_sfc(sfc, allocations, deploy_cost)` | Ghi vào `active_embeddings` (kèm `deadline`), `accepted += 1`, `total_attempted += 1`, cộng dồn `total_deploy_cost` |

Cơ chế "allocate tạm rồi rollback nếu thất bại" đảm bảo tính **nguyên tử**. `partial_allocations` gồm cả K allocation VNF và allocation cuối tới `dest`.

---

## 7. Điều khiển đa mục tiêu: trọng số `w` và scalarizer

Xem chi tiết công thức ở mục 7 tài liệu Pareto/HV phía trên (cùng file này). Tóm tắt vai trò: `r_H` (Chebyshev 2 chiều) chỉ dùng để log (`episode_hl_reward`) và cập nhật `utopia`/`nadir`/`w_accept`/`w_cost` — **không** tham gia loss Q-network (Q-network học từ vector 4 chiều qua `hl_reward_vec`/`ll reward_vec`).

`w = (w_accept, w_cost)` được ghép vào đầu vào của cả hai Q-network (2 chiều cuối). Train: mỗi episode lấy ngẫu nhiên 1 trong 11 điểm qua `scalarizer.sample_weight()`. Evaluate mặc định: dùng `scalarizer.w_accept/w_cost` lưu trong checkpoint. Evaluate `--pareto-scan`: quét tuần tự cố định mỗi điểm cho cả episode.

---

## 8. Khám phá và lịch huấn luyện

**Epsilon-greedy** áp dụng độc lập cho HL và LL. Lịch (`train.py::main`) — khớp code:

```
warmup_episodes   = warmup_epochs × số_file × passes_per_file
decay_episodes    = int(0.75 × (tổng_episode − warmup_episodes))
eps_decay_per_ep  = (eps_end / eps_start) ** (1 / max(1, decay_episodes))

warm-up (epoch ≤ warmup_epochs):  ε = 1.0 cố định
sau warm-up:                       ε ← max(eps_end, ε × eps_decay_per_ep)   # sau mỗi episode
```

`train_step()` của cả hai agent **vẫn được gọi** ngay khi buffer đủ mẫu tối thiểu trong lúc warm-up — mạng học từ dữ liệu ngẫu nhiên ở giai đoạn này, không phải warmup chỉ để lấp buffer thụ động.

**[SỬA]** `cfg.qnet.eps_decay = 0.995` (trong `QNetConfig`) **không được `train.py` sử dụng ở đâu cả** — lịch decay thực tế hoàn toàn dựa trên công thức hàm mũ tính từ `eps_start`, `eps_end`, `decay_episodes` ở trên; trường `eps_decay` trong config là tham số chết (dead config).

Cấu trúc vòng lặp — khớp `train.py::main`:

```
for epoch in 1..epochs:
    for file in shuffle(train_files):
        G, reqs, topo_id = parse_episode(file)
        for pass in 1..passes_per_file:
            env.reset(G, reqs, topo_id); env._episode_id = global_ep
            stats = run_episode(...)
            decay ε (nếu hết warm-up)
            log
lưu checkpoint: vgae, hl_q, ll_q, hl_eps, ll_eps, w_accept, w_cost, hv_ref_point, pareto_archive
```

---

## 9. Chạy thử tay một SFC (K = 3)

Giả sử `t = 40`, hàng đợi có 3 SFC hợp lệ, `w = (0.7, 0.3)`.

1. **Encode.** `z_nodes [50×32]`, `z_global [64]` lấy từ cache.
2. **HL.** Chấm 3 SFC bằng 3 vector Q ∈ ℝ⁴; chọn SFC có HV lớn nhất, gọi là **S** (hoặc ngẫu nhiên nếu trúng ε).
3. **VNF₁.** `current_source = S.source`. `cpu_mask` cho biết N node đủ CPU. LL chấm các node hợp lệ → chọn node `v₁`. Dijkstra `S.source → v₁`. `allocate` CPU node `v₁` và băng thông đường đi. Ghi record.
4. **VNF₂, VNF₃.** Lặp tương tự; `current_source` cập nhật thành node vừa chọn.
5. **Đoạn cuối.** Dijkstra `v₃ → S.dest`, `allocate` với `cpu = 0`.
6. **Thành công** → tính `deploy_cost`, `load_std`; `commit_sfc`; dựng vector thưởng HL; lưu 3 transition LL; mỗi transition lưu xong gọi `ll_agent.train_step()`. `graph_dirty = True`.
   **Thất bại** → `rollback` mọi allocation tạm; HL nhận `failure_vec`; transition LL đã có được lưu với `failure_vec/J`; SFC bị chặn tới `t + 1`.
7. **HL transition** được lưu và `hl_agent.train_step()` chạy một lần.
8. **Chuyển timestep** khi hàng đợi rỗng/HL không còn hành động hợp lệ: `step_time()` giải phóng SFC hết hạn, nhận SFC mới, encode lại nếu `graph_dirty`.

---

## 10. Hạn chế đã biết của triển khai hiện tại

1. **`w` chỉ là đầu vào, không đi vào target.** Vector thưởng 4 chiều không phụ thuộc `w`; `w` chủ yếu ảnh hưởng qua `r_H` (chỉ log) và qua phân bố dữ liệu training.
2. **Trạng thái kế tiếp của HL không được encode lại.** `next_z_global` bằng `z_global` hiện tại — target HL xấp xỉ.
3. **`valid_action_mask` của HL luôn toàn `True`**, kể cả SFC đang bị `blocked_until` (không bị lọc khỏi `next_sfc_feats`) — target có thể tính trên ứng viên mà thực tế chưa được chọn.
4. **Thất bại tại VNF đầu tiên không tạo transition LL**.
5. **Thang đo đặc trưng SFC chưa chuẩn hóa** (mục 2.2).
6. **Ngưỡng tham chiếu HV cố định = −1** trong khi thang đo mục tiêu có thể vượt −1 — ảnh hưởng trực tiếp chọn hành động (mục 5.3 phần Pareto/HV phía trên).
7. **`sfc_quota_per_timestep` trong config không được sử dụng** ở bất kỳ đâu trong `train.py`/`env/nfv_env.py`.
8. **[BỔ SUNG] `cfg.qnet.eps_decay` là tham số chết**, không ảnh hưởng lịch epsilon thực tế (mục 8).
9. **[BỔ SUNG] Bug `_hv_wfg`** ở ≥3 mục tiêu, ≥2 điểm không bị trội (chi tiết mục 4.3 phần Pareto/HV) — ảnh hưởng gián tiếp tới việc cắt tỉa tập target khi `|targets| > 10`.

---

## 11. Bảng tra cứu code

| Chức năng | File | Hàm / lớp |
|---|---|---|
| Vòng lặp episode | `train.py` | `run_episode` |
| Lịch ε, epoch, checkpoint | `train.py` | `main` |
| Chọn SFC | `agents/hl_agent.py` | `HLAgent.select_sfc` |
| Cập nhật Q của HL | `agents/hl_agent.py` | `HLAgent.train_step` |
| Chọn node | `agents/ll_agent.py` | `LLAgent.select_node` |
| Cập nhật Q của LL | `agents/ll_agent.py` | `LLAgent.train_step` |
| Mạng HL / LL | `models/hl_scorer.py` / `models/ll_dqn.py` | `HLSharedQScorer` / `LLNodeScorer` |
| Mask node hợp lệ | `env/nfv_env.py` | `build_ll_mask` |
| Allocate / rollback / commit | `env/nfv_env.py` | `allocate`, `_rollback_resources`, `commit_sfc` |
| Đồng hồ và hàng đợi | `env/nfv_env.py` | `step_time`, `_flush_arrivals`, `_purge_expired_queue` |
| Định tuyến | `utils/dijkstra.py` | `custom_dijkstra` |
| Chất lượng đường đi, `load_std` | `utils/dijkstra.py` | `compute_path_cost_delay`, `compute_load_std` |
| Chi phí triển khai | `utils/deploy_cost.py` | `total_deploy_cost` |
| Scalarizer, archive | `utils/pareto.py` | `ParetoScalarizer`, `ParetoArchive` |