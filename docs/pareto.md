# DRL đa mục tiêu: Q-learning vector, Pareto và Hypervolume

> **Ghi chú xác thực (checkpoint rà soát):** Toàn bộ nội dung dưới đây đã được đối chiếu với source thật `utils/pareto.py`, `train.py`, `agents/hl_agent.py`, `agents/ll_agent.py`, `models/hl_scorer.py`, `models/ll_dqn.py`, `utils/dijkstra.py`, `utils/deploy_cost.py`, `env/nfv_env.py`, `config.py`.
>
> **[CẬP NHẬT — 2 objectives]** Kiến trúc hiện tại dùng **2 mục tiêu**, không phải 4: `N_OBJ_HL = 2` (`models/hl_scorer.py`), `N_OBJ = 2` (`models/ll_dqn.py`), và `train.py::_build_reward_vec_2obj(r_perf, r_cost)` trả về `np.array([r_perf, r_cost])`. Ở tầng HL hai thành phần là `r_acc_c` (thành phần chấp nhận) và `r_cost_c` (chi phí triển khai); ở tầng LL là `r_quality` và `−μ_deploy_cost × cost_per_step`.
>
> **[CẬP NHẬT — bỏ WFG]** Hàm `_hv_wfg` **không còn tồn tại** trong `utils/pareto.py` (đã xác minh: `grep -rn 'hv_wfg' --include='*.py'` không có kết quả). `compute_hypervolume` hiện chỉ xử lý không gian **2 chiều**: lọc `(pts[:,0] > ref[0]) & (pts[:,1] > ref[1])` rồi gọi `_hv_2d`. `dominates` và `non_dominated_indices` cũng được viết cứng cho 2 chiều (lexsort trên 2 cột). Mọi nội dung về "3 mục tiêu trở lên", nhánh `_hv_wfg`, và lỗi lát cắt của `_hv_wfg` (mục 4.3 phiên bản trước) do đó **không còn giá trị** và đã được loại bỏ.
>
> Các sửa/bổ sung được đánh dấu **[SỬA]** / **[BỔ SUNG]** / **[CẬP NHẬT]** inline.

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

Hai mục tiêu của hệ thống — **thành phần chấp nhận** `r_acc_c` (doanh thu + urgency + chất lượng đường đi) và **chi phí triển khai** `r_cost_c` — **mâu thuẫn nhau**, và thang đo khác nhau. Nếu cộng có trọng số `Σ wᵢ·rᵢ` ngay từ đầu:

- Phải chọn `w` trước, mà không ai biết `w` nào là đúng.
- Weighted sum chỉ tìm được các nghiệm nằm trên phần **lồi** của mặt Pareto.
- Mỗi lần đổi ưu tiên lại phải huấn luyện lại.

**Hướng tiếp cận của dự án:** giữ nguyên **vector** phần thưởng `r ∈ ℝ²` và học **vector** giá trị:

```
Q(s, a) = [ Q_acc, Q_cost ] ∈ ℝ²
```

Ở tầng HL: `Q(s,a) = [Q_acc, Q_cost]` với `r = [r_acc_c, r_cost_c]` (`train.py`:
`r_acc_c = revenue + θ·urgency + λ_L·r̄_quality`, `r_cost_c = −μ_deploy_cost·deploy_cost`).
Ở tầng LL: `r = [r_quality, −μ_deploy_cost·cost_per_step]` với `cost_per_step = deploy_cost / (F_k + 1)`.

Mạng có **một thân chung và 2 head tuyến tính** (mỗi head ước lượng một mục tiêu) — xác nhận đúng `models/hl_scorer.py::HLSharedQScorer` (`N_OBJ_HL = 2`) và `models/ll_dqn.py::LLNodeScorer` (`N_OBJ = 2`):

```python
h = self.trunk(x)                                           # [batch, 128]
return torch.cat([head(h) for head in self.q_heads], -1)    # [batch, 2]
```

Bellman dạng vector (cộng vector, nhân vô hướng `γ`) vẫn hợp lệ theo từng thành phần:

```
Q(s, a) = r + γ · Q(s', a*)         với r, Q ∈ ℝ²
```

Nhưng phép `max_{a'}` **không xác định** với vector — không có thứ tự toàn phần trên ℝ². Hai câu hỏi cần trả lời:

1. Khi *chọn hành động*, so sánh `Q(s, a₁)` với `Q(s, a₂)` thế nào? → **Pareto dominance + Hypervolume** (mục 3–5).
2. Khi *tính target*, "hành động tốt nhất kế tiếp" là gì? → **tập Pareto các Q-vector kế tiếp** (mục 6).

**[CẬP NHẬT]** Ngoài vector Q 2 chiều này, `train.py` còn duy trì một tầng scalarization 2 chiều riêng cho phía "admission" (`r_acc_c`, `r_cost_c` → `r_H` qua `ParetoScalarizer`). Hai tầng này dùng **cùng cặp** `(r_acc_c, r_cost_c)` — khác với phiên bản 4 mục tiêu trước đây, nơi tầng scalarization dùng một cặp khác với vector Q. Chi tiết ở mục 7.

---

## 3. Pareto dominance

### 3.1. Định nghĩa

Vector `a` **dominates** (trội hơn) `b` nếu `a` không tệ hơn `b` ở **mọi** mục tiêu và tốt hơn hẳn ở **ít nhất một** mục tiêu (với quy ước maximize):

```
a ≻ b   ⇔   ∀i: aᵢ ≥ bᵢ   VÀ   ∃j: aⱼ > bⱼ
```

Hai vector mà không cái nào trội hơn cái kia là **không so sánh được** (incomparable): mỗi cái tốt hơn ở một số mục tiêu. **Tập không bị trội** (non-dominated set) của một tập vector là các phần tử không bị phần tử nào khác trong tập trội hơn; khi áp dụng lên toàn bộ không gian nghiệm gọi là **mặt Pareto** (Pareto front).

### 3.2. Cài đặt (`utils/pareto.py`)

**[CẬP NHẬT]** Code thật viết **cứng cho 2 mục tiêu**, không phải bản `O(n²·d)` tổng quát như tài liệu phiên bản trước từng mô tả:

```python
def dominates(a, b):
    return bool((a[0] >= b[0] and a[1] >= b[1]) and (a[0] > b[0] or a[1] > b[1]))

def non_dominated_2d_fast(points):          # O(n log n)
    order = np.lexsort((-points[:, 1], -points[:, 0]))   # x giảm dần, rồi y giảm dần
    pts = points[order]
    keep, max_y = np.zeros(n, dtype=bool), -np.inf
    for i in range(n):
        if pts[i, 1] > max_y:               # y lớn hơn mọi y đã thấy → không bị trội
            keep[i] = True
            max_y = pts[i, 1]
    return pts[keep]

def non_dominated_indices(points):          # trả về chỉ số gốc, đã sort
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    ...                                     # cùng logic quét max_y, giữ order[idx]

def non_dominated(vectors):                 # dùng bởi LLAgent
    return [res[i] for i in range(res.shape[0])]   # bọc non_dominated_2d_fast
```

Điểm cần lưu ý (đã xác minh bằng code):

- Độ phức tạp `O(n log n)` nhờ `lexsort` + quét một lượt, **không** còn `O(n²·d)`; với `n` ≤ vài chục ứng viên là rất rẻ.
- Hai vector **bằng nhau** không trội nhau (cần `a[0] > b[0] or a[1] > b[1]`), nên **cả hai đều được giữ** trong tập không bị trội.
- Cả `dominates` lẫn hai hàm `non_dominated*` chỉ đọc `[:, 0]` và `[:, 1]` — **chỉ đúng với không gian 2 mục tiêu**; nếu sau này nâng `N_OBJ` lên ≥ 3 thì các hàm này phải được viết lại.
- `pareto_front(points)` (2D) vẫn tồn tại và được `evaluate.py::scan_pareto_front` dùng cho mặt Pareto theo episode. Các hàm `coverage_metric`, `spacing_metric`, `spread_metric`, `normalize_objectives` cũng giả định 2 chiều (`np.empty((0, 2))`), tuy **không** nằm trên đường chạy training.
- **[CẬP NHẬT]** Hai tên `pareto_rank` và `select_by_pareto_dominance` mà tài liệu phiên bản trước có nhắc **không tồn tại** trong `utils/pareto.py` hiện tại và cũng không được export trong `utils/__init__.py` — đã bỏ khỏi mô tả.

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

### 4.2. Cài đặt (`compute_hypervolume`) — chỉ 2 chiều

**[CẬP NHẬT]** Code thật **không còn** nhánh phân nhánh theo số chiều mục tiêu. Toàn bộ `utils/pareto.py` hiện chỉ phục vụ không gian **2 chiều**:

```python
def compute_hypervolume(points, reference_point):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    ref = np.asarray(reference_point, dtype=np.float64)
    valid_mask = (pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])   # cần > ref ở CẢ HAI chiều
    pts = pts[valid_mask]
    if pts.shape[0] == 0:
        return 0.0
    return _hv_2d(pts, ref)
```

Điểm có một thành phần `≤ ref` tương ứng bị loại khỏi tập trước khi tính.

**Quét (sweep) 2 chiều** `_hv_2d` — sắp `x` giảm dần, duy trì `prev_y` là chiều cao đã được phủ; mỗi điểm có `y > prev_y` cộng thêm dải `(x − ref_x) × (y − prev_y)`:

```python
def _hv_2d(points, ref):
    order = np.argsort(-points[:, 0])
    pts = points[order]
    hv, prev_y = 0.0, ref[1]
    for i in range(pts.shape[0]):
        y = pts[i, 1]
        if y > prev_y:
            hv += (pts[i, 0] - ref[0]) * (y - prev_y)
            prev_y = y
    return float(hv)
```

**Trường hợp một điểm** (quan trọng nhất trong dự án, xem mục 5.1): HV là thể tích một hộp

```
HV({q}) = (q₀ − ref₀) × (q₁ − ref₁)     nếu q₀ > ref₀ và q₁ > ref₁,  ngược lại 0
```

### 4.3. ~~Lỗi trong hàm `_hv_wfg`~~ — **[CẬP NHẬT] đã loại bỏ cùng với hàm**

**[CẬP NHẬT]** Hàm `_hv_wfg` **không còn tồn tại** trong `utils/pareto.py`. Xác minh: `grep -rn 'hv_wfg' --include='*.py' .` trả về rỗng (exit code 1); các định nghĩa còn lại trong file là `dominates`, `non_dominated_2d_fast`, `non_dominated_indices`, `non_dominated`, `_hv_2d`, `compute_hypervolume`, `hypervolume_contribution`, `prune_by_hypervolume_fast`, `prune_by_hypervolume`, `hv_score_for_action`, `select_by_hypervolume`, `build_target_q_set_fast`, `build_target_q_set`, `ParetoArchive`, `ParetoScalarizer`, `pareto_front`, `normalize_objectives`, `coverage_metric`, `spacing_metric`, `spread_metric`.

Hệ quả:

- **Không còn** đường đệ quy N-chiều nào có thể sinh lỗi "lát cắt lệch chỉ số" hay `IndexError` khi đệ quy chạm đáy — hai cảnh báo ở mục 4.3 phiên bản trước **không còn áp dụng**.
- Bù lại, toàn bộ code Pareto hiện **chỉ đúng cho 2 mục tiêu**. Nếu `N_OBJ_HL`/`N_OBJ` được nâng lên ≥ 3, cả `compute_hypervolume`, `dominates` lẫn `non_dominated_indices` đều phải được thay bằng bản N-chiều, và `ParetoConfig.hv_reference_point()` phải trả về vector cùng số chiều.
- `hypervolume_contribution` và `prune_by_hypervolume_fast` cũng chỉ gọi `_hv_2d` trên 2 cột, nên cũng chỉ đúng trong không gian 2 chiều.
- Trong luồng chọn hành động (HL/LL) mỗi `q_sets[i]` luôn có **đúng 1 phần tử** vì mạng chỉ trả 1 Q-vector/ứng viên (xác nhận `HLAgent.select_sfc` dựng `q_sets` bằng `prune_by_hypervolume_fast(non_dominated_2d_fast(q_np[j:j+1]), …)[0]`, và `LLAgent.select_node` tương tự) → HV rơi vào công thức tích 2 chiều ở mục 4.2, **đúng**.

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

`ref` là "điểm tệ nhất chấp nhận được", đặt bởi `ParetoConfig.hv_reference_point()` = **`[hv_ref_cost, hv_ref_success]` = `[−25.0, −25.0]`** (xác nhận trong `config.py`; cả hai mặc định `-25.0`). Bất kỳ vector nào có **một thành phần ≤ ref** có HV = 0.

**[CẬP NHẬT] Hai điểm tham chiếu, một tĩnh một động:**

| Tầng | Giá trị khởi tạo | Có thích nghi? | Căn cứ code |
|---|---|---|---|
| **HL** | `[-25.0, -25.0]` (`hv_reference_point()`) | **Không** — `self._ref_point` gán một lần trong `HLAgent.__init__` (dòng 63) và không bao giờ cập nhật | `agents/hl_agent.py:63` |
| **LL** | `[-25.0, -25.0]` (khởi tạo) rồi bị **ghi đè ngay** ở lần chọn node đầu tiên | **Có** — `LLAgent._update_ref_point(q_np)` | `agents/ll_agent.py:34, 46-55, 104` |

Cơ chế thích nghi của LL (chỉ LL có):

```
batch_min = q_np.min(axis=0)                                  # min theo từng mục tiêu trên batch ứng viên
obs_min   = min(obs_min_trước đó, momentum·obs_min + (1−momentum)·batch_min)   # lần đầu: lấy batch_min
ref       = obs_min − ll_ref_margin
```

với `ll_ref_momentum = 0.99` (trượt chậm) và `ll_ref_margin = 1.0` (`ParetoConfig`). Vì `ref` của LL phụ thuộc phân phối Q thực tế quan sát được, nó **không** còn là hằng số so sánh được giữa các episode — khác với `ref` của HL.

**[CẬP NHẬT]** `ParetoConfig` vẫn còn khai báo `hv_ref_delay` và `hv_ref_balance` (cùng với `failure_penalty_delay`, `failure_penalty_balance`) nhưng **cả 4 trường này hiện là tham số chết**: `hv_reference_point()` và `failure_penalty_vector()` chỉ đọc `hv_ref_cost`/`hv_ref_success` và `failure_penalty_cost`/`failure_penalty_success` tương ứng, nên hai giá trị kia không bao giờ được dùng (xác nhận `config.py:82-100`).

---

## 5. Chọn hành động bằng Hypervolume

### 5.1. Thuật toán (`select_by_hypervolume`, `utils/pareto.py`) — xác nhận khớp code

```
Đầu vào: q_sets[i] — Q-set của hành động i;  valid_mask;  ref

1. valid = { i | valid_mask[i] };  nếu rỗng → ValueError; nếu 1 phần tử → trả về luôn.
2. hv[i] = HV( non_dominated(q_sets[i]), ref )   ∀ i ∈ valid
3. tied  = { i | hv[i] ≥ max(hv) − 1e-12 }        # dung sai số học
4. nếu |tied| = 1 → trả về.
5. Tie-break 1: với mỗi i ∈ tied, tính trung bình thành phần 0 của q_sets[i]; chọn nhóm có giá trị
                lớn nhất, sau đó thu hẹp `tied` xuống `cost_winners` — các phần tử có giá trị đó
                trong sai số 1e-12. **[CẬP NHẬT]** Thành phần 0 trong vector 2 mục tiêu hiện tại là
                **`r_acc_c`** (HL) / **`r_quality`** (LL) — tức là mục tiêu *maximize*, **không** còn là
                `r_cost` như ở bản 4 mục tiêu trước đây. Vì vậy tie-break 1 bây giờ ưu tiên ứng viên có
                thành phần chấp nhận/chất lượng **cao hơn**, chứ không phải "chi phí thấp hơn"
                (xác nhận `utils/pareto.py::select_by_hypervolume`, dòng `mco = float(np.mean([v[0] for v in q_sets[idx]]))`
                và điều kiện `if mco > best_cost`).
6. nếu |cost_winners| = 1 → trả về.
7. Tie-break 2: chọn ngẫu nhiên bằng RNG có seed (HL: `cfg.train.seed`, LL: `cfg.train.seed + 1`)
                để tái lập được.
```

Trong dự án mỗi hành động chỉ có **một** Q-vector (một lần truyền xuôi), nên:

```
HV(q_sets[i]) = (q_i,0 − ref_0) × (q_i,1 − ref_1)     (nếu cả hai > ref, ngược lại 0)
```

Hệ quả cần hiểu rõ:

- **Chọn theo HV ≡ argmax của tích hai "biên độ" so với ref.** Tương đương cực đại hóa `log(q_0 − ref_0) + log(q_1 − ref_1)` — ưu tiên vector **cân bằng** giữa hai mục tiêu hơn vector cực đoan ở một chiều và sát ref ở chiều kia.
- Không cần trọng số `w` để so sánh hành động.

### 5.2. Code trong hai agent — xác nhận khớp

**[CẬP NHẬT]** Hai agent dùng hai bộ hàm khác nhau (HL dùng biến thể `_fast`, LL dùng bản trả list):

```python
# HLAgent.select_sfc  (agents/hl_agent.py:113-116)
q_sets = [[prune_by_hypervolume_fast(
              non_dominated_2d_fast(q_np[j:j+1]), self._max_q_vecs, self._ref_point)[0]]
          for j in range(m)]
best   = select_by_hypervolume(q_sets, np.ones(m, dtype=bool), self._ref_point,
                               tie_break_rng=self._tie_rng)

# LLAgent.select_node  (agents/ll_agent.py:105-112)
self._update_ref_point(q_np)                                   # chỉ LL thích nghi ref
for local_j, global_i in enumerate(valid_nodes):
    nd_set = non_dominated([q_np[local_j]])
    q_sets[global_i] = prune_by_hypervolume(nd_set, self._max_q_vecs, self._ref_point)
chosen = select_by_hypervolume(q_sets, cpu_mask, self._ref_point, tie_break_rng=self._tie_rng)
```

Nhánh ablation của `select_node` chạy **trước** khối này: với `ablation_mode ∈ {DefaultBaseline, GreedyLowLevel}` và `G is not None`, agent trả về ngay node có `cpu_free` lớn nhất (`max(valid_nodes, key=lambda n: G.nodes[n].get("cpu_free", 0.0))`), bỏ qua hoàn toàn Q-network/HV (`agents/ll_agent.py:75-78`).

### 5.3. ⚠ Nhạy cảm với thang đo của reference point

**[CẬP NHẬT]** `ref` của HL là hằng số `[−25, −25]` (không còn `−1`), và HV = 0 khi bất kỳ thành phần nào `≤ ref`. Với ngưỡng `−25` này, phần thưởng thành phần phải nằm **trên** `−25` mới đóng góp HV. Thang đo thực tế (xác nhận qua `config.py` + `train.py`):

- `r_cost_c = −μ_deploy_cost × deploy_cost` với `μ_deploy_cost = 0.001`: mỗi VNF (CPU 100–500) tốn cỡ vài trăm đơn vị cho CPU + RAM + storage (`w_cpu=w_ram=w_storage=w_bw=w_init=1.0` mặc định, `ram_per_cpu_unit=1.0`, `storage_per_vnf=1.0`), cả SFC cỡ hàng nghìn → `r_cost_c` mỗi SFC cỡ **−1 đến −5**, tức **trên** ngưỡng `−25`.
- `r_acc_c = revenue + θ·urgency + λ_L·r̄_quality`: `revenue = (μ_cpu·total_cpu + μ_bw·bw·(F_k+1)) / F_k` với `μ_cpu = μ_bw = 0.001` cho giá trị dương cỡ vài đơn vị; `θ·urgency` (`θ = 2.0`, `urgency = 1/max(1, deadline − t)`) đóng góp dương; `λ_L·r̄_quality` (`λ_L = 1.0`) là phần **âm** đến từ `−(α·path_cost + β·path_delay + γ_load·load_std)` với `α=1.0, β=0.5, γ_load=0.5` và trễ link 2–5 ms.
- Q là tổng chiết khấu nhiều bước (`γ = 0.95`) nên có thể âm hơn phần thưởng một bước, nhưng với `|Q|` cỡ vài chục thì vẫn xa trên `−25`.

**[CẬP NHẬT]** Rủi ro "mọi HV = 0" do đó **thấp hơn đáng kể** so với bản `ref = −1` trước đây — ngưỡng `−25` cho dư địa rất lớn so với thang đo phần thưởng quan sát được. Tuy vậy, nếu `μ_deploy_cost` được tăng hoặc topology/`bw` lớn hơn nhiều, `r_cost_c` vẫn có thể chạm `−25`; khi đó HV về 0 đồng loạt và **tie-break 1** (theo thành phần 0, tức `r_acc_c`/`r_quality`) sẽ quyết định → chính sách suy biến thành "tham lam theo thành phần chấp nhận". Ở tầng LL, `ref` thích nghi (`mục 4.5`) nên nguy cơ này tự nhiên được giảm.

Cách kiểm tra: chạy với `verbose=True` (tham số của `run_episode`, `select_sfc`, `select_node`) và quan sát `hv_scores` in ra — nếu đồng loạt `0.0000` là dấu hiệu suy biến. **[BỔ SUNG]** `train.py::main()` không truyền `--verbose` xuống `run_episode` theo per-step mà chỉ bật `verbose_this = args.verbose and (global_ep <= 2)` — tức chỉ 2 episode đầu tiên in chi tiết dù `--verbose` được bật cho cả quá trình training.

---

## 6. Target Bellman dạng tập (`build_target_q_set`, `utils/pareto.py`) — xác nhận khớp code

Trong DQN vô hướng: `target = r + γ · max_{a'} Q_target(s', a')`. Bản đa mục tiêu thay `max` bằng **tập Pareto**:

```
1. Với mỗi hành động kế tiếp a', lấy Q-vector từ target_net:   Q_target(s', a')  ∈ ℝ²
2. ND      = non_dominated( { Q_target(s', a') } )
3. targets = non_dominated( { r + γ·q  |  q ∈ ND } )
4. nếu |targets| > max_q_vectors_per_action (10):  targets = prune_by_hypervolume(targets)
```

**[CẬP NHẬT]** Có hai đường cài đặt song song, mỗi agent dùng một đường:

| Hàm | Dùng bởi | Ghi chú |
|---|---|---|
| `build_target_q_set_fast(r_vec, cand_q_matrix, …)` | `HLAgent.train_step` | nhận **ma trận** ứng viên `[k × 2]`, gom `non_dominated_2d_fast` + `prune_by_hypervolume_fast`, trả về `np.ndarray` |
| `build_target_q_set(reward_vec, next_q_sets, …)` | `LLAgent.train_step` | nhận **list các list**, bọc `build_target_q_set_fast` bên trong và trả về `list` |

Cả hai đều trả `[r]` khi `done = True`, `cand_q_matrix.size == 0`, hoặc không có ứng viên kế tiếp hợp lệ (`slices[i] = (cursor, cursor)` nên `st == ed`).

**Từ tập về hồi quy.** Mạng chỉ xuất **một** vector nên không thể khớp cả tập; cả hai agent nén tập target thành **trung bình** — xác nhận đúng trong `HLAgent.train_step` (`target_q_vecs[i] = np.mean(t_set, axis=0)`) và `LLAgent.train_step` (`rep = np.mean(target_sets, axis=0)`):

```python
rep = np.mean(target_sets, axis=0)              # [2]
loss = MSE(q_net(state, action), rep)
```

**[CẬP NHẬT] LL có thêm một bước co giãn mục tiêu trong loss.** `LLAgent.__init__` đặt `self._obj_scale = torch.tensor([1.0, 1000.0])` và `train_step` tính `loss = nn.MSELoss()(current_q * self._obj_scale, target_t * self._obj_scale)` (`agents/ll_agent.py:44, 203`) — tức thành phần 1 (`r_quality`) bị nhân trọng số **1000×** so với thành phần 0 (`r_cost_c`). `HLAgent.train_step` **không** có cơ chế này (`nn.MSELoss()(current_q, target_t)`). Đây là khác biệt thực tế giữa hai tầng mà phiên bản tài liệu trước chưa nêu.

**[BỔ SUNG]** Tại `HLAgent.train_step`, `valid_action_mask` được áp dụng bằng cách **lọc hàng** của ma trận ứng viên trước khi vào `build_target_q_set_fast` (`cand_np = cand_np[mask]`, chỉ khi `len(mask) == cand_np.shape[0]`). Vì `train.py` luôn truyền `np.ones(len(next_feats))`, điều kiện độ dài luôn đúng và mask hiện là no-op — nhưng nếu sau này lọc theo `blocked_until` thì cơ chế này đã sẵn sàng.

---

## 7. Hai tầng "Pareto" trong dự án

| | Tầng A: Q-vector 2 chiều + HV | Tầng B: Scalarizer / Archive 2 chiều |
|---|---|---|
| Mục tiêu | `[r_acc_c, r_cost_c]` | `[r_accept, r_cost]` — **cùng cặp giá trị** `(r_acc_c, r_cost_c)` |
| Dùng để | **Chọn hành động** (SFC, node) và tạo target học | **Điều khiển & theo dõi** (weights, utopia/nadir, mặt Pareto theo episode) |
| Cơ chế | Dominance + Hypervolume | Chebyshev tăng cường + `ParetoArchive` |
| Ảnh hưởng tới loss Q | Có | Không (chỉ `r_H` để log và cập nhật trọng số) |
| Code | `utils/pareto.py`: `dominates`, `compute_hypervolume`, `select_by_hypervolume`, `build_target_q_set` | `ParetoScalarizer`, `ParetoArchive` (cùng file `utils/pareto.py`) |

**[CẬP NHẬT]** Ở bản 4 mục tiêu trước đây, hai tầng dùng **hai cặp giá trị khác nhau**: tầng A dùng `[r_cost, r_delay, r_balance, r_success]` còn tầng B dùng `[r_accept_component, r_cost_component]`. Hiện tại `train.py` chỉ tính một cặp `(r_acc_c, r_cost_c)` và dùng nó cho **cả hai**: `_build_reward_vec_2obj(r_acc_c, r_cost_c)` cho replay buffer, và `scalarizer.scalarize(r_acc_c, r_cost_c, …)` cho `r_H`. Hai tầng vẫn tách biệt về mặt *cơ chế* (HV vs Chebyshev) nhưng không còn khác biệt về *đầu vào số*.

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

**[CẬP NHẬT] Lưu ý về `pareto_test.py`:** file kiểm thử này **không còn tồn tại** trong repo. Xác minh: `find . -name 'pareto_test.py'` không trả về kết quả nào; `overview.md` và `hrl.md` cũng đang nhắc lệnh `python pareto_test.py` trong phần "Cách chạy" — **đây là nội dung lỗi thời cần loại bỏ hoặc thay bằng bộ test thực tế**. Các nhóm kiểm thử mà tài liệu phiên bản trước liệt kê (`dominates`/`non_dominated`, `compute_hypervolume`/`hypervolume_contribution`, `prune_by_hypervolume`, `select_by_hypervolume` (tie-break theo seed và theo thành phần 0), `build_target_q_set`, `ParetoArchive`, `Config` (ref point, failure vector, `N_OBJ=2` ở cả `hl_scorer.py` và `ll_dqn.py`), replay buffer HL (vector thưởng + mask), `epsilon=0` khi suy luận, `pareto_front`) là các mục *nên* được phủ nếu viết lại test — không phải mô tả test đang chạy.

**[CẬP NHẬT]** Không còn "điểm chưa được kiểm thử" liên quan tới `_hv_wfg` (mục 4.3) vì hàm đó đã bị xóa. Với cấu hình 2 mục tiêu, `compute_hypervolume` đi qua `_hv_2d` — đường code duy nhất — nên rủi ro HV sai giá trị do đệ quy N-chiều không còn.

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
| `pareto.hv_ref_cost` / `pareto.hv_ref_success` | −25.0 | Hai thành phần của điểm tham chiếu HV 2 chiều (`hv_reference_point()`) |
| `pareto.hv_ref_delay`, `pareto.hv_ref_balance` | −25.0 | **[CẬP NHẬT] tham số chết** — không được `hv_reference_point()` đọc |
| `pareto.ll_ref_momentum` / `pareto.ll_ref_margin` | 0.99 / 1.0 | Thích nghi `ref` của LL (`LLAgent._update_ref_point`) |
| `pareto.failure_penalty_cost` / `pareto.failure_penalty_success` | 10.0 | Thưởng −10 cho từng thành phần khi thất bại → `failure_penalty_vector()` = `[-10, -10]` |
| `pareto.failure_penalty_delay`, `pareto.failure_penalty_balance` | 10.0 | **[CẬP NHẬT] tham số chết** — không được `failure_penalty_vector()` đọc |
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
| Hypervolume (2 chiều) | `utils/pareto.py` | `compute_hypervolume`, `_hv_2d` |
| Đóng góp HV, cắt tỉa | `utils/pareto.py` | `hypervolume_contribution`, `prune_by_hypervolume`, `prune_by_hypervolume_fast` |
| Chọn hành động | `utils/pareto.py` | `hv_score_for_action`, `select_by_hypervolume` |
| Target Bellman dạng tập | `utils/pareto.py` | `build_target_q_set`, `build_target_q_set_fast` |
| Scalarizer, archive | `utils/pareto.py` | `ParetoScalarizer`, `ParetoArchive` |
| Q-network 2 head | `models/hl_scorer.py`, `models/ll_dqn.py` | `HLSharedQScorer` (`N_OBJ_HL = 2`), `LLNodeScorer` (`N_OBJ = 2`) |
| Học Q (HL / LL) | `agents/hl_agent.py`, `agents/ll_agent.py` | `train_step` |
| Quét mặt Pareto | `evaluate.py` | `scan_pareto_front` |
| Kiểm thử | `pareto_test.py` | **[CẬP NHẬT] file không tồn tại trong repo** |

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
- **[CẬP NHẬT] Tài nguyên giải phóng theo từng operation, không theo deadline.** Xem mục 2.1.1 bên dưới — đây là thay đổi lớn nhất so với bản mô tả trước ("deadline đóng hai vai trò").
- `acceptance_ratio = accepted / total_attempted`, với `total_attempted = accepted + rejected` (property `acceptance_ratio()`).

**Đồng hồ chỉ tiến khi cần.** Trong `run_episode`, `t` **không** tăng sau mỗi SFC. Time chỉ tiến (`env.step_time()`) khi hàng đợi rỗng, hoặc HL không tìm được SFC hợp lệ nào (`hl_result is None` — mọi SFC còn lại bị `blocked_until` hoặc đã hết hạn).

#### 2.1.1. Giải phóng tài nguyên theo từng operation (min-heap `_op_heap`)

**[CẬP NHẬT]** `deadline` **chỉ còn** là hạn phải được phục vụ — nó **không còn** là thời điểm giải phóng tài nguyên. Thay vào đó, `NFVEnvironment` giải phóng tài nguyên theo **từng operation**, dựa trên **thời điểm hoàn thành thực tế** của operation đó, quản lý bằng một **min-heap** `_op_heap`.

**Cấu trúc dữ liệu** (`env/nfv_env.py:38-40`, khởi tạo/reset ở `__init__` và `reset`):

```python
self._op_heap: list = []   # mỗi phần tử: (finish_time, counter, spec)
self._op_counter: int = 0  # bộ đếm đơn điệu, phá hoà khi finish_time bằng nhau
```

`counter` là bắt buộc: `heapq` so sánh tuple theo thứ tự, và nếu hai `finish_time` bằng nhau nó sẽ cố so sánh `spec` (dict/tuple chứa node id) — gây lỗi hoặc thứ tự không xác định.

**Đưa sự kiện vào heap** — `_schedule_operation_releases(allocations, t_start, sfc)` được gọi **một lần duy nhất** từ `commit_sfc(...)` (dòng 352), với `t_start = float(self.t)`. Hàm duyệt tuần tự các allocation, **nhân thời gian t tích lũy dọc theo embedding** (routing rồi mới tới serving của cùng một hop):

```python
t = float(t_start)
for (node, path, cpu_req, bw, ram_req, hdd_req) in allocations:
    # (1) RoutingOperation — chỉ khi path có ≥ 2 node
    if len(path) > 1:
        link_delay  = sum(adj[path[i]][path[i+1]]["delay"] for i in range(len(path)-1))
        finish_route = t + link_delay
        heappush(_op_heap, (finish_route, counter++, ("link", tuple(path), bw)))
        t = finish_route                                  # mốc sau cho operation kế tiếp

    # (2) ServingOperation — chỉ khi cpu_req > 0 (đoạn cuối tới dest có cpu_req = 0)
    if cpu_req > 0.0:
        proc   = nodes[node].get("proc_delay", 0.0)
        vnf_d  = next(vnf_iter).delay_at(node)            # VNF thứ k trên node này
        finish_serve = t + proc + vnf_d
        heappush(_op_heap, (finish_serve, counter++, ("cpu", node, cpu_req, ram_req, hdd_req)))
        t = finish_serve
```

Hai loại `spec` và tài nguyên giải phóng tương ứng:

| Loại | `spec` | Thời điểm hoàn thành | Tài nguyên giải phóng |
|---|---|---|---|
| `"link"` | `("link", tuple(path), bw)` | `t + Σ delay(link)` trên `path` | `bw_free` **cộng lại** `bw` cho **từng link** `(path[i], path[i+1])` |
| `"cpu"` | `("cpu", node, cpu_req, ram_req, hdd_req)` | `t + proc_delay[node] + VNF.delay_at(node)` | `cpu_free` cộng `cpu_req`; `ram_free` cộng `ram_req` và `hdd_free` cộng `hdd_req` **nếu node có các key đó** |

Lưu ý thứ tự nhân `t`: vì routing của hop thứ `k+1` bắt đầu từ `finish` của serving hop thứ `k`, các thời điểm trong heap phản ánh đúng **thứ tự tuần tự dọc theo SFC**, không phải song song.

**Rút sự kiện khỏi heap** — `_release_expired_active_sfcs()`, được gọi **đầu tiên** trong `step_time()` (ngay sau `self.t += 1`):

```python
t_float = float(self.t)
while self._op_heap and self._op_heap[0][0] <= t_float:    # lấy hết mọi op đã xong
    _, _, spec = heapq.heappop(self._op_heap)
    if spec[0] == "link":
        _, path, bw = spec
        for i in range(len(path) - 1):
            adj[path[i]][path[i+1]]["bw_free"] += bw
        changed = True
    elif spec[0] == "cpu":
        _, node, cpu_req, ram_req, hdd_req = spec
        nd = nodes[node]
        nd["cpu_free"] += cpu_req
        if ram_req and "ram_free" in nd: nd["ram_free"] += ram_req
        if hdd_req and "hdd_free" in nd: nd["hdd_free"] += hdd_req
        changed = True
if changed:
    self._load_std_cache = None                          # vô hiệu hóa cache độ lệch tải
```

Vì `t` là số nguyên còn `finish_time` là float, điều kiện `<=` nghĩa là operation được giải phóng ở **timestep nguyên đầu tiên lớn hơn hoặc bằng** thời điểm hoàn thành thực của nó (hiệu ứng `ceil`).

**Vòng đời `active_embeddings`** — sau khi xử lý heap, `_release_expired_active_sfcs` vẫn làm một việc **khác**: lọc `active_embeddings` chỉ giữ các SFC còn `deadline > t`. Nhưng danh sách này **không còn** giữ bản ghi `allocations` dùng để giải phóng tài nguyên (tài nguyên đã do heap lo) — nó chỉ còn là công cụ theo dõi vòng đời/deadline. `commit_sfc` vẫn ghi `{"sfc_id", "deadline", "allocations"}` vào đó, và `step_time()` ở chế độ dataset còn dùng `active_embeddings` để tính `next_event = min(deadline)` khi nhảy thời gian.

**[SỬA] Hệ quả so với mô tả trước:** SFC **không còn** "sống trên mạng tới hết deadline". Một SFC đặt xong thường **giải phóng CPU/bandwidth sớm hơn hẳn deadline** vì `finish_serve` chỉ là `t + Σ(link delay) + Σ(proc_delay + VNF delay)` — cỡ vài chục đơn vị thời gian, trong khi `deadline_min = 20`, `deadline_max = 100`. Trong chế độ ngẫu nhiên, do đồng hồ chỉ tiến khi hàng đợi rỗng, một SFC thường được giải phóng trong vài timestep sau khi commit.

### 2.2. Đặc trưng SFC — `to_feature_vector` (7 chiều, khớp `utils/graph_generator.py::SFCRequest`)

```python
[source, dest, bandwidth, F_k, deadline - t, urgency, total_cpu_req]
```

với `urgency = 1 / max(1, deadline - t)`. Các chiều **chưa được chuẩn hóa** (`source`/`dest` id 0–49, `total_cpu_req` có thể lên hàng nghìn).

Đặc trưng VNF (`train.py::build_vnf_feature`, 2 chiều): `[vnf.cpu_req / cfg.network.cpu_capacity_mips, 1.0]`.

### 2.3. Vector phần thưởng 2 mục tiêu (xác nhận khớp `train.py`)

**[CẬP NHẬT]** Mọi Q-network dự đoán vector **2 chiều**, tất cả theo hướng **maximize**:

| Index | Tên | Giá trị khi SFC thành công | Giá trị khi thất bại |
|---|---|---|---|
| 0 | `r_acc_c` (HL) / `r_quality` (LL) | HL: `revenue + θ·urgency + λ_L·r̄_quality`; LL: `r_quality_k` | −10 |
| 1 | `r_cost_c` | HL: `−μ_deploy_cost × deploy_cost`; LL: `−μ_deploy_cost × cost_per_step` | −10 |

Thất bại dùng `failure_penalty_vector()` = **`[-10, -10]`** (`ParetoConfig.failure_penalty_cost = failure_penalty_success = 10.0`).

`r̄_quality` **không còn** là một mục tiêu riêng trong vector thưởng — nó được **gộp vào** thành phần 0 (`r_acc_c`) qua `λ_L` ở tầng HL, và ở tầng LL nó **chính là** thành phần 0. Tương tự `−load_std` không còn là mục tiêu riêng: `load_std` chỉ còn ảnh hưởng gián tiếp qua `r_quality_seg` (thành phần `γ_load·load_std`) và qua `r_H`. Xây dựng vector qua `train.py::_build_reward_vec_2obj(r_perf, r_cost)` → `np.array([r_perf, r_cost], dtype=np.float32)`.

**Thành phần chất lượng đường đi:**

```
r_quality_seg = -( α × path_cost + β × path_delay + γ_load × load_std )
```

tính **sau khi đã allocate** đoạn đó — xác nhận đúng thứ tự trong `train.py` (`env.allocate(...)` gọi trước `compute_path_cost_delay(env.G, path)`):

- `path_cost = Σ_{link ∈ path} (1 − bw_free/bw_total)` (`utils/dijkstra.py::compute_path_cost_delay`);
- `path_delay = Σ delay_link`;
- `load_std` (`env.get_load_std()`, có cache `_load_std_cache`) — tính **một lần sau khi commit** rồi dùng chung cho mọi đoạn của SFC (xác nhận: `load_std = env.get_load_std()` gọi 1 lần trước vòng `for rec in ll_records`).

**[CẬP NHẬT] `r̄_quality` xuất hiện ở 3 chỗ, với 3 vai trò khác nhau** (`train.py:328-345, 362-364`):

| Biến | Công thức | Vai trò trong vector thưởng |
|---|---|---|
| `rec["r_quality"]` | `r_quality_seg` của từng đoạn | **Chính là thành phần 0** của transition LL (`r_quality`) |
| `r̄_quality` | `(Σ rec["r_quality"] + r_dest) / n_steps`, `n_steps = F_k + 1` | Vào **bên trong** `r_acc_c` qua `λ_L` (tầng HL) — không phải một mục tiêu độc lập |
| `r_dest` | `−(α·dest_cost + β·dest_delay + γ_load·load_std)` | Đoạn cuối tới `dest`: **không** có transition LL riêng, chỉ đóng góp vào `r̄_quality` |

**[CẬP NHẬT]** Cả `r_quality` (LL) và `r_acc_c` (HL) đều **không còn** thành phần `r_success = +1.0` — vector chỉ còn 2 chiều nên không còn slot cho cờ "thành công". Tín hiệu thành công hiện chỉ được mã hóa gián tiếp qua `r_acc_c > 0` (có `revenue`) và qua `done = 1.0` ở transition LL cuối. Đây là thay đổi có hệ quả cho credit assignment: mô hình không còn nhận tín hiệu trực tiếp "+1 khi đặt thành công".

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
return torch.cat([head(h) for head in self.q_heads], dim=-1)   # [batch, 2]
```

**[CẬP NHẬT]** `N_OBJ_HL = 2` nên chỉ có **2 head** `128→1`. Toàn bộ `m` SFC hợp lệ được chấm trong **một batch duy nhất** (`z_global` và `pareto_w` được `expand` thành `m` hàng — xác nhận `agents/hl_agent.py:110-112`). Tổng tham số: `d_in = 73 → 256 → 128` + 2 head `128→1` ≈ 73×256 + 256 + 256×128 + 128 + 2×128 + 2 ≈ **52K** (ước tính, chưa có công cụ đếm tham số chính xác trong phiên làm việc này — giá trị gần như không đổi so với bản 4 head vì phần thân `256→128` chi phối).

### 3.2. Chọn hành động (`select_sfc`) — khớp `agents/hl_agent.py`

```
valid = [i | SFC i chưa hết hạn  VÀ  blocked_until.get(sfc_id, t) ≤ t]
nếu valid rỗng → trả về None  (→ step_time)

# [CẬP NHẬT] nhánh ablation chạy TRƯỚC ε-greedy (agents/hl_agent.py:89-98):
#   DefaultBaseline → min(valid, key=lambda i: queue[i].deadline - t)   # ít thời gian còn lại nhất
#   Random-Fit      → random.choice(valid)
#   GreedyFIFO      → min(valid, key=lambda i: queue[i].arrival_time)   # đến sớm nhất

với xác suất ε:      chọn ngẫu nhiên trong valid
ngược lại:
    Q_i = q_net(z_global, sfc_i, w)              # vector 2 chiều cho mỗi SFC
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
| `reward_vec` | **[CẬP NHẬT]** `[r_acc_c, r_cost_c]` nếu thành công (2 chiều, dựng qua `_build_reward_vec_2obj(r_acc_c, r_cost_c)`); `failure_vec` = `[-10, -10]` nếu thất bại |
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

Hành động = "đặt VNF lên node `i`".

**[CẬP NHẬT] Mask hợp lệ — `env.build_ll_mask(cpu_req, ram_req, hdd_req)` kiểm tra cả CPU, RAM và HDD** (không chỉ CPU như mô tả trước). Code thật (`env/nfv_env.py:281-296`):

```python
def build_ll_mask(self, cpu_req, ram_req=0.0, hdd_req=0.0) -> np.ndarray:
    mask = np.zeros(self.G.number_of_nodes(), dtype=bool)
    for u, nd in self.G.nodes(data=True):
        if not nd.get("is_function_node", True):        # (1) phải là function node
            continue
        if nd["cpu_free"] < cpu_req:                    # (2) đủ CPU
            continue
        if ram_req > 0.0 and nd.get("ram_free", float("inf")) < ram_req:   # (3) đủ RAM
            continue
        if hdd_req > 0.0 and nd.get("hdd_free", float("inf")) < hdd_req:   # (4) đủ HDD
            continue
        mask[u] = True
    return mask
```

Bốn điều kiện lọc, **tất cả đều phải thỏa mãn**:

| # | Điều kiện | Mặc định khi thiếu dữ liệu |
|---|---|---|
| 1 | `nd.get("is_function_node", True)` phải truthy | **`True`** — nếu node không có key này thì vẫn được coi là function node |
| 2 | `nd["cpu_free"] >= cpu_req` | key bắt buộc, dùng `nd["cpu_free"]` trực tiếp (không `.get`) |
| 3 | `ram_req > 0` thì `nd.get("ram_free", inf) >= ram_req` | **`inf`** — nếu node không có `ram_free`, điều kiện **luôn đúng** (coi như vô hạn) |
| 4 | `hdd_req > 0` thì `nd.get("hdd_free", inf) >= hdd_req` | **`inf`** — tương tự |

**[CẬP NHẬT] Điểm quan trọng về tác dụng thực tế của điều kiện RAM/HDD:** hai điều kiện (3) và (4) chỉ *có hiệu lực* khi `ram_req > 0` / `hdd_req > 0` **và** node thực sự có key `ram_free` / `hdd_free`. Trong **chế độ ngẫu nhiên**, `build_substrate_network` (`utils/graph_generator.py:64-88`) **không** tạo `ram_free`/`hdd_free`, còn `VNF` mặc định `ram_req = hdd_req = 0.0` (`generate_sfc_requests` khởi tạo `VNF(vnf_id, cpu, cpu, cpu)` nên `ram_req`/`hdd_req` nhận giá trị CPU — nhưng node không có `ram_free` để so) → hai điều kiện này **không loại node nào**. Chúng chỉ thực sự lọc ở **chế độ dataset**, nơi `data/loader.py:56-62` gán `ram_total/ram_free/hdd_total/hdd_free` cho node server và `VNF` đọc `r_f`/`h_f` thành `ram_req`/`hdd_req` (dòng 92-93).

Mask **không kiểm tra đường đi** tới node, nên vẫn có thể chọn phải node không định tuyến được (→ thất bại loại (b)/(c) mục 2.4). Lưu ý `train.py` gọi mask với đủ 3 tham số: `env.build_ll_mask(vnf.cpu_req, vnf.ram_req, vnf.hdd_req)` (dòng 196 khi chọn, và dòng 374-378 khi dựng `next_state`).

### 4.2. Chọn hành động (`select_node`) — khớp `agents/ll_agent.py`

```
valid_nodes = {i | cpu_mask[i]}       rỗng → None (SFC thất bại)

# [CẬP NHẬT] nhánh ablation chạy TRƯỚC ε-greedy (agents/ll_agent.py:75-78):
#   ablation_mode ∈ {DefaultBaseline, GreedyLowLevel} và G is not None:
#     → max(valid_nodes, key=lambda n: G.nodes[n].get("cpu_free", 0.0))  # node rảnh CPU nhiều nhất

với xác suất ε: chọn ngẫu nhiên trong valid_nodes
                (nếu G và current_source khả dụng và k_hop > 0: chỉ chọn trong các node
                 nằm trong bán kính k_hop = cfg.train.k_hop_explore = 3 hop từ current_source)
ngược lại:
    Q_i = q_net(z_global, z_prev, z_i, vnf_feat, sfc_feat, w)  ∀ i     # vector 2 chiều
    Q-set_i = prune(non_dominated({Q_i})) nếu i hợp lệ, ngược lại ∅
    chọn node có HV(Q-set_i) lớn nhất (mask loại node không hợp lệ)
```

### 4.3. Transition và target — khớp `train.py`

Sau mỗi SFC **thành công**, K transition LL được lưu:

| Trường | Giá trị |
|---|---|
| state | `(z_global, z_prev, z_cand, vnf_feat, sfc_feat, w)` |
| reward | **[CẬP NHẬT]** `[ r_quality_k, −μ_deploy_cost × cost_per_step ]` (2 chiều, dựng qua `_build_reward_vec_2obj(cur_rec["r_quality"], −μ_deploy_cost × cost_per_step)`), với `cost_per_step = deploy_cost / n_steps`, `n_steps = F_k + 1`. **Không còn** cờ `r_success` ở transition cuối — chỉ có `done = 1.0`. |
| next state | `None` nếu k cuối; ngược lại `(z_global, z_prev'=z_nodes[node vừa chọn], toàn bộ z_nodes, vnf_feat kế, sfc_feat, mask kế, w)` |
| done | `1.0` nếu k cuối |

Chi phí triển khai chia đều cho `n_steps = F_k + 1` (kể cả đoạn tới `dest` không có transition riêng), nên tổng thưởng chi phí trên K transition là `F_k/(F_k+1)` tổng chi phí.

**[CẬP NHẬT] Loss của LL có co giãn mục tiêu:** `LLAgent.train_step` tính `loss = nn.MSELoss()(current_q * self._obj_scale, target_t * self._obj_scale)` với `self._obj_scale = torch.tensor([1.0, 1000.0])` (`agents/ll_agent.py:44, 203`) — thành phần 1 (`r_quality`) được nhân **1000×**. `HLAgent.train_step` **không** có bước này (`nn.MSELoss()(current_q, target_t)`). Chi tiết ở mục 6 phần Pareto/HV của file này.

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

**[CẬP NHẬT] `λ` bị vô hiệu hóa ở `DefaultBaseline`.** `train.py` (dòng 232 cho đoạn tới node VNF và dòng 283 cho đoạn cuối tới `dest`) truyền:

```python
0.0 if cfg.train.ablation_mode == "DefaultBaseline" else cfg.reward.lambda_penalty
```

Nên ở `DefaultBaseline`, `w = delay` (Dijkstra thuần delay, không phạt tắc nghẽn) — đúng như `docs/ABLATION.md` mô tả ("Basic Dijkstra ignores link utilization penalties"). Cả 4 mode còn lại (`Random-Fit`, `GreedyFIFO`, `GreedyLowLevel`, `HRL-VGAE`) đều dùng `λ = lambda_penalty = 1`. Lưu ý ràng buộc `bw_free ≥ req_bw` **vẫn được áp dụng** ở mọi mode, kể cả `DefaultBaseline` — chỉ có *hệ số phạt mềm* bị đặt về 0.

`source == target` trả về `[source]` ngay (không có cạnh, `path_len=0`).

Dijkstra gọi `K + 1` lần mỗi SFC; mỗi đoạn được `env.allocate()` ngay trước khi tính đoạn kế tiếp, nên các đoạn cùng SFC "nhìn thấy" băng thông các đoạn trước đã chiếm. `PathCache` tồn tại trong `utils/dijkstra.py` nhưng **`train.py` không sử dụng** (xác nhận: không có lời gọi `PathCache` nào trong `train.py`).

---

## 6. Commit / Rollback

`NFVEnvironment` (`env/nfv_env.py`) giữ hai thao tác đối xứng:

| Thao tác | Tác dụng |
|---|---|
| `allocate(node, path, cpu, bw, ram_req=0.0, hdd_req=0.0)` | Nếu `cpu_req > 0`: trừ `cpu_free`, trừ `ram_free` (nếu key tồn tại), trừ `hdd_free` (nếu key tồn tại). Luôn trừ `bw_free` mỗi link trên path; xóa `_load_std_cache` nếu `len(path) > 1` |
| `_rollback_resources(allocations)` | Cộng trả đúng các giá trị trên (dùng `_unpack_alloc` để đọc `ram_req`/`hdd_req` ở vị trí 4/5 khi allocation có dài hơn 4); xóa `_load_std_cache` |
| `commit_sfc(sfc, allocations, deploy_cost)` | Ghi vào `active_embeddings` (kèm `deadline`), `accepted += 1`, `total_attempted += 1`, cộng dồn `total_deploy_cost`, **và** gọi `_schedule_operation_releases(allocations, float(self.t), sfc)` để đưa sự kiện giải phóng vào `_op_heap` |

**[CẬP NHẬT]** `commit_sfc` **không còn** trực tiếp giữ `allocations` để giải phóng sau — nó chuyển chúng cho min-heap `_op_heap` (mục 2.1.1). `active_embeddings` vẫn lưu `allocations` trong dict nhưng chỉ phục vụ theo dõi vòng đời/deadline, không dùng để trả tài nguyên.

**[CẬP NHẬT]** `allocate` có thêm hai tham số `ram_req` và `hdd_req` (mặc định `0.0`), và `train.py` truyền đủ 5 giá trị. `_rollback_resources` dùng `_unpack_alloc` nên tương thích cả tuple 4-phần-tử (định dạng cũ) và 6-phần-tử.

Cơ chế "allocate tạm rồi rollback nếu thất bại" đảm bảo tính **nguyên tử**. `partial_allocations` gồm cả K allocation VNF và allocation cuối tới `dest`.

---

## 7. Điều khiển đa mục tiêu: trọng số `w` và scalarizer

Xem chi tiết công thức ở mục 7 tài liệu Pareto/HV phía trên (cùng file này). Tóm tắt vai trò: `r_H` (Chebyshev 2 chiều) chỉ dùng để log (`episode_hl_reward`) và cập nhật `utopia`/`nadir`/`w_accept`/`w_cost` — **không** tham gia loss Q-network (Q-network học từ vector 2 chiều qua `hl_reward_vec`/LL `reward_vec`).

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
2. **HL.** Chấm 3 SFC bằng 3 vector Q ∈ ℝ²; chọn SFC có HV lớn nhất, gọi là **S** (hoặc ngẫu nhiên nếu trúng ε).
3. **VNF₁.** `current_source = S.source`. `cpu_mask` cho biết N node đủ CPU. LL chấm các node hợp lệ → chọn node `v₁`. Dijkstra `S.source → v₁`. `allocate` CPU node `v₁` và băng thông đường đi. Ghi record.
4. **VNF₂, VNF₃.** Lặp tương tự; `current_source` cập nhật thành node vừa chọn.
5. **Đoạn cuối.** Dijkstra `v₃ → S.dest`, `allocate` với `cpu = 0`.
6. **Thành công** → tính `deploy_cost`, `load_std`; `commit_sfc` (giải phóng operation được **đưa vào `_op_heap`**, mục 2.1.1); dựng `hl_reward_vec = [r_acc_c, r_cost_c]`; lưu 3 transition LL với `reward_vec = [r_quality_k, −μ·cost_per_step]`; mỗi transition lưu xong gọi `ll_agent.train_step()`. `graph_dirty = True`.
   **Thất bại** → `rollback` mọi allocation tạm (heap **không** bị đụng tới vì `commit_sfc` không chạy); HL nhận `failure_vec = [-10, -10]`; transition LL đã có được lưu với `failure_vec/J`; SFC bị chặn tới `t + 1`.
7. **HL transition** được lưu và `hl_agent.train_step()` chạy một lần.
8. **Chuyển timestep** khi hàng đợi rỗng/HL không còn hành động hợp lệ: `step_time()` gọi `_release_expired_active_sfcs()` (giải phóng tài nguyên của mọi operation đã hoàn thành trước `t`), nhận SFC mới, encode lại nếu `graph_dirty`.

---

## 10. Hạn chế đã biết của triển khai hiện tại

1. **`w` chỉ là đầu vào, không đi vào target.** Vector thưởng 2 chiều không phụ thuộc `w`; `w` chủ yếu ảnh hưởng qua `r_H` (chỉ log) và qua phân bố dữ liệu training.
2. **Trạng thái kế tiếp của HL không được encode lại.** `next_z_global` bằng `z_global` hiện tại — target HL xấp xỉ.
3. **`valid_action_mask` của HL luôn toàn `True`**, kể cả SFC đang bị `blocked_until` (không bị lọc khỏi `next_sfc_feats`) — target có thể tính trên ứng viên mà thực tế chưa được chọn.
4. **Thất bại tại VNF đầu tiên không tạo transition LL**.
5. **Thang đo đặc trưng SFC chưa chuẩn hóa** (mục 2.2).
6. **[CẬP NHẬT] Ngưỡng tham chiếu HV của HL cố định = −25** (không còn −1); với thang đo mục tiêu hiện tại, ngưỡng này cho dư địa lớn nên rủi ro HV = 0 đồng loạt thấp — nhưng vẫn tồn tại nếu `μ_deploy_cost`/băng thông tăng mạnh (mục 5.3 phần Pareto/HV phía trên). Ngược lại, `ref` của **LL** thích nghi theo Q quan sát được nên không còn là hằng số so sánh được giữa các episode.
7. **`sfc_quota_per_timestep` trong config không được sử dụng** ở bất kỳ đâu trong `train.py`/`env/nfv_env.py`.
8. **[BỔ SUNG] `cfg.qnet.eps_decay` là tham số chết**, không ảnh hưởng lịch epsilon thực tế (mục 8).
9. **[BỔ SUNG] Tham số chết trong `ParetoConfig`**: `hv_ref_delay`, `hv_ref_balance`, `failure_penalty_delay`, `failure_penalty_balance` đều không được `hv_reference_point()`/`failure_penalty_vector()` đọc (mục 4.5 phần Pareto/HV) — di sản của bản 4 mục tiêu.
10. **[CẬP NHẬT] ~~Bug `_hv_wfg`~~ đã không còn** — hàm `_hv_wfg` đã bị xóa khỏi `utils/pareto.py`, nên không còn rủi ro cắt tỉa target sai. Thay vào đó, toàn bộ code Pareto giờ **chỉ đúng cho 2 mục tiêu**: nếu nâng `N_OBJ` lên ≥ 3 thì `compute_hypervolume`, `dominates`, `non_dominated_indices` và `hv_reference_point()` đều phải được viết lại.
11. **[BỔ SUNG] Hai tầng LL/HL bất đối xứng về huấn luyện**: (a) LL có `_obj_scale = [1.0, 1000.0]` trong loss còn HL không có; (b) `ref` của LL thích nghi còn của HL cố định; (c) HL dùng biến thể `_fast` (`non_dominated_2d_fast`, `prune_by_hypervolume_fast`, `build_target_q_set_fast`) còn LL dùng bản trả list.
12. **[CẬP NHẬT] Bỏ qua cờ `r_success`**: vector 2 chiều không còn mục tiêu "tỷ lệ đặt thành công" riêng, nên mô hình chỉ nhận tín hiệu thành công gián tiếp qua `r_acc_c > 0` và `done = 1.0`.

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
| Thích nghi `ref` (chỉ LL) | `agents/ll_agent.py` | `LLAgent._update_ref_point` |
| Mạng HL / LL | `models/hl_scorer.py` / `models/ll_dqn.py` | `HLSharedQScorer` (`N_OBJ_HL = 2`) / `LLNodeScorer` (`N_OBJ = 2`) |
| Mask node hợp lệ (CPU/RAM/HDD) | `env/nfv_env.py` | `build_ll_mask` |
| Allocate / rollback / commit | `env/nfv_env.py` | `allocate`, `_rollback_resources`, `commit_sfc` |
| **[CẬP NHẬT]** Lịch giải phóng theo operation | `env/nfv_env.py` | `_schedule_operation_releases`, `_release_expired_active_sfcs`, `_unpack_alloc`, `compute_embedding_delay` |
| Đồng hồ và hàng đợi | `env/nfv_env.py` | `step_time`, `_flush_arrivals`, `_purge_expired_queue`, `_flush_arrivals_initial` |
| Định tuyến | `utils/dijkstra.py` | `custom_dijkstra` |
| Chất lượng đường đi, `load_std` | `utils/dijkstra.py` | `compute_path_cost_delay`, `compute_load_std` |
| Chi phí triển khai | `utils/deploy_cost.py` | `total_deploy_cost` |
| Scalarizer, archive | `utils/pareto.py` | `ParetoScalarizer`, `ParetoArchive` |