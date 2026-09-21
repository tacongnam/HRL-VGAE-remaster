# DRL đa mục tiêu: Q-learning vector, Pareto và Hypervolume

Tài liệu này giải thích ba lớp kiến thức xếp chồng lên nhau trong dự án, đi từ nền tảng đến chi tiết cài đặt:

1. **DRL (Deep Q-Learning)** — cách agent học giá trị của hành động bằng mạng nơ-ron.
2. **Tối ưu đa mục tiêu Pareto** — khi phần thưởng là một *vector*, "tốt hơn" nghĩa là gì.
3. **Hypervolume (HV)** — cách biến "độ tốt" của một vector/tập vector thành một số để chọn hành động.

Các mảng này cùng nằm trong `hl_agent.py`, `ll_agent.py`, `hl_scorer.py`, `ll_dqn.py` và `pareto.py`.

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
| **Replay buffer** | Các mẫu liên tiếp tương quan mạnh; học trực tiếp từ chúng làm mạng dao động | `HLReplayBuffer`, `LLReplayBuffer`: `deque(maxlen)`, lấy mẫu `random.sample(batch_size = 64)` |
| **Target network** | Target phụ thuộc chính tham số đang học → "đuổi theo cái đuôi" | `target_net`, sao chép từ `q_net` mỗi `target_update_freq = 50` lần cập nhật |
| **ε-greedy** | Cân bằng khám phá / khai thác | `select_sfc`, `select_node`: xác suất ε chọn ngẫu nhiên trong tập hợp lệ |

Ngoài ra code dùng `clip_grad_norm_(…, 10.0)` để chặn gradient bùng nổ, `Adam(lr = 5e-4)`, `γ = 0.95`.

### 1.3. Mạng Q kiểu "scorer"

DQN chuẩn có một đầu ra cho mỗi hành động. Ở đây tập hành động thay đổi liên tục (số SFC trong hàng đợi, số node còn đủ CPU), nên **hành động được đưa vào đầu vào** của mạng:

```
DQN chuẩn:      Q_θ(s)      → [Q(s,a₁), Q(s,a₂), …, Q(s,a_n)]     (n cố định)
Scorer (dự án): Q_θ(s, a)   → Q(s,a)                              (a là input, gọi n lần)
```

Khi chọn hành động, tất cả `m` ứng viên được chấm trong **một batch** (`z_global`, `pareto_w` được `expand` thành `m` hàng, chỉ đặc trưng ứng viên khác nhau). Khi học, hành động đã thực hiện được lưu dưới dạng *đặc trưng của nó* (`sfc_feat` với HL; `z_cand = z_nodes[node]` với LL) nên `current_q = q_net(state, a_đã_chọn)` là một phép truyền xuôi thông thường — không cần `gather` theo chỉ số.

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

Mạng có **một thân chung và 4 head tuyến tính** (mỗi head ước lượng một mục tiêu):

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

Ví dụ 2 mục tiêu `(r_accept, r_cost)`, cả hai maximize:

```
r_cost
  ▲
  │   ○ C(3,9)                ○ = không bị trội (nằm trên front)
  │       ○ B(7,5)            × = bị trội
  │  ×D(5,4)  ← bị B trội (7≥5, 5≥4)
  │           ○ A(10,1)
  └────────────────────────► r_accept
```

### 3.2. Cài đặt (`pareto.py`)

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
- Hai vector **bằng nhau** không trội nhau (cần `np.any(a > b)`), nên **cả hai đều được giữ** trong tập không bị trội. Nếu một bản sao bị vector thứ ba trội thì bị loại (kiểm tra ở `pareto_test.py`, mục 3).
- Có thêm hàm phụ: `pareto_rank` (đếm số vector trội mình), `pareto_front` (cho danh sách điểm 2D) và `select_by_pareto_dominance` (lựa chọn cũ dựa trên tích vô hướng với `w`, giữ lại để tương thích ngược).

---

## 4. Hypervolume

### 4.1. Ý tưởng

Dominance chỉ cho biết "tốt hơn / không so sánh được". Để **xếp hạng** các hành động ta cần một thang vô hướng nhưng vẫn tôn trọng thứ tự Pareto. **Hypervolume** (HV) là thước đo chuẩn: thể tích vùng của không gian mục tiêu bị các điểm "chiếm" tính từ một **điểm tham chiếu** (reference point) `ref`.

```
HV(P, ref) = thể tích của hợp các hộp  [ref, p]  với p ∈ P
```

Trong 2D:

```
y ▲
  │ ┌──┐
  │ │ ○p₁────┐        HV = diện tích vùng tô (hợp các hình chữ nhật
  │ │  │ ○p₂ ├──┐                 có góc dưới-trái là ref,
  │ │  │  │  │○p₃                 góc trên-phải là p₁, p₂, p₃)
  │ └──┴──┴──┴──┘
  └─ ref ─────────────► x
```

Tính chất khiến HV lý tưởng cho việc này:

- **Đơn điệu Pareto:** nếu `a ≻ b` thì `HV({a}) ≥ HV({b})` (và lớn hơn hẳn khi `a` vượt `ref`); thêm một điểm không bao giờ làm HV giảm.
- **Nhạy với chất lượng lẫn độ phủ:** cho phép so sánh cả *tập* vector, không chỉ một vector.
- **Không cần trọng số** giữa các mục tiêu (nhưng có phụ thuộc vào `ref`, xem 4.4).

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

Ví dụ với `ref = (−1, −1, −1, −1)` và `q = (−0.2, −0.5, −0.1, 0.8)`:

```
(−0.2+1)(−0.5+1)(−0.1+1)(0.8+1) = 0.8 × 0.5 × 0.9 × 1.8 = 0.648
```

**Hai mục tiêu — quét (sweep)** `_hv_2d`: sắp theo `x` giảm dần, duy trì `prev_y` là chiều cao đã được phủ; mỗi điểm có `y > prev_y` cộng thêm dải `(x − ref_x) × (y − prev_y)`:

```
P = {(3,1), (2,2), (1,3)},  ref = (0,0)
  (3,1): 3×(1−0) = 3      prev_y = 1
  (2,2): 2×(2−1) = 2      prev_y = 2
  (1,3): 1×(3−2) = 1      prev_y = 3            → HV = 6
```

**Ba mục tiêu trở lên — cắt lát (slicing / HSO):** đệ quy theo từng chiều. Sắp điểm giảm dần theo mục tiêu đầu `x`; giữa hai giá trị `x` liên tiếp, phần thể tích là (độ rộng lát) × (HV của các điểm đang "bao phủ" lát đó trên các chiều còn lại):

```
HV(P) = Σ_i  (xᵢ − x_{i+1}) × HV_{d−1}( { p₀ … pᵢ }  chiếu bỏ chiều x ),   x_n := ref_x
```

### 4.3. ⚠ Lưu ý về hàm `_hv_wfg` hiện tại

Tên hàm gợi tới thuật toán WFG nhưng thực tế cài đặt kiểu cắt lát, và **độ rộng lát bị tính lệch chỉ số**: code dùng `p_i.x − p_{i−1}.x` (với `i = 0` dùng `p₀.x − ref`) thay vì `p_i.x − p_{i+1}.x`. Kết quả là **sai khi có ≥ 2 điểm không bị trội và ≥ 3 mục tiêu**. Ví dụ phản chứng:

```
P = {(2,1,1), (1,2,2)},  ref = (0,0,0)
  Đúng (bao hàm-loại trừ):  2 + 4 − 1 = 5
  Code hiện tại:                                6
```

Kiểm tra ngẫu nhiên trong 4D với `ref = −1` cho thấy code lệch cả hai phía (có lúc cao hơn, có lúc thấp hơn) so với giá trị đúng. Ảnh hưởng thực tế tới dự án được giới hạn vì:

- Chọn hành động (HL/LL) chỉ tính HV cho tập **một điểm** → công thức tích, **đúng**.
- Nhánh sai chỉ chạy khi `prune_by_hypervolume` phải loại bớt vector khỏi tập target có > `max_q_vectors_per_action = 10` phần tử (xem mục 6): hệ quả là vài vector bị loại "không tối ưu", không làm hỏng chọn hành động.

Phiên bản đã sửa (đã đối chiếu với bao hàm–loại trừ trên 200 tập ngẫu nhiên 4D, sai số ~10⁻¹⁶):

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

Nên bổ sung vào `pareto_test.py` một test so sánh với bao hàm–loại trừ:

```python
import itertools
def hv_bruteforce(P, ref):                      # đúng, O(2ⁿ) — chỉ dùng cho test
    P = np.asarray(P, float); tot = 0.0
    for k in range(1, len(P) + 1):
        for S in itertools.combinations(range(len(P)), k):
            m = P[list(S)].min(axis=0)
            tot += (-1) ** (k + 1) * np.prod(np.maximum(m - ref, 0))
    return tot

rng = np.random.default_rng(0)
ref4 = -np.ones(4)
for _ in range(50):
    P = rng.uniform(-0.95, 0, (rng.integers(1, 6), 4))
    assert abs(compute_hypervolume(P, ref4) - hv_bruteforce(P, ref4)) < 1e-9
```

### 4.4. Hypervolume contribution và cắt tỉa

**Đóng góp** của điểm `pᵢ` trong tập `P` là phần thể tích mất đi nếu bỏ nó:

```
contrib(pᵢ) = HV(P) − HV(P \ {pᵢ})
```

`prune_by_hypervolume(vectors, max_size, ref)`:

```
vectors ← non_dominated(vectors)
while |vectors| > max_size:
    xóa vector có contribution nhỏ nhất          # ít "đóng góp" nhất cho độ phủ
```

Ý nghĩa: giữ lại tập con **đa dạng** nhất theo tiêu chí HV. Hàm ném `ValueError` nếu `max_size ≤ 0` và trả `[]` cho tập rỗng.

### 4.5. Điểm tham chiếu (reference point)

`ref` là "điểm tệ nhất chấp nhận được", đặt bởi `ParetoConfig.hv_ref_*` = `(−1, −1, −1, −1)` (`hv_reference_point()`). Bất kỳ vector nào có **một thành phần ≤ ref** có HV = 0 (bị coi như không đóng góp). Chọn `ref` vì vậy cần đủ thấp so với phạm vi thực tế của Q-vector (xem mục 5.3).

---

## 5. Chọn hành động bằng Hypervolume

### 5.1. Thuật toán (`select_by_hypervolume`)

```
Đầu vào: q_sets[i] — Q-set của hành động i;  valid_mask;  ref

1. valid = { i | valid_mask[i] };  nếu rỗng → ValueError; nếu 1 phần tử → trả về luôn.
2. hv[i] = HV( non_dominated(q_sets[i]), ref )   ∀ i ∈ valid
3. tied  = { i | hv[i] ≥ max(hv) − 1e-12 }        # dung sai số học
4. nếu |tied| = 1 → trả về.
5. Tie-break 1: với mỗi i ∈ tied, tính trung bình thành phần 0 (r_cost) của q_sets[i];
                chọn nhóm có giá trị lớn nhất  (r_cost lớn hơn = chi phí thấp hơn).
6. Tie-break 2: nếu vẫn hòa → chọn ngẫu nhiên bằng RNG có seed (HL: seed, LL: seed+1) để tái lập được.
```

Trong dự án mỗi hành động chỉ có **một** Q-vector (một lần truyền xuôi), nên:

```
HV(q_sets[i]) = Π_k ( q_i,k − ref_k )         (nếu mọi q_i,k > ref_k, ngược lại 0)
```

Hệ quả cần hiểu rõ:

- **Chọn theo HV ≡ argmax của tích các "biên độ" so với ref.** Tương đương cực đại hóa `Σ_k log(q_k − ref_k)` — dạng "công bằng theo tỷ lệ" (giống Nash bargaining): ưu tiên vector **cân bằng** giữa các mục tiêu hơn vector cực đoan ở một chiều và sát ref ở chiều khác.
- Không cần trọng số `w` để so sánh hành động.
- Cấu trúc "Q-set" và các hàm `prune`/`non_dominated` được giữ để mở rộng thành nhiều vector mỗi hành động (đúng tinh thần Pareto Q-learning), nhưng ở trạng thái hiện tại các bước này không đổi kết quả khi chọn.

Ví dụ: 3 ứng viên, `ref = −1`:

| Ứng viên | Q-vector | HV = Π(q − ref) |
|---|---|---|
| A | (−0.2, −0.5, −0.1, 0.8) | 0.8 × 0.5 × 0.9 × 1.8 = **0.648** |
| B | (−0.1, −0.9, −0.1, 0.9) | 0.9 × 0.1 × 0.9 × 1.9 = 0.154 |
| C | (−1.3, −0.2, −0.1, 0.9) | 0 (`q_cost ≤ ref`) |

A thắng dù B tốt hơn ở `r_cost`, vì B gần sát ref ở `r_delay`. Ứng viên C có HV = 0 dù các mục tiêu khác rất tốt.

### 5.2. Code trong hai agent

```python
q_sets = [prune(non_dominated([q_np[j]]), max_q_vecs, ref) for j in valid]   # mỗi tập 1 phần tử
best   = select_by_hypervolume(q_sets, valid_mask, ref, tie_break_rng=self._tie_rng)
```

Với `verbose=True` hai agent in `hv_scores` và kích thước Q-set đã chọn — công cụ dùng để kiểm tra HV có suy biến hay không (mục 5.3).

### 5.3. ⚠ Nhạy cảm với thang đo của reference point

Vì `ref = −1` cố định và HV = 0 khi bất kỳ thành phần nào ≤ ref, **thang đo phần thưởng phải nằm trong tầm** của `ref`. Với cấu hình mặc định, ước lượng bậc độ lớn:

- `r_cost = −0.001 × deploy_cost`: mỗi VNF (CPU 100–500) tốn cỡ vài trăm đơn vị cho CPU + RAM, cả SFC cỡ hàng nghìn → `r_cost` mỗi SFC cỡ **−1 đến −5**.
- `r_delay = −(α·path_cost + β·path_delay + γ·load_std)` với trễ link 2–5 ms và `β = 0.5` → mỗi hop đã đóng góp ~ −1 đến −2.5.
- Q là tổng chiết khấu của nhiều bước (`γ = 0.95`) nên còn âm hơn phần thưởng một bước.

Do đó **rất có thể Q_cost và Q_delay nằm dưới −1** ở nhiều (thậm chí mọi) ứng viên. Khi đó `HV = 0` cho tất cả → mọi hành động hòa → luật tie-break 1 hoạt động: chọn theo `Q_cost` lớn nhất. Nghĩa là chính sách suy biến thành "tham lam theo một mục tiêu chi phí" mà không còn cân nhắc đa mục tiêu.

Cách kiểm tra và khắc phục:

1. Chạy với `--verbose` và quan sát `hv_scores`; nếu chúng đồng loạt `0.0000` là dấu hiệu suy biến.
2. Đặt `hv_ref_*` **thấp hơn** phạm vi thực tế (ví dụ dựa trên phân vị thấp của Q/thưởng đo được, hoặc `−(failure_penalty) × 1/(1−γ)` làm cận rất an toàn), hoặc
3. Chuẩn hóa từng mục tiêu về cùng thang (ví dụ chia cho hệ số ước lượng hoặc `tanh`) trước khi đưa vào Q-vector — khi đó `ref = −1` mới có ý nghĩa nhất quán.

---

## 6. Target Bellman dạng tập (`build_target_q_set`)

Trong DQN vô hướng: `target = r + γ · max_{a'} Q_target(s', a')`. Bản đa mục tiêu thay `max` bằng **tập Pareto**:

```
1. Với mỗi hành động kế tiếp a', lấy Q-vector từ target_net:   Q_target(s', a')  ∈ ℝ⁴
2. ND      = non_dominated( { Q_target(s', a') } )            # các "hành động kế tiếp tốt" (không so sánh được)
3. targets = non_dominated( { r + γ·q  |  q ∈ ND } )
4. nếu |targets| > max_q_vectors_per_action (10):  targets = prune_by_hypervolume(targets)
```

Các trường hợp đầu cuối trả `[r]`: `done = True`, hoặc không còn hành động kế tiếp. Phép biến đổi `q ↦ r + γ·q` (`γ > 0`) bảo toàn quan hệ trội, nên bước lọc ở (3) không làm thay đổi tập ND ở (2) — bước này được giữ như một bước kiểm tra an toàn.

**Từ tập về hồi quy.** Mạng chỉ xuất **một** vector nên không thể khớp cả tập; cả hai agent nén tập target thành **trung bình**:

```python
rep = np.mean(target_sets, axis=0)              # [4]
loss = MSE(q_net(state, action), rep)
```

Mỗi mục tiêu hồi quy độc lập về giá trị trung bình của các phần tử trong front.

Nhận xét khi đối chiếu với lý thuyết:

- **Pareto Q-learning** (Van Moffaert & Nowé, 2014) lưu một *tập* Q cho mỗi cặp (s, a) trong bảng và cập nhật tập bằng hợp không bị trội; dự án là phiên bản hàm xấp xỉ, mỗi (s, a) một điểm.
- **Chọn hành động bằng HV** đi theo ý tưởng của *Hypervolume-based multi-objective RL* (Van Moffaert, Drugan & Nowé, 2013).
- Target là **trung bình trên toàn front kế tiếp**, không phải Q-vector của hành động mà chính sách HV sẽ chọn — nên hành vi (HV-greedy) và bootstrap (trung bình front) không hoàn toàn khớp. Đây là một xấp xỉ có thể ảnh hưởng độ ổn định hội tụ; nếu muốn chặt chẽ hơn, có thể bootstrapping bằng Q-vector của hành động có HV lớn nhất.

---

## 7. Hai tầng "Pareto" trong dự án

Có hai cơ chế mang tên Pareto, cần phân biệt rõ:

| | Tầng A: Q-vector 4 chiều + HV | Tầng B: Scalarizer / Archive 2 chiều |
|---|---|---|
| Mục tiêu | `[cost, delay, balance, success]` | `[r_accept, r_cost]` |
| Dùng để | **Chọn hành động** (SFC, node) và tạo target học | **Điều khiển & theo dõi** (weights, utopia/nadir, mặt Pareto theo episode) |
| Cơ chế | Dominance + Hypervolume | Chebyshev tăng cường + `ParetoArchive` |
| Ảnh hưởng tới loss Q | Có | Không (chỉ `r_H` để log và cập nhật trọng số) |
| Code | `pareto.py`: `dominates`, `compute_hypervolume`, `select_by_hypervolume`, `build_target_q_set` | `ParetoScalarizer`, `ParetoArchive` |

### 7.1. `ParetoScalarizer`

- **Chebyshev tăng cường** biến `(r_accept, r_cost)` thành một số dựa trên khoảng cách có trọng số tới điểm lý tưởng (utopia):
  ```
  d_k = w_k · (utopia_k − r_k) / (utopia_k − nadir_k)
  r_H = −( max(d_accept, d_cost) + ρ · (d_accept + d_cost) ),   ρ = 0.05
  ```
  Khác weighted sum, Chebyshev với `ρ` nhỏ có thể tìm nghiệm ở cả phần **lõm** của mặt Pareto; số hạng `ρ·Σd` loại nghiệm "yếu" (weakly Pareto).
- `update_utopia`: utopia tăng ngay khi có thưởng tốt hơn, còn lại trượt theo trung bình mũ (`momentum = 0.98`); nadir chỉ giảm.
- `sample_weight`: chọn `w_accept ∈ {0, 0.1, …, 1}`, `w_cost = 1 − w_accept`.
- `adapt_weights`: tăng `w_accept` khi tỷ lệ chấp nhận thấp hơn 0.9 (tốc độ thích nghi tăng theo khoảng cách), giảm nửa tốc độ khi vượt.

### 7.2. `ParetoArchive`

Lưu các điểm `[avg_r_accept, avg_r_cost]` của từng episode kèm metadata (`w`, `acceptance_ratio`, `total_deploy_cost`).

```
add(obj):
   nếu bị điểm nào trong archive trội  → từ chối (trả False)
   ngược lại: xóa mọi điểm bị obj trội, thêm obj
   nếu size > max_size (200): xóa điểm có crowding distance nhỏ nhất
```

Crowding distance (kiểu NSGA-II): điểm ở biên có giá trị vô hạn (luôn được giữ), điểm còn lại đo bằng khoảng cách chuẩn hóa tới hai láng giềng — loại điểm ở vùng đông đúc để archive phân bố đều. `best_by_weight(w)` trả điểm có tích vô hướng lớn nhất.

### 7.3. Quét mặt Pareto khi đánh giá (`evaluate.py --pareto-scan`)

```
for k in 0..10:  w = (k/10, 1 − k/10)          # cố định cho cả episode
    chạy toàn bộ file test với fixed_weight = w (ε = 0)
    ghi lại (acc_ratio trung bình, deploy_cost trung bình)
front = pareto_front([(acc_ratio, −deploy_cost)])   # lọc điểm không bị trội
```

Kết quả lưu vào `pareto_front.json`; số điểm không bị trội cho thấy `w` thực sự tạo ra những đánh đổi khác nhau hay chỉ cho cùng một hành vi.

---

## 8. Kiểm thử (`pareto_test.py`)

Chạy bằng `python pareto_test.py` (không cần GPU). 15 nhóm kiểm tra:

| Nhóm | Nội dung |
|---|---|
| 1–3 | `dominates`, `non_dominated_indices`, `non_dominated` (kể cả bản sao) |
| 4–5 | `compute_hypervolume` (1 điểm, dưới ref, 2D, bỏ điểm bị trội, 4D dương), `hypervolume_contribution` |
| 6 | `prune_by_hypervolume`: giới hạn kích thước, kết quả vẫn không bị trội, `max_size=0` ném lỗi, rỗng → `[]` |
| 7 | `select_by_hypervolume`: chọn HV tốt nhất, mask, Q-set rỗng, không hợp lệ ném lỗi, tie-break tái lập theo seed và theo `r_cost` |
| 8 | `build_target_q_set`: dạng, không bị trội, đầu cuối, rỗng |
| 9 | `ParetoArchive`: thêm/từ chối/xóa điểm bị trội, giới hạn kích thước |
| 10–11 | `Config`: ref và failure vector đúng shape/dấu; `N_OBJ = 4` ở cả hai mạng |
| 12–13 | Replay buffer HL lưu vector thưởng và mask; ε = 0 khi suy luận |
| 14–15 | `pareto_front` helper; failure vector dùng nhất quán |

Điểm chưa được kiểm thử: **giá trị chính xác** của HV khi ≥ 3 mục tiêu và nhiều điểm (chỉ kiểm "dương" ở 4D) — đây là lý do lỗi ở mục 4.3 không bị phát hiện; nên bổ sung test đối chiếu bao hàm–loại trừ đã nêu.

---

## 9. Tham số liên quan

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
| Trội, lọc không bị trội | `pareto.py` | `dominates`, `non_dominated_indices`, `non_dominated` |
| Hypervolume | `pareto.py` | `compute_hypervolume`, `_hv_2d`, `_hv_wfg` |
| Đóng góp HV, cắt tỉa | `pareto.py` | `hypervolume_contribution`, `prune_by_hypervolume` |
| Chọn hành động | `pareto.py` | `hv_score_for_action`, `select_by_hypervolume` |
| Target Bellman dạng tập | `pareto.py` | `build_target_q_set` |
| Scalarizer, archive | `pareto.py` | `ParetoScalarizer`, `ParetoArchive` |
| Q-network 4 head | `hl_scorer.py`, `ll_dqn.py` | `HLSharedQScorer`, `LLNodeScorer` |
| Học Q (HL / LL) | `hl_agent.py`, `ll_agent.py` | `train_step` |
| Quét mặt Pareto | `evaluate.py` | `scan_pareto_front` |
| Kiểm thử | `pareto_test.py` | — |