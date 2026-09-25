# Hypervolume cho đánh giá Pareto front

## 1. Mục đích

Hệ thống tối ưu đồng thời hai objective mâu thuẫn nhau nên không có nghiệm tốt nhất duy nhất — kết quả là một **Pareto front**. Hypervolume (HV) đo thể tích vùng không gian objective bị Pareto front "chiếm" tính từ một điểm tham chiếu, cho phép so sánh chất lượng của toàn bộ front bằng một số duy nhất. Front càng rộng và càng đẩy ra xa điểm tham chiếu thì HV càng lớn.

## 2. Hai objective

| Objective | Ký hiệu trong code | Hướng |
|---|---|---|
| Tỷ lệ chấp nhận SFC | `acc_ratio` | **maximize** |
| Chi phí triển khai | `deploy_cost` | **minimize** |

## 3. Không gian tính HV

HV yêu cầu tất cả objective đều theo hướng **maximize**. Vì `deploy_cost` cần minimize, nó được đổi dấu thành `-deploy_cost` trước khi tính:

```
HV space: (acc_ratio, -deploy_cost)
```

Đổi dấu giúp điểm "tốt hơn" (chi phí thấp hơn) có giá trị `-deploy_cost` lớn hơn, tức là nằm xa điểm tham chiếu hơn — đúng với định nghĩa HV. Mọi thao tác Pareto dominance và `compute_hypervolume` đều chạy trong không gian này.

## 4. Pareto front

Pareto front được lấy bằng `pareto_front()` từ `utils/pareto.py`, áp dụng trên tập điểm đã chuyển sang HV space. Hàm này dùng `non_dominated_indices()` để lọc các điểm không bị trội bởi bất kỳ điểm nào khác. Kết quả được lưu vào `hv_result.json["pareto_front"]`; `plot_results.py` đọc trực tiếp từ đây, không tự tính lại.

## 5. Reference point

Reference point được tính **động từ toàn bộ kết quả scan** trong HV space:

```
ref[0] = min(acc_ratio)      − eps      # phía dưới điểm acc_ratio thấp nhất
ref[1] = min(-deploy_cost)   − eps      # tương đương: −max(deploy_cost) − eps
```

`eps` (mặc định `0.01`, điều chỉnh qua `--hv-eps`) đảm bảo reference point nằm **ngoài** mọi điểm scan, tức là mọi điểm front đều có HV contribution dương. Nếu `eps` quá nhỏ, điểm biên có thể có HV = 0; nếu quá lớn, HV bị phóng đại nhưng vẫn dùng được để so sánh nội bộ.

Reference point được lưu trong `hv_result.json["reference_point"]` với hai trường:
- `acc_ratio`: giá trị trong HV space (nhỏ hơn `min(acc_ratio)`)
- `neg_deploy_cost`: giá trị trong HV space (= `−max(deploy_cost) − eps`)

Khi hiển thị trên plot, `plot_results.py` chuyển `neg_deploy_cost` về `deploy_cost` dương bằng cách đổi dấu.

## 6. Pipeline thực nghiệm

```
Bước 1: Quét Pareto front
python evaluate.py \
  --checkpoint vgae_hrl_ql_checkpoint.pt \
  --data-dir data/dataset_01/test \
  --pareto-scan \
  --pareto-points 11 \
  --pareto-out pareto_front.json

→ pareto_front.json: list 11 điểm {w_accept, w_cost, acc_ratio, deploy_cost}

Bước 2: Tính Hypervolume (standalone, không cần checkpoint)
python evaluate.py \
  --compute-hv \
  --pareto-out pareto_front.json \
  --hv-out hv_result.json \
  --hv-eps 0.01

→ hv_result.json: hypervolume, reference_point, pareto_front, all_points

Bước 3: Vẽ kết quả
python plot_results.py \
  --pareto-json pareto_front.json \
  --hv-json hv_result.json \
  --out plots

→ plots/pareto_front.png
```

Có thể kết hợp bước 1 và 2 trong một lệnh bằng cách thêm `--compute-hv` vào bước 1.

## 7. CLI của `evaluate.py`

| Argument | Mặc định | Ý nghĩa |
|---|---|---|
| `--pareto-scan` | `False` | Chạy quét Pareto front |
| `--pareto-points` | `11` | Số điểm weight (0.0 đến 1.0) |
| `--pareto-out` | `pareto_front.json` | File lưu kết quả scan |
| `--compute-hv` | `False` | Tính HV từ `--pareto-out` |
| `--hv-out` | `hv_result.json` | File lưu kết quả HV |
| `--hv-eps` | `0.01` | Biên ngoài reference point |
| `--checkpoint` | `vgae_hrl_ql_checkpoint.pt` | Checkpoint (chỉ cần cho scan) |
| `--data-dir` | `data/dataset_01/test` | Thư mục episode test |

`--compute-hv` không cần `--pareto-scan` và không cần checkpoint — có thể chạy độc lập trên file scan đã có.

## 8. Output

**`pareto_front.json`** — list các dict, mỗi dict là kết quả trung bình trên toàn bộ test episodes với một bộ weight cố định:
- `w_accept`, `w_cost`: bộ weight dùng khi evaluate
- `acc_ratio`: acceptance ratio trung bình
- `deploy_cost`: deploy cost trung bình

**`hv_result.json`** — dict với các trường:
- `hypervolume`: HV của Pareto front trong không gian `(acc_ratio, -deploy_cost)`
- `reference_point`: `{acc_ratio, neg_deploy_cost}` — tọa độ trong HV space
- `pareto_front`: list các điểm không bị trội, mỗi điểm có `acc_ratio`, `neg_deploy_cost`, `deploy_cost`
- `all_points`: toàn bộ điểm scan, mỗi điểm có `acc_ratio`, `deploy_cost`, `w_accept`, `w_cost`

**`plots/pareto_front.png`** — scatter toàn bộ điểm scan (màu theo `w_accept`), Pareto front highlight màu đỏ, annotation HV và reference point.

## 9. Đọc kết quả

- **`hypervolume`**: số lớn hơn = Pareto front tốt hơn (chỉ so sánh được khi cùng reference point và cùng định nghĩa objective).
- **`reference_point`**: dùng để kiểm tra tính nhất quán khi so sánh nhiều phương pháp; lưu lại để tái sử dụng.
- **`pareto_front`**: các nghiệm không bị dominate — dùng để phân tích trade-off giữa `acc_ratio` và `deploy_cost`.
- **`all_points`**: toàn bộ điểm scan kèm weight — dùng để vẽ scatter và phân tích ảnh hưởng của weight.

## 10. Lưu ý thực nghiệm

Khi so sánh nhiều phương pháp:
- Phải giữ nguyên định nghĩa `acc_ratio` và `deploy_cost` (không thay đổi cách tính từng metric).
- Phải dùng **cùng reference point** cho tất cả phương pháp; cách an toàn nhất là tính reference point từ tập hợp tất cả kết quả của mọi phương pháp rồi dùng chung.
- Không so sánh HV từ các reference point khác nhau — kết quả sẽ không có ý nghĩa.
- `--hv-eps` phải giống nhau giữa các lần so sánh.
- Số điểm `--pareto-points` và tập test episodes phải giống nhau để kết quả có thể so sánh được.