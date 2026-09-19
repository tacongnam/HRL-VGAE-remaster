"""
pareto.py — Pareto-based Multi-Objective Mechanism

Hai objectives (cả hai đều MAXIMIZE, sau khi chuẩn hoá dấu):
  obj[0] = r_accept  ≡  revenue + urgency + LL_quality   (maximize)
  obj[1] = r_cost    ≡  -mu * deploy_cost                (maximize, tức minimize cost)

Quy tắc dominance (maximize cả hai):
  A dominates B  ⟺  A[i] >= B[i] ∀i  AND  A[j] > B[j] ∃j

ParetoArchive lưu tập non-dominated solutions qua các episode.

ParetoScalarizer còn được giữ lại với MỤC ĐÍCH PHỤ:
  - Tính scalar r_H để HL replay buffer có thể dùng Bellman update
    (DQN cần scalar target; đây là bước scalarization phụ trợ, KHÔNG phải
    cơ chế lựa chọn chính).
  - HL action selection KHÔNG dùng argmax(scalar) nữa; xem hl_agent.py.
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from config import Config

# ─────────────────────────────────────────────────────────────
# 1. Pareto Dominance Logic
# ─────────────────────────────────────────────────────────────

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Return True nếu vector a dominates vector b.
    Giả định TẤT CẢ objectives đều MAXIMIZE.
    Để minimize obj_i, hãy truyền vào -obj_i trước khi gọi hàm này.

    A dominates B ⟺
      - a[i] >= b[i]  ∀i   (a không kém hơn b trên mọi chiều)
      - a[j] >  b[j]  ∃j   (a tốt hơn b trên ít nhất một chiều)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_rank(candidates: np.ndarray) -> np.ndarray:
    """
    Tính Pareto rank (front index) cho mỗi candidate.
    candidates: shape [N, n_obj], tất cả objectives maximize.
    Trả về array int [N] — rank 0 = non-dominated (Pareto front).
    """
    n = len(candidates)
    ranks = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(candidates[j], candidates[i]):
                ranks[i] += 1
    return ranks


def non_dominated_indices(points: np.ndarray) -> List[int]:
    """
    Trả về danh sách index của các điểm không bị dominated trong `points`.
    points: shape [N, n_obj], tất cả objectives maximize.
    """
    n = len(points)
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if dominates(points[j], points[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


# ─────────────────────────────────────────────────────────────
# 2. Pareto Archive
# ─────────────────────────────────────────────────────────────

class ParetoArchive:
    """
    Non-dominated solution archive.

    Mỗi entry là một dict với ít nhất hai key bắt buộc:
      'obj': np.ndarray  — vector objective (tất cả maximize)
      + metadata tùy ý (episode, w_accept, w_cost, acceptance_ratio, ...)

    Thuật toán thêm solution mới:
      1. Nếu solution mới bị dominated bởi bất kỳ entry nào → bỏ qua.
      2. Xoá các entry bị solution mới dominate.
      3. Thêm solution mới vào archive.
      4. Nếu archive đầy (> max_size), xoá entry có crowding distance nhỏ nhất.
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._entries: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, obj: np.ndarray, **metadata) -> bool:
        """
        Thêm solution mới với objective vector `obj`.
        Trả về True nếu được thêm vào, False nếu bị dominated.
        """
        obj = np.asarray(obj, dtype=np.float64)

        # Bước 1: kiểm tra xem solution mới có bị dominated không
        for entry in self._entries:
            if dominates(entry['obj'], obj):
                return False  # bị dominated → bỏ qua

        # Bước 2: xoá các entry bị solution mới dominate
        self._entries = [e for e in self._entries if not dominates(obj, e['obj'])]

        # Bước 3: thêm vào
        entry = {'obj': obj.copy()}
        entry.update(metadata)
        self._entries.append(entry)

        # Bước 4: nếu đầy, loại bỏ điểm có crowding distance nhỏ nhất
        if len(self._entries) > self.max_size:
            self._prune_by_crowding()

        return True

    def _prune_by_crowding(self):
        """Xoá 1 entry có crowding distance nhỏ nhất (giữ diversity)."""
        if len(self._entries) <= 1:
            return
        objs = np.array([e['obj'] for e in self._entries])
        n, m = objs.shape
        crowd = np.zeros(n)
        for k in range(m):
            col = objs[:, k]
            order = np.argsort(col)
            range_k = col[order[-1]] - col[order[0]]
            if range_k < 1e-12:
                continue
            crowd[order[0]] += np.inf
            crowd[order[-1]] += np.inf
            for r in range(1, n - 1):
                crowd[order[r]] += (col[order[r + 1]] - col[order[r - 1]]) / range_k
        # Xoá entry không phải biên và có crowd nhỏ nhất
        finite_mask = np.isfinite(crowd)
        if finite_mask.any():
            remove_idx = int(np.where(finite_mask, crowd, np.inf).argmin())
        else:
            remove_idx = 0
        self._entries.pop(remove_idx)

    def get_front(self) -> List[Dict[str, Any]]:
        """Trả về toàn bộ non-dominated front."""
        return list(self._entries)

    def get_obj_matrix(self) -> Optional[np.ndarray]:
        """Trả về ma trận [N, n_obj] của tất cả objectives trong archive."""
        if not self._entries:
            return None
        return np.array([e['obj'] for e in self._entries])

    def best_by_weight(self, w: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Lấy solution tốt nhất theo weighted sum của objectives.
        Dùng cho MỤC ĐÍCH TRUY VẤN / HIỂN THỊ, không phải để chọn policy
        trong training loop.
        """
        if not self._entries:
            return None
        w = np.asarray(w, dtype=np.float64)
        scores = [float(np.dot(e['obj'], w)) for e in self._entries]
        return self._entries[int(np.argmax(scores))]

    def clear(self):
        self._entries.clear()

    def summary(self) -> str:
        if not self._entries:
            return "ParetoArchive(empty)"
        objs = self.get_obj_matrix()
        return (f"ParetoArchive(size={len(self._entries)}, "
                f"obj0=[{objs[:,0].min():.3f}, {objs[:,0].max():.3f}], "
                f"obj1=[{objs[:,1].min():.3f}, {objs[:,1].max():.3f}])")


# ─────────────────────────────────────────────────────────────
# 3. Pareto-based Action Selection Utilities
# ─────────────────────────────────────────────────────────────

def select_by_pareto_dominance(q_vectors: np.ndarray,
                                valid_mask: np.ndarray,
                                pareto_w: Optional[np.ndarray] = None,
                                epsilon_decomp: float = 0.0) -> int:
    """
    Chọn action dựa trên Pareto dominance thay vì argmax scalar.

    Thuật toán:
      1. Lọc các candidate hợp lệ (valid_mask).
      2. Tìm tập non-dominated trong {Q(s, a) | a valid}.
      3. Trong tập non-dominated:
         a. Nếu có duy nhất 1 candidate → chọn nó.
         b. Nếu có nhiều (không có candidate nào dominant hoàn toàn):
            - Nếu pareto_w được cung cấp: dùng weighted sum CHỈ TRONG
              tập non-dominated để phá tie (bước phụ hợp lệ vì
              tất cả candidates đã non-dominated với nhau).
            - Nếu không: random uniform trong tập non-dominated.

    Tại sao bước b không phá vỡ tính Pareto-based:
      Trong tập non-dominated, không có solution nào tốt hơn hẳn
      solution khác trên mọi objective → không có dominance relation.
      Dùng weighted sum để phá tie ở bước này tương đương chọn một
      điểm cụ thể trên Pareto front, là cơ chế hợp lệ.

    q_vectors: shape [N, n_obj]
    valid_mask: shape [N], bool
    pareto_w: shape [n_obj], optional
    Returns: int index trong [0, N)
    """
    n = len(q_vectors)
    valid_indices = [i for i in range(n) if valid_mask[i]]
    if not valid_indices:
        raise ValueError("No valid action available")

    if len(valid_indices) == 1:
        return valid_indices[0]

    # Lấy Q vectors của các candidate hợp lệ
    valid_qs = np.array([q_vectors[i] for i in valid_indices])  # [M, n_obj]

    # Tìm non-dominated trong valid candidates
    nd_local = non_dominated_indices(valid_qs)  # indices trong valid_qs
    nd_global = [valid_indices[i] for i in nd_local]  # indices trong [0, N)

    if len(nd_global) == 1:
        return nd_global[0]

    # Nhiều non-dominated candidates: phá tie trong tập này
    nd_qs = np.array([q_vectors[i] for i in nd_global])

    if pareto_w is not None:
        w = np.asarray(pareto_w, dtype=np.float64)
        scores = nd_qs @ w
        best_local = int(np.argmax(scores))
    else:
        best_local = random.randint(0, len(nd_global) - 1)

    return nd_global[best_local]


# ─────────────────────────────────────────────────────────────
# 4. ParetoScalarizer — giữ cho Bellman update (HL DQN auxiliary)
# ─────────────────────────────────────────────────────────────

class ParetoScalarizer:
    """
    MỤC ĐÍCH PHỤ: Tính scalar r_H cho Bellman update trong HL replay buffer.

    Lý do giữ lại scalarization:
      DQN yêu cầu scalar target Q-value để tính Bellman MSE loss.
      Scalarization ở đây KHÔNG phải là cơ chế chọn policy:
        - HL action selection dùng Pareto dominance (xem hl_agent.py).
        - r_H chỉ là "chất lượng tổng hợp" phục vụ gradient signal.
      Augmented Chebyshev tốt hơn linear weighted sum vì nó đảm bảo
      coverage của cả concave Pareto front.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.w_accept = cfg.pareto.accept_weight_init
        self.w_cost = cfg.pareto.cost_weight_init
        self.utopia_accept = 0.0
        self.utopia_cost = 0.0
        self.nadir_accept = -cfg.reward.r_fail
        self.nadir_cost = -cfg.reward.r_fail
        self._rho = cfg.pareto.chebyshev_rho

    def current_weight_vector(self) -> np.ndarray:
        return np.array([self.w_accept, self.w_cost], dtype=np.float32)

    def sample_weight(self) -> Tuple[float, float]:
        bins = self.cfg.pareto.num_weight_bins
        k = random.randint(0, bins - 1)
        w = k / max(1, bins - 1)
        return w, 1.0 - w

    def update_utopia(self, r_accept: float, r_cost: float):
        m = self.cfg.pareto.utopia_momentum
        if r_accept > self.utopia_accept:
            self.utopia_accept = r_accept
        else:
            self.utopia_accept = m * self.utopia_accept + (1 - m) * r_accept
        if r_cost > self.utopia_cost:
            self.utopia_cost = r_cost
        else:
            self.utopia_cost = m * self.utopia_cost + (1 - m) * r_cost
        self.nadir_accept = min(self.nadir_accept, r_accept)
        self.nadir_cost = min(self.nadir_cost, r_cost)

    def scalarize(self, r_accept: float, r_cost: float,
                  w_accept: float, w_cost: float) -> float:
        """
        Augmented Chebyshev scalarization — dùng cho gradient signal DQN.
        KHÔNG dùng để chọn action trong Pareto mechanism.
        """
        range_accept = max(1e-6, self.utopia_accept - self.nadir_accept)
        range_cost = max(1e-6, self.utopia_cost - self.nadir_cost)
        d_accept = w_accept * (self.utopia_accept - r_accept) / range_accept
        d_cost = w_cost * (self.utopia_cost - r_cost) / range_cost
        max_term = max(d_accept, d_cost)
        sum_term = d_accept + d_cost
        return -(max_term + self._rho * sum_term)

    def adapt_weights(self, recent_accept_ratio: float,
                      recent_cost_norm: float, target_accept: float = 0.9):
        rate = self.cfg.pareto.weight_adapt_rate
        gap = abs(recent_accept_ratio - target_accept)
        adaptive_rate = rate * (1.0 + 5.0 * gap)
        if recent_accept_ratio < target_accept:
            self.w_accept = min(0.95, self.w_accept + adaptive_rate)
        else:
            self.w_accept = max(0.05, self.w_accept - adaptive_rate * 0.5)
        self.w_cost = 1.0 - self.w_accept


# ─────────────────────────────────────────────────────────────
# 5. Legacy helper (giữ tương thích với evaluate.py)
# ─────────────────────────────────────────────────────────────

def is_dominated(a: np.ndarray, b: np.ndarray) -> bool:
    """b dominates a."""
    return dominates(np.asarray(b), np.asarray(a))


def pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Trả về tập non-dominated từ danh sách (obj0, obj1), cả hai maximize."""
    if not points:
        return []
    pts = np.array(points, dtype=np.float64)
    nd = non_dominated_indices(pts)
    return [tuple(pts[i]) for i in nd]
