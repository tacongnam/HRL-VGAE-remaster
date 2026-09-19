"""
hl_scorer.py — High-Level Q-Network với Vector Output

THAY ĐỔI CHÍNH so với bản cũ:
  Cũ: output scalar  Q(s, a) ∈ ℝ¹
  Mới: output vector Q(s, a) ∈ ℝ^{N_OBJ_HL}

Mỗi chiều tương ứng với một objective:
  Q_vec[0] = Q-value cho objective "acceptance" (maximize)
  Q_vec[1] = Q-value cho objective "cost"       (maximize, tức min deploy cost)

Action selection trong HL Agent KHÔNG dùng argmax(scalar) nữa.
Xem chi tiết tại hl_agent.py::select_sfc().

Scalarization CHỈ dùng để tính Bellman MSE target (gradient signal phụ).
"""

import torch
import torch.nn as nn
from config import Config

# Số objectives (phải khớp với ll_dqn.N_OBJ)
N_OBJ_HL = 2


class HLSharedQScorer(nn.Module):
    """
    Pareto-conditioned High-Level Q-Network.

    Input:  [z_global || sfc_features || pareto_w]
            shape: [batch, d_hl_input] = [batch, 64+7+2] = [batch, 73]

    Output: Q-vector ∈ ℝ^{N_OBJ_HL}  per candidate SFC
            shape: [batch, N_OBJ_HL]

    Cấu trúc: shared trunk → N_OBJ_HL linear heads (giống LLNodeScorer)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        d_in = cfg.d_hl_input   # 64 + 7 + 2 = 73
        self.trunk = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        # Một head riêng cho mỗi objective
        self.q_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(N_OBJ_HL)
        ])

    def forward(self,
                z_global: torch.Tensor,
                sfc_features: torch.Tensor,
                pareto_w: torch.Tensor
                ) -> torch.Tensor:          # [batch, N_OBJ_HL]
        x = torch.cat([z_global, sfc_features, pareto_w], dim=-1)
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.q_heads], dim=-1)


HLSharedQScorerTarget = HLSharedQScorer
