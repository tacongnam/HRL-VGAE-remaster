"""
hl_agent.py — High-Level Agent với Pareto-based Action Selection

THAY ĐỔI so với bản cũ:

1. select_sfc():
   CŨ:  scores = Q_net(state) → scalar → argmax(scores)
   MỚI: q_vecs = Q_net(state) → [M, N_OBJ] → Pareto dominance selection
        - Tìm tập non-dominated trong {Q(s,a_i) | i ∈ valid}
        - Trong tập non-dominated: phá tie bằng weighted sum CHỈ trong
          tập đó (hợp lệ vì tất cả members đều non-dominated với nhau)
        - Epsilon-greedy: nếu random → chọn uniform từ valid candidates

2. store_transition():
   CŨ:  lưu scalar r_H
   MỚI: lưu vector reward [r_accept, r_cost] RIÊNG BIỆT
        + scalar r_H_scalar (để tính Bellman target)

3. train_step():
   CŨ:  MSE(Q_scalar, Bellman_scalar)
   MỚI: MSE(Q_vec, Bellman_vec) — vector Bellman update
        Bellman target_vec[k] = r_vec[k] + γ * max_valid Q_target_vec[k]
        → Q-network học objective-aware value function

4. Replay buffer lưu vector reward thay vì scalar.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List, Optional, Tuple
from config import Config
from models.hl_scorer import HLSharedQScorer, N_OBJ_HL
from utils.graph_generator import SFCRequest
from utils.pareto import select_by_pareto_dominance


class HLReplayBuffer:
    """
    Buffer lưu trữ transitions với VECTOR reward thay vì scalar.

    Một transition:
      (z_global, sfc_feat, pareto_w, action_idx,
       reward_vec[N_OBJ_HL],
       next_z_global, next_sfc_feats, next_pareto_w,
       done)
    """

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self,
             z_global, sfc_feat, pareto_w,
             action_idx: int,
             reward_vec: np.ndarray,       # shape [N_OBJ_HL]
             next_z_global, next_sfc_feats, next_pareto_w,
             done: float):
        self.buf.append((
            z_global, sfc_feat, pareto_w,
            action_idx,
            reward_vec,
            next_z_global, next_sfc_feats, next_pareto_w,
            done
        ))

    def sample(self, batch_size: int):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


class HLAgent:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.epsilon = cfg.qnet.eps_start
        self.train_steps = 0
        self.q_net = HLSharedQScorer(cfg).to(device)
        self.target_net = HLSharedQScorer(cfg).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.qnet.lr)
        self.buffer = HLReplayBuffer(cfg.qnet.hl_buffer_size)

    # ─────────────────────────────────────────────────
    # Action Selection: PARETO DOMINANCE
    # ─────────────────────────────────────────────────

    def select_sfc(self,
                   z_global: torch.Tensor,
                   queue: List[SFCRequest],
                   t: int,
                   pareto_w: torch.Tensor,
                   blocked_until: Optional[dict] = None
                   ) -> Optional[Tuple[int, SFCRequest]]:
        """
        Chọn SFC từ queue dựa trên Pareto dominance của Q-vectors.

        Thay vì argmax(scalar score), bây giờ:
          1. Q_net trả về [M, N_OBJ_HL] cho M valid SFCs.
          2. Tìm non-dominated set trong M Q-vectors.
          3. Phá tie trong non-dominated set bằng weighted sum.
             (Hợp lệ: tất cả members đã non-dominated với nhau,
              không có solution nào tốt hơn hẳn solution khác.)
        """
        blocked_until = blocked_until or {}
        valid_indices = [
            i for i, q in enumerate(queue)
            if not q.is_expired(t) and blocked_until.get(q.sfc_id, t) <= t
        ]
        if not valid_indices:
            return None

        # Epsilon-greedy: random chọn trong valid candidates
        if random.random() < self.epsilon:
            chosen_local = random.choice(valid_indices)
            return chosen_local, queue[chosen_local]

        # Forward pass: lấy Q-vectors cho tất cả valid SFCs
        self.q_net.eval()
        with torch.no_grad():
            sfc_feats = [
                torch.tensor(queue[i].to_feature_vector(t),
                             dtype=torch.float32, device=self.device)
                for i in valid_indices
            ]
            sfc_feats_t = torch.stack(sfc_feats)        # [M, 7]
            m = len(valid_indices)
            z_global_exp = z_global.unsqueeze(0).expand(m, -1)   # [M, 64]
            pw_exp = pareto_w.unsqueeze(0).expand(m, -1)         # [M, 2]
            q_vecs = self.q_net(z_global_exp, sfc_feats_t, pw_exp)  # [M, N_OBJ_HL]
        self.q_net.train()

        # Pareto dominance selection trong M candidates
        q_np = q_vecs.cpu().numpy()                   # [M, N_OBJ_HL]
        valid_mask = np.ones(m, dtype=bool)           # tất cả M đều valid
        pw_np = pareto_w.cpu().numpy()

        # Hàm select_by_pareto_dominance trả về index trong [0, M)
        best_local = select_by_pareto_dominance(q_np, valid_mask, pw_np)
        best_idx = valid_indices[best_local]
        return best_idx, queue[best_idx]

    # ─────────────────────────────────────────────────
    # Store Transition: VECTOR REWARD
    # ─────────────────────────────────────────────────

    def store_transition(self,
                         z_global, sfc_feat, pareto_w,
                         action_idx: int,
                         reward_vec: np.ndarray,     # [N_OBJ_HL]
                         next_z_global, next_sfc_feats, next_pareto_w,
                         done: float):
        self.buffer.push(
            z_global, sfc_feat, pareto_w,
            action_idx,
            reward_vec,
            next_z_global, next_sfc_feats, next_pareto_w,
            done
        )

    # ─────────────────────────────────────────────────
    # Training: VECTOR BELLMAN UPDATE
    # ─────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """
        Vector Bellman update:
          target_vec[k] = r_vec[k] + γ * (1-done) * max_i Q_target[i, k]
        
        "max" ở đây được tính bằng Pareto dominance selection trên target net,
        sau đó lấy Q-vector của action được chọn.
        
        Loss: MSE(Q_vec, target_vec) trên cả N_OBJ_HL objectives đồng thời.
        """
        min_buf = min(self.cfg.qnet.batch_size,
                      self.cfg.qnet.hl_buffer_size // 10)
        if len(self.buffer) < min_buf:
            return None

        batch = self.buffer.sample(self.cfg.qnet.batch_size)
        (z_globals, sfc_feats, pareto_ws, actions,
         reward_vecs,
         next_z_globals, next_sfc_feats_list, next_pareto_ws, dones) = zip(*batch)

        z_globals_t = torch.stack(z_globals).to(self.device)          # [B, 64]
        sfc_feats_t = torch.stack(sfc_feats).to(self.device)          # [B, 7]
        pareto_ws_t = torch.stack(pareto_ws).to(self.device)          # [B, 2]
        rewards_t = torch.tensor(
            np.array(reward_vecs), dtype=torch.float32, device=self.device
        )                                                               # [B, N_OBJ_HL]
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)  # [B]

        # Current Q-vectors
        current_q = self.q_net(z_globals_t, sfc_feats_t, pareto_ws_t)  # [B, N_OBJ_HL]

        # Target Q-vectors via Pareto dominance selection on next state
        with torch.no_grad():
            next_q_vecs = []
            for i in range(len(batch)):
                nz = next_z_globals[i].to(self.device)
                nfeats = next_sfc_feats_list[i]
                npw = next_pareto_ws[i].to(self.device)

                if not nfeats:
                    # Terminal: không có next state
                    next_q_vecs.append(
                        torch.zeros(N_OBJ_HL, device=self.device)
                    )
                else:
                    nfeats_t = torch.stack(nfeats).to(self.device)   # [M', 7]
                    m_next = nfeats_t.shape[0]
                    nz_exp = nz.unsqueeze(0).expand(m_next, -1)
                    npw_exp = npw.unsqueeze(0).expand(m_next, -1)

                    # Q-vectors của tất cả next candidates
                    cand_q = self.target_net(nz_exp, nfeats_t, npw_exp)  # [M', N_OBJ_HL]

                    # Pareto dominance selection để chọn best next action
                    cand_np = cand_q.cpu().numpy()
                    valid_mask = np.ones(m_next, dtype=bool)
                    pw_np = npw.cpu().numpy()
                    best_local = select_by_pareto_dominance(cand_np, valid_mask, pw_np)
                    next_q_vecs.append(cand_q[best_local])

            next_q_t = torch.stack(next_q_vecs)  # [B, N_OBJ_HL]

        # Vector Bellman target
        target_q = (rewards_t
                    + self.cfg.qnet.gamma * next_q_t
                    * (1.0 - dones_t.unsqueeze(-1)))  # [B, N_OBJ_HL]

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.qnet.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
