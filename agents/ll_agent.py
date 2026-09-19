"""
ll_agent.py — Low-Level Agent với Pareto-based Node Selection

THAY ĐỔI so với bản cũ:

1. select_node():
   CŨ:  scores = (q_vecs * w).sum(dim=-1) → argmax(weighted_sum)
        → Đây là weighted sum selection, không phải Pareto-based.
   MỚI: Pareto dominance selection trong tập valid nodes
        - Q-net vẫn output [N_nodes, N_OBJ] (giữ nguyên LLNodeScorer)
        - Lọc valid nodes bằng cpu_mask
        - Tìm non-dominated set trong {Q(s, node_i) | mask[i]=True}
        - Phá tie trong non-dominated set bằng weighted sum (hợp lệ)

2. train_step():
   Giữ nguyên vector Bellman update (đã có từ bản gốc).
   Chỉ cập nhật: "best next action" dùng Pareto dominance thay vì
   weighted sum argmax.

3. store_transition(): Giữ nguyên (đã lưu vector reward từ bản gốc).

LÝ DO LL CÓ PHẦN WEIGHTED SUM CÒN LẠI:
  Trong train_step(), sau khi Pareto selection chọn best next action,
  ta lấy Q-vector của action đó làm Bellman target.
  Không có weighted sum trong vòng training chính.
  Tie-breaking trong select_by_pareto_dominance là weighted sum
  TRONG TẬP NON-DOMINATED — đây là bước phụ hợp lệ.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional, Tuple
from config import Config
from models.ll_dqn import LLNodeScorer, N_OBJ
from utils.pareto import select_by_pareto_dominance


class LLReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state_tuple, next_state_tuple, reward_vec, done: float):
        self.buf.append((state_tuple, next_state_tuple, reward_vec, done))

    def sample(self, batch_size: int):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


class LLAgent:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.epsilon = cfg.qnet.eps_start
        self.train_steps = 0
        self.q_net = LLNodeScorer(cfg).to(device)
        self.target_net = LLNodeScorer(cfg).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.qnet.lr)
        self.buffer = LLReplayBuffer(cfg.qnet.ll_buffer_size)

    # ─────────────────────────────────────────────────
    # Action Selection: PARETO DOMINANCE
    # ─────────────────────────────────────────────────

    def select_node(self,
                    z_global, z_prev_node, z_nodes,
                    vnf_features, sfc_features,
                    cpu_mask: np.ndarray,
                    pareto_w: torch.Tensor
                    ) -> Optional[int]:
        """
        Chọn node dựa trên Pareto dominance của Q-vectors.

        CŨ: scores = (q_vecs * w).sum(-1) → masked argmax
        MỚI:
          1. Lấy Q-vectors [N_nodes, N_OBJ] từ LLNodeScorer.
          2. Apply cpu_mask để lọc valid nodes.
          3. select_by_pareto_dominance() tìm non-dominated set
             trong valid nodes, sau đó phá tie bằng pareto_w.
        """
        valid_nodes = np.where(cpu_mask)[0]
        if len(valid_nodes) == 0:
            return None

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return int(random.choice(valid_nodes))

        num_nodes = z_nodes.shape[0]
        self.q_net.eval()
        with torch.no_grad():
            zg_exp = z_global.unsqueeze(0).expand(num_nodes, -1)
            zp_exp = z_prev_node.unsqueeze(0).expand(num_nodes, -1)
            vf_exp = vnf_features.unsqueeze(0).expand(num_nodes, -1)
            sf_exp = sfc_features.unsqueeze(0).expand(num_nodes, -1)
            pw_exp = pareto_w.unsqueeze(0).expand(num_nodes, -1)
            q_vecs = self.q_net(zg_exp, zp_exp, z_nodes, vf_exp, sf_exp, pw_exp)
            # q_vecs: [N_nodes, N_OBJ]
        self.q_net.train()

        q_np = q_vecs.cpu().numpy()           # [N_nodes, N_OBJ]
        pw_np = pareto_w.cpu().numpy()        # [N_OBJ]

        # Pareto dominance selection với cpu_mask
        chosen = select_by_pareto_dominance(q_np, cpu_mask, pw_np)
        return int(chosen)

    # ─────────────────────────────────────────────────
    # Store Transition (giữ nguyên interface)
    # ─────────────────────────────────────────────────

    def store_transition(self,
                         state_tuple: Tuple,
                         next_state_tuple: Optional[Tuple],
                         reward_vec: np.ndarray,
                         done: float):
        self.buffer.push(state_tuple, next_state_tuple, reward_vec, done)

    # ─────────────────────────────────────────────────
    # Training: VECTOR BELLMAN + PARETO DOMINANCE
    # ─────────────────────────────────────────────────

    def train_step(self, pareto_w: torch.Tensor) -> Optional[float]:
        """
        Vector Bellman update với Pareto dominance cho next action selection.

        Thay đổi so với bản cũ:
          CŨ: best_idx = scores.argmax()  (scores = weighted sum)
          MỚI: best_idx = select_by_pareto_dominance(cand_q, mask, pw)
        """
        min_buf = min(self.cfg.qnet.batch_size,
                      self.cfg.qnet.ll_buffer_size // 10)
        if len(self.buffer) < min_buf:
            return None

        batch = self.buffer.sample(self.cfg.qnet.batch_size)
        states, next_states, reward_vecs, dones = zip(*batch)

        zg_l, zp_l, zc_l, vf_l, sf_l, pw_l = zip(*states)
        zg_t = torch.stack(zg_l).to(self.device)
        zp_t = torch.stack(zp_l).to(self.device)
        zc_t = torch.stack(zc_l).to(self.device)
        vf_t = torch.stack(vf_l).to(self.device)
        sf_t = torch.stack(sf_l).to(self.device)
        pw_t = torch.stack(pw_l).to(self.device)

        rewards_t = torch.tensor(
            np.array(reward_vecs), dtype=torch.float32, device=self.device
        )                                                          # [B, N_OBJ]
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.q_net(zg_t, zp_t, zc_t, vf_t, sf_t, pw_t)  # [B, N_OBJ]

        with torch.no_grad():
            next_q_vecs = []
            for i, ns in enumerate(next_states):
                if ns is None:
                    next_q_vecs.append(torch.zeros(N_OBJ, device=self.device))
                else:
                    nzg, nzp, n_all_z, nvf, nsf, n_mask, npw = ns
                    n_all_z = n_all_z.to(self.device)
                    num_nodes = n_all_z.shape[0]
                    nzg_exp = nzg.unsqueeze(0).expand(num_nodes, -1).to(self.device)
                    nzp_exp = nzp.unsqueeze(0).expand(num_nodes, -1).to(self.device)
                    nvf_exp = nvf.unsqueeze(0).expand(num_nodes, -1).to(self.device)
                    nsf_exp = nsf.unsqueeze(0).expand(num_nodes, -1).to(self.device)
                    npw_exp = npw.unsqueeze(0).expand(num_nodes, -1).to(self.device)

                    cand_q = self.target_net(
                        nzg_exp, nzp_exp, n_all_z, nvf_exp, nsf_exp, npw_exp
                    )  # [N_nodes, N_OBJ]

                    mask_np = np.asarray(n_mask, dtype=bool)
                    if not mask_np.any():
                        next_q_vecs.append(torch.zeros(N_OBJ, device=self.device))
                    else:
                        # Pareto dominance selection cho next action
                        cand_np = cand_q.cpu().numpy()
                        pw_np = npw.numpy() if isinstance(npw, torch.Tensor) else npw
                        best_idx = select_by_pareto_dominance(cand_np, mask_np, pw_np)
                        next_q_vecs.append(cand_q[best_idx])

            next_q_t = torch.stack(next_q_vecs)  # [B, N_OBJ]

        target_q = (rewards_t
                    + self.cfg.qnet.gamma * next_q_t
                    * (1.0 - dones_t.unsqueeze(-1)))  # [B, N_OBJ]

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.qnet.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
