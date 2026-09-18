import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional, Tuple
from config import Config
from models.ll_dqn import LLNodeScorer, N_OBJ

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

    def select_node(self, z_global, z_prev_node, z_nodes, vnf_features, sfc_features,
                    cpu_mask: np.ndarray, pareto_w: torch.Tensor) -> Optional[int]:
        valid_nodes = np.where(cpu_mask)[0]
        if len(valid_nodes) == 0:
            return None
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
        self.q_net.train()
        w = pareto_w.to(q_vecs.device)
        scores = (q_vecs * w.unsqueeze(0)).sum(dim=-1)
        mask_tensor = torch.tensor(cpu_mask, dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(~mask_tensor, float('-inf'))
        return int(scores.argmax().item())

    def store_transition(self, state_tuple: Tuple, next_state_tuple: Optional[Tuple],
                         reward_vec: np.ndarray, done: float):
        self.buffer.push(state_tuple, next_state_tuple, reward_vec, done)

    def train_step(self, pareto_w: torch.Tensor) -> Optional[float]:
        if len(self.buffer) < min(self.cfg.qnet.batch_size, self.cfg.qnet.ll_buffer_size // 10):
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
        rewards_t = torch.tensor(np.array(reward_vecs), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        current_q = self.q_net(zg_t, zp_t, zc_t, vf_t, sf_t, pw_t)
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
                    cand_q = self.target_net(nzg_exp, nzp_exp, n_all_z, nvf_exp, nsf_exp, npw_exp)
                    mask_t = torch.tensor(n_mask, dtype=torch.bool, device=self.device)
                    w_i = pw_t[i]
                    scores = (cand_q * w_i.unsqueeze(0)).sum(dim=-1)
                    scores = scores.masked_fill(~mask_t, float('-inf'))
                    if not mask_t.any():
                        next_q_vecs.append(torch.zeros(N_OBJ, device=self.device))
                    else:
                        best_idx = scores.argmax()
                        next_q_vecs.append(cand_q[best_idx])
            next_q_t = torch.stack(next_q_vecs)
        target_q = rewards_t + self.cfg.qnet.gamma * next_q_t * (1.0 - dones_t.unsqueeze(-1))
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.cfg.qnet.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()