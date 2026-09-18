import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Optional, Tuple
from config import Config
from models.hl_scorer import HLSharedQScorer
from utils.graph_generator import SFCRequest

class HLReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, z_global, sfc_feat, pareto_w, action_idx, reward, next_z_global, next_sfc_feats, next_pareto_w, done):
        self.buf.append((z_global, sfc_feat, pareto_w, action_idx, reward, next_z_global, next_sfc_feats, next_pareto_w, done))

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

    def select_sfc(self, z_global: torch.Tensor, queue: List[SFCRequest], t: int, pareto_w: torch.Tensor, blocked_until: Optional[dict] = None) -> Optional[Tuple[int, SFCRequest]]:
        blocked_until = blocked_until or {}
        valid_indices = [i for i, q in enumerate(queue) if not q.is_expired(t) and blocked_until.get(q.sfc_id, t) <= t]
        if not valid_indices:
            return None
        if random.random() < self.epsilon:
            chosen_local = random.choice(valid_indices)
            return chosen_local, queue[chosen_local]
        self.q_net.eval()
        with torch.no_grad():
            sfc_feats = [torch.tensor(queue[i].to_feature_vector(t), dtype=torch.float32, device=self.device) for i in valid_indices]
            sfc_feats_t = torch.stack(sfc_feats)
            m = len(valid_indices)
            z_global_exp = z_global.unsqueeze(0).expand(m, -1)
            pw_exp = pareto_w.unsqueeze(0).expand(m, -1)
            scores = self.q_net(z_global_exp, sfc_feats_t, pw_exp).squeeze(-1)
        self.q_net.train()
        best_idx = valid_indices[scores.argmax().item()]
        return best_idx, queue[best_idx]

    def store_transition(self, z_global, sfc_feat, pareto_w, action_idx, reward, next_z_global, next_sfc_feats, next_pareto_w, done):
        self.buffer.push(z_global, sfc_feat, pareto_w, action_idx, reward, next_z_global, next_sfc_feats, next_pareto_w, done)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < min(self.cfg.qnet.batch_size, self.cfg.qnet.hl_buffer_size // 10):
            return None
        batch = self.buffer.sample(self.cfg.qnet.batch_size)
        z_globals, sfc_feats, pareto_ws, actions, rewards, next_z_globals, next_sfc_feats_list, next_pareto_ws, dones = zip(*batch)
        z_globals_t = torch.stack(z_globals).to(self.device)
        sfc_feats_t = torch.stack(sfc_feats).to(self.device)
        pareto_ws_t = torch.stack(pareto_ws).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        current_q = self.q_net(z_globals_t, sfc_feats_t, pareto_ws_t).squeeze(-1)
        with torch.no_grad():
            next_q_vals = []
            for i in range(len(batch)):
                nz = next_z_globals[i].to(self.device)
                nfeats = next_sfc_feats_list[i]
                npw = next_pareto_ws[i].to(self.device)
                if not nfeats:
                    next_q_vals.append(torch.tensor(0.0, device=self.device))
                else:
                    nfeats_t = torch.stack(nfeats).to(self.device)
                    m = nfeats_t.shape[0]
                    nz_exp = nz.unsqueeze(0).expand(m, -1)
                    npw_exp = npw.unsqueeze(0).expand(m, -1)
                    scores = self.target_net(nz_exp, nfeats_t, npw_exp).squeeze(-1)
                    next_q_vals.append(scores.max())
            next_q_t = torch.stack(next_q_vals)
        target_q = rewards_t + self.cfg.qnet.gamma * next_q_t * (1.0 - dones_t)
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.cfg.qnet.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()