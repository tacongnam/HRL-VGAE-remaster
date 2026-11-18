import random
from collections import deque
from typing import Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from models import HLSharedQScorer
from utils.pareto import ParetoScalarizer


class HLReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(
        self,
        z_global,
        sfc_feat,
        pareto_w,
        action_idx: int,
        reward_vec: np.ndarray,
        next_z_global,
        next_sfc_feats,
        next_pareto_w,
        done: float,
        valid_action_mask: Optional[np.ndarray] = None,
    ):
        self.buf.append(
            (
                z_global,
                sfc_feat,
                pareto_w,
                action_idx,
                reward_vec.copy(),
                next_z_global,
                next_sfc_feats,
                next_pareto_w,
                done,
                valid_action_mask,
            )
        )

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
        self._tie_rng = random.Random(cfg.train.seed)
        self.q_net = HLSharedQScorer(cfg).to(device)
        self.target_net = HLSharedQScorer(cfg).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.qnet.lr)
        self.buffer = HLReplayBuffer(cfg.qnet.hl_buffer_size)

    def select_sfc(
        self,
        z_global: torch.Tensor,
        queue: List,
        t: int,
        pareto_w: torch.Tensor,
        z_nodes: Optional[torch.Tensor] = None,
        env=None,
    ) -> Optional[Tuple[int, Any]]:
        valid_indices = [
            i for i, q in enumerate(queue) if not q.is_expired(t)
        ]
        if not valid_indices:
            return None

        mode = self.cfg.train.ablation_mode
        if mode == "DefaultBaseline":
            best_idx = min(valid_indices, key=lambda i: queue[i].deadline - t)
            return best_idx, queue[best_idx]
        elif mode == "Random-Fit":
            chosen = random.choice(valid_indices)
            return chosen, queue[chosen]
        elif mode == "GreedyFIFO":
            best_idx = min(valid_indices, key=lambda i: queue[i].arrival_time)
            return best_idx, queue[best_idx]

        if random.random() < self.epsilon:
            chosen = random.choice(valid_indices)
            return chosen, queue[chosen]

        m = len(valid_indices)
        self.q_net.eval()
        with torch.inference_mode():
            if z_nodes is not None and env is not None:
                src_idx = [queue[i].source for i in valid_indices]
                dst_idx = [queue[i].dest for i in valid_indices]
                scalars_np = np.stack([env.get_sfc_scalar_features(queue[i], t) for i in valid_indices]).astype(np.float32)
                scalars_t = torch.from_numpy(scalars_np).to(self.device)
                sfc_feats_t = torch.cat([z_nodes[src_idx].to(self.device), z_nodes[dst_idx].to(self.device), scalars_t], dim=-1)
            else:
                sfc_feats_np = np.stack([queue[i].to_feature_vector(t) for i in valid_indices]).astype(np.float32)
                sfc_feats_t = torch.from_numpy(sfc_feats_np).to(self.device)
            zg_exp = z_global.unsqueeze(0).expand(m, -1)
            pw_exp = pareto_w.unsqueeze(0).expand(m, -1)
            q_vecs = self.q_net(zg_exp, sfc_feats_t, pw_exp)
            q_np = q_vecs.cpu().numpy()

        w_np = pareto_w.detach().cpu().numpy()
        scores = ParetoScalarizer.scalarize_q(q_np, w_np)
        best_score = float(np.max(scores))
        tied = np.flatnonzero(scores >= best_score - 1e-12)
        best_local = int(tied[0]) if len(tied) == 1 else int(self._tie_rng.choice(tied.tolist()))
        best_idx = valid_indices[best_local]
        return best_idx, queue[best_idx]

    def store_transition(self, *args, **kwargs):
        self.buffer.push(*args, **kwargs)

    def train_step(self) -> Optional[float]:
        min_buf = min(self.cfg.qnet.batch_size, self.cfg.qnet.hl_buffer_size // 10)
        if len(self.buffer) < min_buf:
            return None

        batch = self.buffer.sample(self.cfg.qnet.batch_size)
        (
            z_globals,
            sfc_feats,
            pareto_ws,
            actions,
            reward_vecs,
            next_z_globals,
            next_sfc_feats_list,
            next_pareto_ws,
            dones,
            valid_masks,
        ) = zip(*batch)

        self.q_net.train()
        zg_t = torch.stack(z_globals).to(self.device)
        sf_t = torch.stack(sfc_feats).to(self.device)
        pw_t = torch.stack(pareto_ws).to(self.device)
        current_q = self.q_net(zg_t, sf_t, pw_t)

        eval_zg, eval_sf, eval_pw = [], [], []
        slices = []
        cursor = 0

        for i in range(len(batch)):
            is_done = bool(dones[i] > 0.5)
            nfeats = next_sfc_feats_list[i]
            if is_done or nfeats is None or len(nfeats) == 0:
                slices.append((cursor, cursor))
            else:
                m_next = len(nfeats)
                eval_zg.append(next_z_globals[i].unsqueeze(0).expand(m_next, -1))
                eval_sf.append(torch.stack(nfeats) if isinstance(nfeats[0], torch.Tensor) else torch.from_numpy(np.stack(nfeats)))
                eval_pw.append(next_pareto_ws[i].unsqueeze(0).expand(m_next, -1))
                slices.append((cursor, cursor + m_next))
                cursor += m_next

        all_cand_np = None
        if eval_zg:
            batched_zg = torch.cat(eval_zg, dim=0).to(self.device)
            batched_sf = torch.cat(eval_sf, dim=0).to(self.device)
            batched_pw = torch.cat(eval_pw, dim=0).to(self.device)
            with torch.inference_mode():
                all_cand_np = self.target_net(batched_zg, batched_sf, batched_pw).cpu().numpy()

        pw_next_np_all = torch.stack(next_pareto_ws).detach().cpu().numpy()
        target_q_vecs = np.empty((len(batch), 2), dtype=np.float32)
        for i in range(len(batch)):
            r_vec = np.asarray(reward_vecs[i], dtype=np.float64)
            is_done = bool(dones[i] > 0.5)
            st, ed = slices[i]
            if is_done or st == ed or all_cand_np is None:
                target_q_vecs[i] = r_vec.astype(np.float32)
            else:
                cand_np = all_cand_np[st:ed]
                mask = valid_masks[i]
                if mask is not None and len(mask) == cand_np.shape[0]:
                    cand_np = cand_np[mask]
                if cand_np.shape[0] == 0:
                    target_q_vecs[i] = r_vec.astype(np.float32)
                else:
                    w_next = pw_next_np_all[i]
                    cand_scores = ParetoScalarizer.scalarize_q(cand_np, w_next)
                    best_next = int(np.argmax(cand_scores))
                    target_q_vecs[i] = (r_vec + self.cfg.qnet.gamma * cand_np[best_next]).astype(np.float32)

        target_t = torch.from_numpy(target_q_vecs).to(self.device)
        loss = nn.MSELoss()(current_q, target_t)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.qnet.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())