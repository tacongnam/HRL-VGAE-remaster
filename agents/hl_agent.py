import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List, Optional, Tuple
from config import Config
from hl_scorer import HLSharedQScorer, N_OBJ_HL
from graph_generator import SFCRequest
from pareto import (
    non_dominated, non_dominated_indices, select_by_hypervolume,
    prune_by_hypervolume, hv_score_for_action, build_target_q_set
)


class HLReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, z_global, sfc_feat, pareto_w,
             action_idx: int,
             reward_vec: np.ndarray,
             next_z_global, next_sfc_feats, next_pareto_w,
             done: float,
             valid_action_mask: Optional[np.ndarray] = None):
        self.buf.append((
            z_global, sfc_feat, pareto_w,
            action_idx,
            reward_vec.copy(),
            next_z_global,
            [f for f in next_sfc_feats],
            next_pareto_w,
            done,
            valid_action_mask.copy() if valid_action_mask is not None else None,
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
        self._ref_point = cfg.pareto.hv_reference_point()
        self._max_q_vecs = cfg.qnet.max_q_vectors_per_action
        self._tie_rng = random.Random(cfg.train.seed)
        self.q_net = HLSharedQScorer(cfg).to(device)
        self.target_net = HLSharedQScorer(cfg).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.qnet.lr)
        self.buffer = HLReplayBuffer(cfg.qnet.hl_buffer_size)

    def _get_q_sets(self, net, z_global_exp, sfc_feats_t, pw_exp,
                    valid_indices) -> List[List[np.ndarray]]:
        q_vecs = net(z_global_exp, sfc_feats_t, pw_exp)
        q_np = q_vecs.cpu().numpy()
        q_sets = []
        for j, global_idx in enumerate(valid_indices):
            vec = q_np[j]
            nd_set = non_dominated([vec])
            nd_set = prune_by_hypervolume(nd_set, self._max_q_vecs, self._ref_point)
            q_sets.append(nd_set)
        return q_sets

    def select_sfc(self,
                   z_global: torch.Tensor,
                   queue: List[SFCRequest],
                   t: int,
                   pareto_w: torch.Tensor,
                   blocked_until: Optional[dict] = None,
                   verbose: bool = False
                   ) -> Optional[Tuple[int, SFCRequest]]:
        blocked_until = blocked_until or {}
        valid_indices = [
            i for i, q in enumerate(queue)
            if not q.is_expired(t) and blocked_until.get(q.sfc_id, t) <= t
        ]
        if not valid_indices:
            return None

        if random.random() < self.epsilon:
            chosen_local = random.choice(valid_indices)
            return chosen_local, queue[chosen_local]

        self.q_net.eval()
        with torch.no_grad():
            sfc_feats = [
                torch.tensor(queue[i].to_feature_vector(t),
                             dtype=torch.float32, device=self.device)
                for i in valid_indices
            ]
            sfc_feats_t = torch.stack(sfc_feats)
            m = len(valid_indices)
            z_global_exp = z_global.unsqueeze(0).expand(m, -1)
            pw_exp = pareto_w.unsqueeze(0).expand(m, -1)
            q_sets = self._get_q_sets(
                self.q_net, z_global_exp, sfc_feats_t, pw_exp, valid_indices)
        self.q_net.train()

        valid_mask = np.ones(m, dtype=bool)
        best_local = select_by_hypervolume(q_sets, valid_mask, self._ref_point,
                                           tie_break_rng=self._tie_rng)

        if verbose:
            hv_scores = [hv_score_for_action(qs, self._ref_point) for qs in q_sets]
            print(f"  [HL-HV] ref={self._ref_point} | "
                  f"hv_scores={[f'{s:.4f}' for s in hv_scores]} | "
                  f"chosen_local={best_local} q_set_size={len(q_sets[best_local])}")

        best_idx = valid_indices[best_local]
        return best_idx, queue[best_idx]

    def store_transition(self,
                         z_global, sfc_feat, pareto_w,
                         action_idx: int,
                         reward_vec: np.ndarray,
                         next_z_global, next_sfc_feats, next_pareto_w,
                         done: float,
                         valid_action_mask: Optional[np.ndarray] = None):
        self.buffer.push(
            z_global, sfc_feat, pareto_w,
            action_idx, reward_vec,
            next_z_global, next_sfc_feats, next_pareto_w,
            done, valid_action_mask,
        )

    def train_step(self) -> Optional[float]:
        min_buf = min(self.cfg.qnet.batch_size,
                      self.cfg.qnet.hl_buffer_size // 10)
        if len(self.buffer) < min_buf:
            return None

        batch = self.buffer.sample(self.cfg.qnet.batch_size)
        (z_globals, sfc_feats, pareto_ws, actions,
         reward_vecs,
         next_z_globals, next_sfc_feats_list, next_pareto_ws,
         dones, valid_masks) = zip(*batch)

        z_globals_t = torch.stack(z_globals).to(self.device)
        sfc_feats_t = torch.stack(sfc_feats).to(self.device)
        pareto_ws_t = torch.stack(pareto_ws).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.q_net(z_globals_t, sfc_feats_t, pareto_ws_t)

        with torch.no_grad():
            target_q_vecs = []
            for i in range(len(batch)):
                r_vec = np.asarray(reward_vecs[i], dtype=np.float64)
                nfeats = next_sfc_feats_list[i]
                is_done = bool(dones[i] > 0.5)

                if is_done or not nfeats:
                    target_sets = [r_vec.copy()]
                else:
                    nz = next_z_globals[i].to(self.device)
                    npw = next_pareto_ws[i].to(self.device)
                    nfeats_t = torch.stack(nfeats).to(self.device)
                    m_next = nfeats_t.shape[0]
                    nz_exp = nz.unsqueeze(0).expand(m_next, -1)
                    npw_exp = npw.unsqueeze(0).expand(m_next, -1)

                    mask = (valid_masks[i] if valid_masks[i] is not None
                            else np.ones(m_next, dtype=bool))

                    cand_q = self.target_net(nz_exp, nfeats_t, npw_exp)
                    cand_np = cand_q.cpu().numpy()

                    next_q_sets = []
                    for j in range(m_next):
                        if mask[j] if j < len(mask) else True:
                            next_q_sets.append([cand_np[j]])

                    target_sets = build_target_q_set(
                        r_vec, next_q_sets,
                        self.cfg.qnet.gamma,
                        self._max_q_vecs,
                        self._ref_point,
                        done=is_done,
                    )

                if target_sets:
                    pts = np.array(target_sets, dtype=np.float32)
                    rep = pts.mean(axis=0)
                else:
                    rep = r_vec.astype(np.float32)
                target_q_vecs.append(rep)

        target_t = torch.tensor(
            np.stack(target_q_vecs), dtype=torch.float32, device=self.device)

        loss = nn.MSELoss()(current_q, target_t)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.qnet.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()