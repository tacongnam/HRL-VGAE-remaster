import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional, Tuple, List
from config import Config
from models.ll_dqn import LLNodeScorer, N_OBJ
from utils.pareto import (
    non_dominated,
    select_by_hypervolume,
    prune_by_hypervolume,
    hv_score_for_action,
    build_target_q_set,
)


class LLReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state_tuple, next_state_tuple, reward_vec, done: float):
        self.buf.append((state_tuple, next_state_tuple, reward_vec.copy(), done))

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
        self._ref_point = cfg.pareto.hv_reference_point()
        self._max_q_vecs = cfg.qnet.max_q_vectors_per_action
        self._tie_rng = random.Random(cfg.train.seed + 1)
        self.q_net = LLNodeScorer(cfg).to(device)
        self.target_net = LLNodeScorer(cfg).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.qnet.lr)
        self.buffer = LLReplayBuffer(cfg.qnet.ll_buffer_size)

    def select_node(
        self,
        z_global,
        z_prev_node,
        z_nodes,
        vnf_features,
        sfc_features,
        cpu_mask: np.ndarray,
        pareto_w: torch.Tensor,
        verbose: bool = False,
    ) -> Optional[int]:
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

        q_np = q_vecs.cpu().numpy()
        q_sets: List[List[np.ndarray]] = []
        for i in range(num_nodes):
            if cpu_mask[i]:
                nd_set = non_dominated([q_np[i]])
                nd_set = prune_by_hypervolume(nd_set, self._max_q_vecs, self._ref_point)
                q_sets.append(nd_set)
            else:
                q_sets.append([])

        chosen = select_by_hypervolume(
            q_sets, cpu_mask, self._ref_point, tie_break_rng=self._tie_rng
        )

        if verbose:
            hv_scores = [
                hv_score_for_action(q_sets[i], self._ref_point) if cpu_mask[i] else 0.0
                for i in range(num_nodes)
            ]
            top5 = sorted(enumerate(hv_scores), key=lambda x: -x[1])[:5]
            print(
                f"  [LL-HV] ref={self._ref_point} | "
                f"top5={([(n, f'{s:.4f}') for n, s in top5])} | "
                f"chosen={chosen} q_set_size={len(q_sets[chosen])}"
            )

        return int(chosen)

    def store_transition(
        self,
        state_tuple: Tuple,
        next_state_tuple: Optional[Tuple],
        reward_vec: np.ndarray,
        done: float,
    ):
        self.buffer.push(state_tuple, next_state_tuple, reward_vec, done)

    def train_step(self, pareto_w: torch.Tensor) -> Optional[float]:
        min_buf = min(self.cfg.qnet.batch_size, self.cfg.qnet.ll_buffer_size // 10)
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

        current_q = self.q_net(zg_t, zp_t, zc_t, vf_t, sf_t, pw_t)

        with torch.no_grad():
            target_q_vecs = []
            for i, ns in enumerate(next_states):
                r_vec = np.asarray(reward_vecs[i], dtype=np.float64)
                is_done = bool(dones[i] > 0.5)

                if ns is None or is_done:
                    target_sets = [r_vec.copy()]
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
                    )
                    cand_np = cand_q.cpu().numpy()

                    mask_np = np.asarray(n_mask, dtype=bool)
                    next_q_sets = [
                        [cand_np[j]] if (j < len(mask_np) and mask_np[j]) else []
                        for j in range(num_nodes)
                    ]
                    next_q_sets_valid = [qs for qs in next_q_sets if qs]

                    target_sets = build_target_q_set(
                        r_vec,
                        next_q_sets_valid,
                        self.cfg.qnet.gamma,
                        self._max_q_vecs,
                        self._ref_point,
                        done=is_done,
                    )

                if target_sets:
                    rep = np.mean(target_sets, axis=0).astype(np.float32)
                else:
                    rep = r_vec.astype(np.float32)
                target_q_vecs.append(rep)

        target_t = torch.tensor(
            np.stack(target_q_vecs), dtype=torch.float32, device=self.device
        )

        loss = nn.MSELoss()(current_q, target_t)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.qnet.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
