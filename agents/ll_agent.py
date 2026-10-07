import random
from collections import deque

import networkx as nx
import numpy as np
import torch
from torch import nn, optim

from config import Config
from models import LLNodeScorer
from utils import build_target_q_set, non_dominated, prune_by_hypervolume, select_by_hypervolume


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
        self._obs_min = np.array([np.inf, np.inf], dtype=np.float64)
        self._max_q_vecs = cfg.qnet.max_q_vectors_per_action
        self._tie_rng = random.Random(cfg.train.seed + 1)
        self.q_net = LLNodeScorer(cfg).to(device)
        self.target_net = LLNodeScorer(cfg).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.qnet.lr)
        self.buffer = LLReplayBuffer(cfg.qnet.ll_buffer_size)
        self._obj_scale = torch.tensor([1.0, 1000.0], dtype=torch.float32, device=device)

    def _update_ref_point(self, q_np: np.ndarray):
        if q_np.size == 0:
            return
        batch_min = q_np.min(axis=0).astype(np.float64)
        m = self.cfg.pareto.ll_ref_momentum
        finite_mask = np.isfinite(self._obs_min)
        self._obs_min = np.where(
            finite_mask, np.minimum(batch_min, m * self._obs_min + (1.0 - m) * batch_min), batch_min
        )
        self._ref_point = self._obs_min - self.cfg.pareto.ll_ref_margin

    def select_node(
        self,
        z_global: torch.Tensor,
        z_prev_node: torch.Tensor,
        z_nodes: torch.Tensor,
        vnf_features: torch.Tensor,
        sfc_features: torch.Tensor,
        cpu_mask: np.ndarray,
        pareto_w: torch.Tensor,
        current_source: int | None = None,
        G: nx.Graph | None = None,
        k_hop: int = 3,
        verbose: bool = False,
    ) -> int | None:
        valid_nodes = np.flatnonzero(cpu_mask)
        if valid_nodes.size == 0:
            return None
        if random.random() < self.epsilon:
            if G is not None and current_source is not None and k_hop > 0:
                try:
                    lengths = nx.single_source_shortest_path_length(G, current_source, cutoff=k_hop)
                    candidates = [n for n in valid_nodes if n in lengths]
                    if candidates:
                        return int(random.choice(candidates))
                except Exception:  # noqa: BLE001, S110
                    pass
            return int(random.choice(valid_nodes))

        num_nodes = z_nodes.shape[0]
        k = len(valid_nodes)
        self.q_net.eval()
        with torch.inference_mode():
            z_valid = z_nodes[valid_nodes]
            zg_exp = z_global.unsqueeze(0).expand(k, -1)
            zp_exp = z_prev_node.unsqueeze(0).expand(k, -1)
            vf_exp = vnf_features.unsqueeze(0).expand(k, -1)
            sf_exp = sfc_features.unsqueeze(0).expand(k, -1)
            pw_exp = pareto_w.unsqueeze(0).expand(k, -1)
            q_vecs = self.q_net(zg_exp, zp_exp, z_valid, vf_exp, sf_exp, pw_exp)
            q_np = q_vecs.cpu().numpy()

        self._update_ref_point(q_np)
        q_sets: list[list[np.ndarray]] = [[] for _ in range(num_nodes)]
        for local_j, global_i in enumerate(valid_nodes):
            nd_set = non_dominated([q_np[local_j]])
            q_sets[global_i] = prune_by_hypervolume(nd_set, self._max_q_vecs, self._ref_point)

        chosen = select_by_hypervolume(
            q_sets, cpu_mask, self._ref_point, tie_break_rng=self._tie_rng
        )
        return int(chosen)

    def store_transition(self, state_tuple, next_state_tuple, reward_vec: np.ndarray, done: float):
        self.buffer.push(state_tuple, next_state_tuple, reward_vec, done)

    def train_step(self, pareto_w: torch.Tensor | None = None) -> float | None:
        min_buf = min(self.cfg.qnet.batch_size, self.cfg.qnet.ll_buffer_size // 10)
        if len(self.buffer) < min_buf:
            return None

        batch = self.buffer.sample(self.cfg.qnet.batch_size)
        states, next_states, reward_vecs, dones = zip(*batch)

        zg_l, zp_l, zc_l, vf_l, sf_l, pw_l = zip(*states)
        self.q_net.train()
        current_q = self.q_net(
            torch.stack(zg_l).to(self.device),
            torch.stack(zp_l).to(self.device),
            torch.stack(zc_l).to(self.device),
            torch.stack(vf_l).to(self.device),
            torch.stack(sf_l).to(self.device),
            torch.stack(pw_l).to(self.device),
        )

        eval_zg, eval_zp, eval_zv, eval_vf, eval_sf, eval_pw = [], [], [], [], [], []
        slices = []
        cursor = 0

        for i, ns in enumerate(next_states):
            is_done = bool(dones[i] > 0.5)
            if ns is None or is_done:
                slices.append((cursor, cursor))
            else:
                nzg, nzp, n_all_z, nvf, nsf, n_mask, npw = ns
                valid_idx = np.flatnonzero(np.asarray(n_mask, dtype=bool))
                k = len(valid_idx)
                if k == 0:
                    slices.append((cursor, cursor))
                else:
                    eval_zg.append(nzg.unsqueeze(0).expand(k, -1))
                    eval_zp.append(nzp.unsqueeze(0).expand(k, -1))
                    eval_zv.append(n_all_z[valid_idx])
                    eval_vf.append(nvf.unsqueeze(0).expand(k, -1))
                    eval_sf.append(nsf.unsqueeze(0).expand(k, -1))
                    eval_pw.append(npw.unsqueeze(0).expand(k, -1))
                    slices.append((cursor, cursor + k))
                    cursor += k

        all_cand_np = None
        if eval_zg:
            batched_zg = torch.cat(eval_zg, dim=0).to(self.device)
            batched_zp = torch.cat(eval_zp, dim=0).to(self.device)
            batched_zv = torch.cat(eval_zv, dim=0).to(self.device)
            batched_vf = torch.cat(eval_vf, dim=0).to(self.device)
            batched_sf = torch.cat(eval_sf, dim=0).to(self.device)
            batched_pw = torch.cat(eval_pw, dim=0).to(self.device)
            with torch.inference_mode():
                all_cand_np = (
                    self.target_net(
                        batched_zg, batched_zp, batched_zv, batched_vf, batched_sf, batched_pw
                    )
                    .cpu()
                    .numpy()
                )

        target_q_vecs = []
        for i, ns in enumerate(next_states):
            r_vec = np.asarray(reward_vecs[i], dtype=np.float64)
            is_done = bool(dones[i] > 0.5)
            st, ed = slices[i]
            if ns is None or is_done or st == ed or all_cand_np is None:
                target_sets = [r_vec]
            else:
                cand_np = all_cand_np[st:ed]
                k = cand_np.shape[0]
                target_sets = build_target_q_set(
                    r_vec,
                    [[cand_np[j]] for j in range(k)],
                    self.cfg.qnet.gamma,
                    self._max_q_vecs,
                    self._ref_point,
                    done=is_done,
                )
            target_q_vecs.append(
                np.mean(target_sets, axis=0).astype(np.float32)
                if target_sets
                else r_vec.astype(np.float32)
            )

        target_t = torch.from_numpy(np.stack(target_q_vecs)).to(self.device)
        loss = nn.MSELoss()(current_q * self._obj_scale, target_t * self._obj_scale)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.qnet.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())
