import random
import bisect
from collections import deque

import networkx as nx
import numpy as np
import torch
from torch import nn, optim

from config import Config
from models import LLNodeScorer
from utils.pareto import ParetoScalarizer


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
        self._tie_rng = random.Random(cfg.train.seed + 1)
        self.q_net = LLNodeScorer(cfg).to(device)
        self.target_net = LLNodeScorer(cfg).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.qnet.lr)
        self.buffer = LLReplayBuffer(cfg.qnet.ll_buffer_size)

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
        t: float = 0.0,
    ) -> int | None:
        valid_nodes = np.flatnonzero(cpu_mask)
        if valid_nodes.size == 0:
            return None

        mode = self.cfg.train.ablation_mode
        if mode in ["DefaultBaseline", "GreedyLowLevel"] and G is not None:
            def _cpu_free(n: int) -> float:
                nd = G.nodes[n]
                times = nd["_cpu_times"]
                values = nd["_cpu_values"]
                idx = bisect.bisect_right(times, t) - 1
                if idx < 0:
                    idx = 0
                return values[idx]

            if hasattr(G, "get_node_cpu_free"):
                best_node = max(valid_nodes, key=lambda n: G.get_node_cpu_free(n, t))
            else:
                best_node = max(valid_nodes, key=_cpu_free)
            return int(best_node)

        if random.random() < self.epsilon:
            if G is not None and current_source is not None and k_hop > 0:
                try:
                    khop_cache = G.graph.setdefault("_khop_cache", {})
                    key = (k_hop, current_source)
                    khop_set = khop_cache.get(key)
                    if khop_set is None:
                        khop_set = set(nx.single_source_shortest_path_length(G, current_source, cutoff=k_hop))
                        khop_cache[key] = khop_set
                    candidates = [n for n in valid_nodes if n in khop_set]
                    if candidates:
                        return int(random.choice(candidates))
                except Exception:
                    pass
            return int(random.choice(valid_nodes))

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

        w_np = pareto_w.detach().cpu().numpy()
        scores = ParetoScalarizer.scalarize_q(q_np, w_np)
        best_score = float(np.max(scores))
        tied = np.flatnonzero(scores >= best_score - 1e-12)
        best_local = int(tied[0]) if len(tied) == 1 else int(self._tie_rng.choice(tied.tolist()))
        chosen = int(valid_nodes[best_local])
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
                target_q_vecs.append(r_vec.astype(np.float32))
            else:
                cand_np = all_cand_np[st:ed]
                npw = ns[6].detach().cpu().numpy()
                cand_scores = ParetoScalarizer.scalarize_q(cand_np, npw)
                best_next = int(np.argmax(cand_scores))
                target_q_vecs.append((r_vec + self.cfg.qnet.gamma * cand_np[best_next]).astype(np.float32))

        target_t = torch.from_numpy(np.stack(target_q_vecs)).to(self.device)
        loss = nn.MSELoss()(current_q, target_t)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.qnet.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())