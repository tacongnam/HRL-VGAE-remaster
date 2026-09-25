import random
import numpy as np
import torch
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
from config import Config
from utils.graph_generator import (
    SFCRequest,
    build_substrate_network,
    generate_sfc_requests,
)


class NFVEnvironment:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        self.G: nx.Graph = None
        self._all_requests: List[SFCRequest] = []
        self.queue: List[SFCRequest] = []
        self.active_embeddings: List[Dict[str, Any]] = []
        self.t: int = 0
        self.next_sfc_id: int = 0
        self.accepted: int = 0
        self.rejected: int = 0
        self.total_attempted: int = 0
        self._dataset_mode: bool = False
        self._req_cursor: int = 0
        self.topology_id: Optional[str] = None
        self.total_deploy_cost: float = 0.0
        self._load_std_cache: Optional[float] = None

    def reset(
        self,
        G: nx.Graph = None,
        requests: List[SFCRequest] = None,
        topology_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.accepted = 0
        self.rejected = 0
        self.total_attempted = 0
        self.total_deploy_cost = 0.0
        self.t = 0
        self.queue = []
        self.active_embeddings = []
        self._load_std_cache = None
        if G is not None and requests is not None:
            self._dataset_mode = True
            self.G = G.copy()
            self.topology_id = topology_id
            self._all_requests = sorted(requests, key=lambda r: r.arrival_time)
            self._req_cursor = 0
            self._flush_arrivals()
        else:
            self._dataset_mode = False
            self.G = build_substrate_network(self.cfg, self.rng)
            self.topology_id = None
            self._all_requests = []
            self.next_sfc_id = 0
            self._arrive_sfcs_random()
        return self._get_obs()

    def _flush_arrivals(self):
        while (
            self._req_cursor < len(self._all_requests)
            and self._all_requests[self._req_cursor].arrival_time <= self.t
        ):
            r = self._all_requests[self._req_cursor]
            if not r.is_expired(self.t):
                self.queue.append(r)
            else:
                self.rejected += 1
                self.total_attempted += 1
            self._req_cursor += 1
        self._purge_expired_queue()

    def _arrive_sfcs_random(self):
        new_sfcs = generate_sfc_requests(self.cfg, self.t, self.next_sfc_id, self.rng)
        self.next_sfc_id += len(new_sfcs)
        self.queue.extend(new_sfcs)
        self._purge_expired_queue()

    def _purge_expired_queue(self):
        valid = []
        for q in self.queue:
            if q.is_expired(self.t):
                self.rejected += 1
                self.total_attempted += 1
            else:
                valid.append(q)
        self.queue = valid

    def _release_expired_active_sfcs(self):
        still_active = []
        changed = False
        for active in self.active_embeddings:
            if active["deadline"] <= self.t:
                self._rollback_resources(active["allocations"])
                changed = True
            else:
                still_active.append(active)
        self.active_embeddings = still_active
        if changed:
            self._load_std_cache = None

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "node_features": self._compute_node_features(),
            "edge_index": self._build_edge_index(),
            "queue": self.queue,
            "t": self.t,
        }

    def _compute_node_features(self) -> np.ndarray:
        from data.loader import get_node_features_from_graph

        if self._dataset_mode:
            return get_node_features_from_graph(self.G, self.cfg.vgae.d_in)
        from utils.graph_generator import get_node_features

        return get_node_features(self.G, self.cfg)

    def _build_edge_index(self) -> np.ndarray:
        edges = list(self.G.edges())
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        src = [u for u, v in edges] + [v for u, v in edges]
        dst = [v for u, v in edges] + [u for u, v in edges]
        return np.array([src, dst], dtype=np.int64)

    def get_node_features_tensor(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = self._get_obs()
        x = torch.tensor(obs["node_features"], dtype=torch.float32, device=device)
        ei = torch.tensor(obs["edge_index"], dtype=torch.long, device=device)
        return x, ei

    def build_ll_mask(self, cpu_req: float) -> np.ndarray:
        num_nodes = self.G.number_of_nodes()
        mask = np.zeros(num_nodes, dtype=bool)
        for u in self.G.nodes():
            nd = self.G.nodes[u]
            if nd.get("is_function_node", True) and nd["cpu_free"] >= cpu_req:
                mask[u] = True
        return mask

    def allocate(self, node: int, path: List[int], cpu_req: float, bw: float):
        if cpu_req > 0.0:
            self.G.nodes[node]["cpu_free"] -= cpu_req
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.G[u][v]["bw_free"] -= bw
        if len(path) > 1:
            self._load_std_cache = None

    def _rollback_resources(self, allocations):
        for node, path, cpu_req, bw in allocations:
            if cpu_req > 0.0:
                self.G.nodes[node]["cpu_free"] += cpu_req
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.G[u][v]["bw_free"] += bw
        if allocations:
            self._load_std_cache = None

    def get_load_std(self) -> float:
        from utils.dijkstra import compute_load_std

        if self._load_std_cache is None:
            self._load_std_cache = compute_load_std(self.G)
        return self._load_std_cache

    def commit_sfc(
        self, sfc: SFCRequest, allocations: List[Tuple], deploy_cost: float = 0.0
    ):
        self.active_embeddings.append(
            {"sfc_id": sfc.sfc_id, "deadline": sfc.deadline, "allocations": allocations}
        )
        self.accepted += 1
        self.total_attempted += 1
        self.total_deploy_cost += deploy_cost

    def reject_sfc(
        self, sfc: SFCRequest, allocations: List[Tuple], requeue: bool = True
    ):
        self._rollback_resources(allocations)
        if requeue and not sfc.is_expired(self.t):
            if not any(q.sfc_id == sfc.sfc_id for q in self.queue):
                self.queue.append(sfc)
        else:
            self.rejected += 1
            self.total_attempted += 1

    def norm_revenue(self, sfc: SFCRequest) -> float:
        rc = self.cfg.reward
        return (
            rc.mu_cpu * sfc.total_cpu_req() + rc.mu_bw * sfc.bandwidth * (sfc.F_k + 1)
        ) / max(1, sfc.F_k)

    def step_time(self) -> bool:
        self.t += 1
        self._release_expired_active_sfcs()
        if self._dataset_mode:
            self._flush_arrivals()
            done = self._req_cursor >= len(self._all_requests) and not self.queue
        else:
            if self.t % self.cfg.sfc.arrival_interval == 0:
                self._arrive_sfcs_random()
            else:
                self._purge_expired_queue()
            done = self.t >= self.cfg.train.episode_horizon
        return done

    def remove_sfc_from_queue(self, sfc: SFCRequest):
        self.queue = [q for q in self.queue if q.sfc_id != sfc.sfc_id]

    def acceptance_ratio(self) -> float:
        return (
            (self.accepted / self.total_attempted) if self.total_attempted > 0 else 0.0
        )

    @property
    def num_nodes(self) -> int:
        return self.G.number_of_nodes()
