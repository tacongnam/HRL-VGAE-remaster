import random
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import torch
from config import Config
from data import get_node_features_from_graph
from utils import (
    SFCRequest,
    build_substrate_network,
    compute_load_std,
    generate_sfc_requests,
)
from utils.graph_generator import get_node_features


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
        self._edge_index_cache: Optional[np.ndarray] = None

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
        self._edge_index_cache = None
        if G is not None and requests is not None:
            self._dataset_mode = True
            self.G = G.copy()
            self.topology_id = topology_id
            self._all_requests = sorted(requests, key=lambda r: r.arrival_time)
            self._req_cursor = 0
            self._flush_arrivals_initial()
        else:
            self._dataset_mode = False
            self.G = build_substrate_network(self.cfg, self.rng)
            self.topology_id = None
            self._all_requests = []
            self.next_sfc_id = 0
            self._arrive_sfcs_random()
        return self._get_obs()

    def _flush_arrivals_initial(self):
        reqs = self._all_requests
        n_reqs = len(reqs)
        cur = self._req_cursor
        t = self.t
        while cur < n_reqs and reqs[cur].arrival_time <= t:
            r = reqs[cur]
            if r.deadline > t:
                self.queue.append(r)
            else:
                self.rejected += 1
                self.total_attempted += 1
            cur += 1
        self._req_cursor = cur

    def _flush_arrivals(self):
        reqs = self._all_requests
        n_reqs = len(reqs)
        cur = self._req_cursor
        t = self.t
        while cur < n_reqs and reqs[cur].arrival_time <= t:
            r = reqs[cur]
            if r.deadline > t:
                self.queue.append(r)
            else:
                self.rejected += 1
                self.total_attempted += 1
            cur += 1
        self._req_cursor = cur
        if self.queue:
            valid = []
            for q in self.queue:
                if q.deadline <= t:
                    self.rejected += 1
                    self.total_attempted += 1
                else:
                    valid.append(q)
            self.queue = valid

    def _arrive_sfcs_random(self):
        new_sfcs = generate_sfc_requests(self.cfg, self.t, self.next_sfc_id, self.rng)
        self.next_sfc_id += len(new_sfcs)
        self.queue.extend(new_sfcs)
        self._purge_expired_queue()

    def _purge_expired_queue(self):
        t = self.t
        valid = []
        for q in self.queue:
            if q.deadline <= t:
                self.rejected += 1
                self.total_attempted += 1
            else:
                valid.append(q)
        self.queue = valid

    def _release_expired_active_sfcs(self):
        if not self.active_embeddings:
            return
        still_active = []
        changed = False
        t = self.t
        adj = self.G._adj
        nodes = self.G.nodes
        for active in self.active_embeddings:
            if active["deadline"] <= t:
                changed = True
                for node, path, cpu_req, bw in active["allocations"]:
                    if cpu_req > 0.0:
                        nodes[node]["cpu_free"] += cpu_req
                    for i in range(len(path) - 1):
                        adj[path[i]][path[i + 1]]["bw_free"] += bw
            else:
                still_active.append(active)
        self.active_embeddings = still_active
        if changed:
            self._load_std_cache = None

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "node_features": (
                get_node_features_from_graph(self.G, self.cfg.vgae.d_in)
                if self._dataset_mode
                else get_node_features(self.G, self.cfg)
            ),
            "edge_index": self._build_edge_index(),
            "queue": self.queue,
            "t": self.t,
        }

    def _build_edge_index(self) -> np.ndarray:
        if self._edge_index_cache is not None:
            return self._edge_index_cache
        edges = list(self.G.edges())
        if not edges:
            self._edge_index_cache = np.zeros((2, 0), dtype=np.int64)
        else:
            src = [u for u, v in edges] + [v for u, v in edges]
            dst = [v for u, v in edges] + [u for u, v in edges]
            self._edge_index_cache = np.array([src, dst], dtype=np.int64)
        return self._edge_index_cache

    def get_node_features_tensor(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = self._get_obs()
        return (
            torch.from_numpy(obs["node_features"]).to(device),
            torch.from_numpy(obs["edge_index"]).to(device),
        )

    def build_ll_mask(self, cpu_req: float) -> np.ndarray:
        mask = np.zeros(self.G.number_of_nodes(), dtype=bool)
        for u, nd in self.G.nodes(data=True):
            if nd.get("is_function_node", True) and nd["cpu_free"] >= cpu_req:
                mask[u] = True
        return mask

    def allocate(self, node: int, path: List[int], cpu_req: float, bw: float):
        if cpu_req > 0.0:
            self.G.nodes[node]["cpu_free"] -= cpu_req
        adj = self.G._adj
        for i in range(len(path) - 1):
            adj[path[i]][path[i + 1]]["bw_free"] -= bw
        if len(path) > 1:
            self._load_std_cache = None

    def _rollback_resources(self, allocations):
        if not allocations:
            return
        adj = self.G._adj
        nodes = self.G.nodes
        for node, path, cpu_req, bw in allocations:
            if cpu_req > 0.0:
                nodes[node]["cpu_free"] += cpu_req
            for i in range(len(path) - 1):
                adj[path[i]][path[i + 1]]["bw_free"] += bw
        self._load_std_cache = None

    def get_load_std(self) -> float:
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
        if requeue and sfc.deadline > self.t:
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
            if self._req_cursor >= len(self._all_requests):
                unexpired = [q for q in self.queue if q.deadline > self.t]
                if unexpired:
                    next_event = (
                        min(e["deadline"] for e in self.active_embeddings)
                        if self.active_embeddings
                        else min(q.deadline for q in unexpired)
                    )
                    if next_event > self.t:
                        self.t = next_event
                        self._release_expired_active_sfcs()
                        self._purge_expired_queue()
            return self._req_cursor >= len(self._all_requests) and not any(
                q.deadline > self.t for q in self.queue
            )
        else:
            if self.t % self.cfg.sfc.arrival_interval == 0:
                self._arrive_sfcs_random()
            else:
                self._purge_expired_queue()
            return self.t >= self.cfg.train.episode_horizon

    def remove_sfc_from_queue(self, sfc: SFCRequest):
        self.queue = [q for q in self.queue if q.sfc_id != sfc.sfc_id]

    def acceptance_ratio(self) -> float:
        return (
            (self.accepted / self.total_attempted) if self.total_attempted > 0 else 0.0
        )

    @property
    def num_nodes(self) -> int:
        return self.G.number_of_nodes()