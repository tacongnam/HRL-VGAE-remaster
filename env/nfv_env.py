import heapq
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
        # P0: operation-level release heap
        self._op_heap: list = []   # (finish_time, counter, spec)
        self._op_counter: int = 0

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
        # P0: reset operation heap
        self._op_heap = []
        self._op_counter = 0
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
        new_sfcs = generate_sfc_requests(self.cfg, self.t, self.next_sfc_id, self.G, self.rng)
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

    # ------------------------------------------------------------------
    # P0: schedule per-operation resource releases into heap
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_alloc(item: Tuple):
        node, path, cpu_req, bw = item[0], item[1], item[2], item[3]
        ram_req = item[4] if len(item) > 4 else 0.0
        hdd_req = item[5] if len(item) > 5 else 0.0
        return node, path, cpu_req, bw, ram_req, hdd_req

    def _schedule_operation_releases(
        self, allocations: List[Tuple], t_start: float, sfc: Optional[SFCRequest] = None
    ):
        """Push per-operation finish events onto _op_heap.

        For each (node, path, cpu_req, bw) in allocations:
          1. RoutingOperation: finish = t_current + sum(link delays on path)
             → schedule BW release for those links
          2. ServingOperation (if cpu_req > 0): finish = t_after_routing
             + proc_delay[node] + VNF d_f[node]
             → schedule CPU/RAM/HDD release for that node
        """
        t = float(t_start)
        adj = self.G._adj
        nodes_data = self.G.nodes
        vnf_iter = iter(sfc.vnf_sequence) if sfc is not None else iter(())
        for item in allocations:
            node, path, cpu_req, bw, ram_req, hdd_req = self._unpack_alloc(item)
            if len(path) > 1:
                link_delay = sum(
                    adj[path[i]][path[i + 1]].get("delay", 0.0)
                    for i in range(len(path) - 1)
                )
                finish_route = t + link_delay
                heapq.heappush(
                    self._op_heap,
                    (finish_route, self._op_counter, ("link", tuple(path), bw)),
                )
                self._op_counter += 1
                t = finish_route
            if cpu_req > 0.0:
                proc = nodes_data[node].get("proc_delay", 0.0)
                vnf_d = 0.0
                try:
                    vnf_d = next(vnf_iter).delay_at(node)
                except StopIteration:
                    pass
                finish_serve = t + proc + vnf_d
                heapq.heappush(
                    self._op_heap,
                    (
                        finish_serve,
                        self._op_counter,
                        ("cpu", node, cpu_req, ram_req, hdd_req),
                    ),
                )
                self._op_counter += 1
                t = finish_serve

    def compute_embedding_delay(
        self, allocations: List[Tuple], sfc: Optional[SFCRequest] = None
    ) -> float:
        """Link delays + node proc_delay + VNF d_f along the embedding."""
        total = 0.0
        adj = self.G._adj
        nodes_data = self.G.nodes
        vnf_iter = iter(sfc.vnf_sequence) if sfc is not None else iter(())
        for item in allocations:
            node, path, cpu_req, _, _, _ = self._unpack_alloc(item)
            for i in range(len(path) - 1):
                total += adj[path[i]][path[i + 1]].get("delay", 0.0)
            if cpu_req > 0.0:
                total += nodes_data[node].get("proc_delay", 0.0)
                try:
                    total += next(vnf_iter).delay_at(node)
                except StopIteration:
                    pass
        return total

    def _release_expired_active_sfcs(self):
        """Release resources for completed operations (P0: per-operation heap).

        Also prune active_embeddings list for SFCs past their deadline.
        """
        t_float = float(self.t)
        adj = self.G._adj
        nodes_data = self.G.nodes
        changed = False
        while self._op_heap and self._op_heap[0][0] <= t_float:
            _, _, spec = heapq.heappop(self._op_heap)
            op_type = spec[0]
            if op_type == "link":
                _, path, bw = spec
                for i in range(len(path) - 1):
                    adj[path[i]][path[i + 1]]["bw_free"] += bw
                changed = True
            elif op_type == "cpu":
                node = spec[1]
                cpu_req = spec[2]
                ram_req = spec[3] if len(spec) > 3 else 0.0
                hdd_req = spec[4] if len(spec) > 4 else 0.0
                nd = nodes_data[node]
                nd["cpu_free"] += cpu_req
                if ram_req and "ram_free" in nd:
                    nd["ram_free"] += ram_req
                if hdd_req and "hdd_free" in nd:
                    nd["hdd_free"] += hdd_req
                changed = True
        # Keep active_embeddings for lifecycle tracking (deadline-based)
        self.active_embeddings = [
            a for a in self.active_embeddings if a["deadline"] > self.t
        ]
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

    def build_ll_mask(
        self, cpu_req: float, ram_req: float = 0.0, hdd_req: float = 0.0
    ) -> np.ndarray:
        mask = np.zeros(self.G.number_of_nodes(), dtype=bool)
        for u, nd in self.G.nodes(data=True):
            if not nd.get("is_function_node", True):
                continue
            if nd["cpu_free"] < cpu_req:
                continue
            if ram_req > 0.0 and nd.get("ram_free", float("inf")) < ram_req:
                continue
            if hdd_req > 0.0 and nd.get("hdd_free", float("inf")) < hdd_req:
                continue
            mask[u] = True
        return mask

    def allocate(
        self,
        node: int,
        path: List[int],
        cpu_req: float,
        bw: float,
        ram_req: float = 0.0,
        hdd_req: float = 0.0,
    ):
        nd = self.G.nodes[node]
        if cpu_req > 0.0:
            nd["cpu_free"] -= cpu_req
            if ram_req and "ram_free" in nd:
                nd["ram_free"] -= ram_req
            if hdd_req and "hdd_free" in nd:
                nd["hdd_free"] -= hdd_req
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
        for item in allocations:
            node, path, cpu_req, bw, ram_req, hdd_req = self._unpack_alloc(item)
            if cpu_req > 0.0:
                nd = nodes[node]
                nd["cpu_free"] += cpu_req
                if ram_req and "ram_free" in nd:
                    nd["ram_free"] += ram_req
                if hdd_req and "hdd_free" in nd:
                    nd["hdd_free"] += hdd_req
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
        # P0: schedule per-operation resource releases
        self._schedule_operation_releases(allocations, float(self.t), sfc=sfc)

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
