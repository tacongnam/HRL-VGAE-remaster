import heapq
import math
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

_MAX_SFC_RETRIES = 3


class NFVEnvironment:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        self.G: nx.Graph = None
        self._all_requests: List[SFCRequest] = []
        self.queue: List[SFCRequest] = []
        self.active_embeddings: List[Dict[str, Any]] = []
        self.t: float = 0.0
        self.next_sfc_id: int = 0
        self.accepted: int = 0
        self.rejected: int = 0
        self.total_attempted: int = 0
        self._dataset_mode: bool = False
        self._req_cursor: int = 0
        self.topology_id: Optional[str] = None
        self.total_deploy_cost: float = 0.0
        self.total_node_placement_cost: float = 0.0
        self._load_std_cache: Optional[float] = None
        self._edge_index_cache: Optional[np.ndarray] = None
        self._op_heap: list = []
        self._op_counter: int = 0
        self._sfc_retry_count: Dict[int, int] = {}

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
        self.total_node_placement_cost = 0.0
        self.t = 0.0
        self.queue = []
        self.active_embeddings = []
        self._load_std_cache = None
        self._edge_index_cache = None
        self._op_heap = []
        self._op_counter = 0
        self._sfc_retry_count = {}
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
        new_sfcs = generate_sfc_requests(self.cfg, int(self.t), self.next_sfc_id, self.G, self.rng)
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

    @staticmethod
    def _unpack_alloc(item: Tuple):
        node, path, cpu_req, bw = item[0], item[1], item[2], item[3]
        ram_req = item[4] if len(item) > 4 else 0.0
        hdd_req = item[5] if len(item) > 5 else 0.0
        return node, path, cpu_req, bw, ram_req, hdd_req

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
            self._res_set(
                nd["_cpu_times"],
                nd["_cpu_values"],
                float(self.t),
                float(self.t) + 1.0,
                nd.get("_cpu_values", [float(nd["cpu_total"])])[0] - cpu_req,
            )
            if ram_req and "_ram_times" in nd:
                cur_ram = self._res_query(nd["_ram_times"], nd["_ram_values"], float(self.t))
                self._res_set(
                    nd["_ram_times"],
                    nd["_ram_values"],
                    float(self.t),
                    float(self.t) + 1.0,
                    cur_ram - ram_req,
                )
            if hdd_req and "_hdd_times" in nd:
                cur_hdd = self._res_query(nd["_hdd_times"], nd["_hdd_values"], float(self.t))
                self._res_set(
                    nd["_hdd_times"],
                    nd["_hdd_values"],
                    float(self.t),
                    float(self.t) + 1.0,
                    cur_hdd - hdd_req,
                )
        adj = self.G._adj
        for i in range(len(path) - 1):
            self._res_set(
                adj[path[i]][path[i + 1]]["_bw_times"],
                adj[path[i]][path[i + 1]]["_bw_values"],
                float(self.t),
                float(self.t) + 1.0,
                float(adj[path[i]][path[i + 1]]["_bw_values"][0]) - bw,
            )
        if len(path) > 1:
            self._load_std_cache = None
        self._edge_index_cache = None

    def _schedule_operation_releases(
        self, allocations: List[Tuple], t_start: float, sfc: Optional[SFCRequest] = None
    ):
        """Schedule resource release after the entire SFC pipeline completes.

        BW on all links is held until the full SFC serving pipeline finishes
        (i.e., after the last CPU operation), matching greedy-KSP semantics
        where bandwidth is reserved for the duration [t_start, t_end] of the SFC.
        CPU on each node is released after that node's serving phase ends.
        """
        adj = self.G._adj
        nodes_data = self.G.nodes
        vnf_iter = iter(sfc.vnf_sequence) if sfc is not None else iter(())

        t = float(t_start)
        cpu_events = []
        link_paths_bw = []

        for item in allocations:
            node, path, cpu_req, bw, ram_req, hdd_req = self._unpack_alloc(item)
            if len(path) > 1:
                link_delay = sum(
                    adj[path[i]][path[i + 1]].get("delay", 0.0)
                    for i in range(len(path) - 1)
                )
                t += link_delay
                link_paths_bw.append((tuple(path), bw))
            if cpu_req > 0.0:
                proc = nodes_data[node].get("proc_delay", 0.0)
                vnf_d = 0.0
                try:
                    vnf_d = next(vnf_iter).delay_at(node)
                except StopIteration:
                    pass
                finish_serve = t + proc + vnf_d
                cpu_events.append((finish_serve, node, cpu_req, ram_req, hdd_req))
                t = finish_serve

        pipeline_end = t

        for path_tup, bw in link_paths_bw:
            ctr = self._op_counter
            heapq.heappush(
                self._op_heap,
                (pipeline_end, ctr, ("link", path_tup, bw)),
            )
            self._op_counter += 1

        for finish_serve, node, cpu_req, ram_req, hdd_req in cpu_events:
            ctr = self._op_counter
            heapq.heappush(
                self._op_heap,
                (finish_serve, ctr, ("cpu", node, cpu_req, ram_req, hdd_req)),
            )
            self._op_counter += 1

    def compute_embedding_delay(
        self, allocations: List[Tuple], sfc: Optional[SFCRequest] = None
    ) -> float:
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
                    u, v = path[i], path[i + 1]
                    e = adj[u][v]
                    self._release_bw_timeline(
                        e["_bw_times"], e["_bw_values"], t_float, t_float + 1.0, bw
                    )
                changed = True
            elif op_type == "cpu":
                node = spec[1]
                cpu_req = spec[2]
                ram_req = spec[3] if len(spec) > 3 else 0.0
                hdd_req = spec[4] if len(spec) > 4 else 0.0
                nd = nodes_data[node]
                res = self._res_range(nd["_cpu_times"], t_float, t_float + 1.0)
                self._release_resource_timeline(
                    nd["_cpu_times"], nd["_cpu_values"], res[0], res[1], cpu_req
                )
                if ram_req and "_ram_times" in nd:
                    res = self._res_range(nd["_ram_times"], t_float, t_float + 1.0)
                    self._release_resource_timeline(
                        nd["_ram_times"], nd["_ram_values"], res[0], res[1], ram_req
                    )
                if hdd_req and "_hdd_times" in nd:
                    res = self._res_range(nd["_hdd_times"], t_float, t_float + 1.0)
                    self._release_resource_timeline(
                        nd["_hdd_times"], nd["_hdd_values"], res[0], res[1], hdd_req
                    )
                changed = True
        self.active_embeddings = [
            a for a in self.active_embeddings if a["deadline"] > self.t
        ]
        if changed:
            self._load_std_cache = None
            self._edge_index_cache = None

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

    def get_node_cpu_free(self, node: int, t: float) -> float:
        nd = self.G.nodes[node]
        return NFVEnvironment._res_query(nd["_cpu_times"], nd["_cpu_values"], float(t))

    def get_node_ram_free(self, node: int, t: float) -> float:
        nd = self.G.nodes[node]
        return NFVEnvironment._res_query(nd["_ram_times"], nd["_ram_values"], float(t))

    def get_node_hdd_free(self, node: int, t: float) -> float:
        nd = self.G.nodes[node]
        return NFVEnvironment._res_query(nd["_hdd_times"], nd["_hdd_values"], float(t))

    def build_ll_mask(
        self, cpu_req: float, ram_req: float = 0.0, hdd_req: float = 0.0
    ) -> np.ndarray:
        mask = np.zeros(self.G.number_of_nodes(), dtype=bool)
        t = self.t
        for u, nd in self.G.nodes(data=True):
            if not nd.get("is_function_node", False):
                continue
            if self.get_node_cpu_free(u, t) < cpu_req:
                continue
            if ram_req > 0.0 and "_ram_times" in nd:
                if self._res_query(nd["_ram_times"], nd["_ram_values"], t) < ram_req:
                    continue
            if hdd_req > 0.0 and "_hdd_times" in nd:
                if self._res_query(nd["_hdd_times"], nd["_hdd_values"], t) < hdd_req:
                    continue
            mask[u] = True
        return mask

    def _rollback_resources(self, allocations):
        if not allocations:
            return
        adj = self.G._adj
        nodes = self.G.nodes
        for item in allocations:
            node, path, cpu_req, bw, ram_req, hdd_req = self._unpack_alloc(item)
            if cpu_req > 0.0:
                nd = nodes[node]
                res = self._res_range(nd["_cpu_times"], float(self.t), float(self.t) + 1.0)
                self._release_resource_timeline(nd["_cpu_times"], nd["_cpu_values"], res[0], res[1], cpu_req)
                if ram_req and "_ram_times" in nd:
                    res = self._res_range(nd["_ram_times"], float(self.t), float(self.t) + 1.0)
                    self._release_resource_timeline(nd["_ram_times"], nd["_ram_values"], res[0], res[1], ram_req)
                if hdd_req and "_hdd_times" in nd:
                    res = self._res_range(nd["_hdd_times"], float(self.t), float(self.t) + 1.0)
                    self._release_resource_timeline(nd["_hdd_times"], nd["_hdd_values"], res[0], res[1], hdd_req)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = adj[u][v]
                self._release_bw_timeline(e["_bw_times"], e["_bw_values"], float(self.t), float(self.t) + 1.0, bw)
        self._load_std_cache = None
        self._edge_index_cache = None

    def get_load_std(self) -> float:
        if self._load_std_cache is None:
            self._load_std_cache = compute_load_std(self.G)
        return self._load_std_cache

    def _node_placement_cost(self, allocations: List[Tuple]) -> float:
        total = 0.0
        nodes_data = self.G.nodes
        for item in allocations:
            node, _, cpu_req, _, ram_req, hdd_req = self._unpack_alloc(item)
            if cpu_req > 0.0:
                nd = nodes_data[node]
                total += (
                    nd.get("cost_cpu", 1.0) * cpu_req
                    + nd.get("cost_ram", 1.0) * ram_req
                    + nd.get("cost_stor", 1.0) * hdd_req
                )
        return total

    def commit_sfc(
        self, sfc: SFCRequest, allocations: List[Tuple], deploy_cost: float = 0.0
    ):
        self._schedule_operation_releases(allocations, float(self.t), sfc=sfc)
        self.active_embeddings.append(
            {"sfc_id": sfc.sfc_id, "deadline": sfc.deadline}
        )
        self.accepted += 1
        self.total_attempted += 1
        self.total_deploy_cost += deploy_cost
        self.total_node_placement_cost += self._node_placement_cost(allocations)
        self._sfc_retry_count.pop(sfc.sfc_id, None)

    def reject_sfc(
        self, sfc: SFCRequest, allocations: List[Tuple], requeue: bool = True
    ):
        self._rollback_resources(allocations)
        if requeue and sfc.deadline > self.t:
            retries = self._sfc_retry_count.get(sfc.sfc_id, 0) + 1
            if retries <= _MAX_SFC_RETRIES:
                self._sfc_retry_count[sfc.sfc_id] = retries
                if not any(q.sfc_id == sfc.sfc_id for q in self.queue):
                    self.queue.append(sfc)
            else:
                self._sfc_retry_count.pop(sfc.sfc_id, None)
                self.rejected += 1
                self.total_attempted += 1
        else:
            self.rejected += 1
            self.total_attempted += 1

    def norm_revenue(self, sfc: SFCRequest) -> float:
        rc = self.cfg.reward
        return (
            rc.mu_cpu * sfc.total_cpu_req() + rc.mu_bw * sfc.bandwidth * (sfc.F_k + 1)
        ) / max(1, sfc.F_k)

    def _next_event_time(self) -> Optional[float]:
        times = []
        if self._op_heap:
            times.append(self._op_heap[0][0])
        if self._dataset_mode and self._req_cursor < len(self._all_requests):
            times.append(float(self._all_requests[self._req_cursor].arrival_time))
        return min(times) if times else None

    def step_time(self) -> bool:
        if self._dataset_mode:
            next_int_t = float(math.floor(self.t) + 1)
            if self._op_heap and self._op_heap[0][0] < next_int_t:
                self.t = self._op_heap[0][0]
                self._release_expired_active_sfcs()
                return False

            self.t = next_int_t
            self._release_expired_active_sfcs()
            self._flush_arrivals()

            if self._req_cursor >= len(self._all_requests):
                unexpired = [q for q in self.queue if q.deadline > self.t]
                if unexpired:
                    candidates = []
                    if self._op_heap:
                        candidates.append(self._op_heap[0][0])
                    if self.active_embeddings:
                        candidates.append(float(min(e["deadline"] for e in self.active_embeddings)))
                    if candidates:
                        next_t = min(candidates)
                        if next_t > self.t:
                            self.t = next_t
                            self._release_expired_active_sfcs()
                            self._purge_expired_queue()
            return self._req_cursor >= len(self._all_requests) and not any(
                q.deadline > self.t for q in self.queue
            )
        else:
            self.t = float(math.floor(self.t) + 1)
            self._release_expired_active_sfcs()
            if int(self.t) % self.cfg.sfc.arrival_interval == 0:
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