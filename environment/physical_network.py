import numpy as np
import networkx as nx
import torch
from typing import Dict, Tuple, List
from copy import deepcopy

class NodeResource:
    def __init__(self, node_id, cpu_cap, ram_cap, storage_cap):
        self.node_id = node_id
        self.cpu_capacity = cpu_cap
        self.ram_capacity = ram_cap
        self.storage_capacity = storage_cap
        self.cpu_residual = cpu_cap
        self.ram_residual = ram_cap
        self.storage_residual = storage_cap
        self.cpu_allocated = 0
        self.ram_allocated = 0
        self.storage_allocated = 0

    def can_allocate(self, cpu_req, ram_req, storage_req):
        return (self.cpu_residual >= cpu_req and 
                self.ram_residual >= ram_req and 
                self.storage_residual >= storage_req)

    def allocate(self, cpu_req, ram_req, storage_req):
        if not self.can_allocate(cpu_req, ram_req, storage_req):
            return False
        self.cpu_residual -= cpu_req
        self.ram_residual -= ram_req
        self.storage_residual -= storage_req
        self.cpu_allocated += cpu_req
        self.ram_allocated += ram_req
        self.storage_allocated += storage_req
        return True

    def deallocate(self, cpu_req, ram_req, storage_req):
        self.cpu_residual += cpu_req
        self.ram_residual += ram_req
        self.storage_residual += storage_req
        self.cpu_allocated -= cpu_req
        self.ram_allocated -= ram_req
        self.storage_allocated -= storage_req

    def get_residual(self):
        return {
            'cpu': self.cpu_residual,
            'ram': self.ram_residual,
            'storage': self.storage_residual,
        }

    def get_features(self):
        return np.array([
            self.cpu_residual / self.cpu_capacity,
            self.ram_residual / self.ram_capacity,
            self.storage_residual / self.storage_capacity,
            nx.degree(self.node_id) if hasattr(self, 'degree') else 0,
        ])


class LinkResource:
    def __init__(self, u, v, bw_cap, latency=1.0):
        self.u = u
        self.v = v
        self.bandwidth_capacity = bw_cap
        self.bandwidth_residual = bw_cap
        self.bandwidth_allocated = 0
        self.latency = latency

    def can_allocate(self, bw_req):
        return self.bandwidth_residual >= bw_req

    def allocate(self, bw_req):
        if not self.can_allocate(bw_req):
            return False
        self.bandwidth_residual -= bw_req
        self.bandwidth_allocated += bw_req
        return True

    def deallocate(self, bw_req):
        self.bandwidth_residual += bw_req
        self.bandwidth_allocated -= bw_req

    def get_residual_bw(self):
        return self.bandwidth_residual

    def get_latency(self):
        return self.latency


class PhysicalNetwork:
    def __init__(self, num_nodes, topology='scale_free', 
                 cpu_cap=200, ram_cap=256, storage_cap=512, bw_cap=1000):
        self.num_nodes = num_nodes
        self.topology = topology
        self.cpu_capacity = cpu_cap
        self.ram_capacity = ram_cap
        self.storage_capacity = storage_cap
        self.bw_capacity = bw_cap

        self.graph = self._create_graph()
        self.nodes = {}
        self.edges = {}
        self._initialize_resources()

    def _create_graph(self):
        if self.topology == 'scale_free':
            return nx.barabasi_albert_graph(self.num_nodes, 3)
        elif self.topology == 'random':
            return nx.erdos_renyi_graph(self.num_nodes, 0.1)
        elif self.topology == 'grid':
            side = int(np.sqrt(self.num_nodes))
            return nx.grid_2d_graph(side, side)
        else:
            return nx.complete_graph(self.num_nodes)

    def _initialize_resources(self):
        for node_id in self.graph.nodes():
            self.nodes[node_id] = NodeResource(
                node_id, self.cpu_capacity, self.ram_capacity, self.storage_capacity
            )

        for u, v in self.graph.edges():
            link = LinkResource(u, v, self.bw_capacity, latency=np.random.uniform(1, 5))
            self.edges[(u, v)] = link
            self.edges[(v, u)] = link

    def get_node_features(self):
        features = []
        for node_id in range(self.num_nodes):
            if node_id in self.nodes:
                res = self.nodes[node_id].get_residual()
                degree = self.graph.degree(node_id)
                features.append([
                    res['cpu'] / self.cpu_capacity,
                    res['ram'] / self.ram_capacity,
                    res['storage'] / self.storage_capacity,
                    degree / self.num_nodes,
                ])
            else:
                features.append([0, 0, 0, 0])
        return np.array(features, dtype=np.float32)

    def get_edge_index(self):
        edges = list(self.graph.edges())
        edge_index = [[], []]
        for u, v in edges:
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])
        return torch.LongTensor(edge_index)

    def get_adjacency_matrix(self):
        return torch.FloatTensor(nx.adjacency_matrix(self.graph).todense())

    def allocate_resource(self, node_id, cpu_req, ram_req, storage_req):
        if node_id not in self.nodes:
            return False
        return self.nodes[node_id].allocate(cpu_req, ram_req, storage_req)

    def deallocate_resource(self, node_id, cpu_req, ram_req, storage_req):
        if node_id not in self.nodes:
            return False
        self.nodes[node_id].deallocate(cpu_req, ram_req, storage_req)
        return True

    def get_node_residual_resources(self, node_id):
        if node_id not in self.nodes:
            return {'cpu': 0, 'ram': 0, 'storage': 0}
        return self.nodes[node_id].get_residual()

    def allocate_link_bw(self, u, v, bw_req):
        key = (u, v) if (u, v) in self.edges else (v, u)
        if key not in self.edges:
            return False
        return self.edges[key].allocate(bw_req)

    def deallocate_link_bw(self, u, v, bw_req):
        key = (u, v) if (u, v) in self.edges else (v, u)
        if key not in self.edges:
            return False
        self.edges[key].deallocate(bw_req)
        return True

    def get_link_residual_bw(self, u, v):
        key = (u, v) if (u, v) in self.edges else (v, u)
        if key not in self.edges:
            return 0
        return self.edges[key].get_residual_bw()

    def get_link_latency(self, u, v):
        key = (u, v) if (u, v) in self.edges else (v, u)
        if key not in self.edges:
            return float('inf')
        return self.edges[key].get_latency()

    def is_valid_path(self, path):
        if len(path) < 2:
            return True
        for i in range(len(path) - 1):
            if not self.graph.has_edge(path[i], path[i + 1]):
                return False
        return True

    def copy(self):
        return deepcopy(self)

    def reset(self):
        self._initialize_resources()