"""
Data Generation Module
Provides NetworkGenerator and SFCRequestGenerator
"""
import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Any


class VNF:
    """Virtual Network Function representation"""
    
    def __init__(self, vnf_id: int, cpu: float, ram: float, storage: float, bw: float):
        self.vnf_id = vnf_id
        self.cpu = cpu
        self.ram = ram
        self.storage = storage
        self.bw = bw
        self.cpu_req = cpu
        self.ram_req = ram
        self.storage_req = storage
        self.bandwidth = bw


class SFCRequest:
    """Service Function Chain Request"""
    
    def __init__(self, request_id: int, vnfs: List[VNF], delay_max: float = 100.0):
        self.request_id = request_id
        self.vnfs = vnfs
        self.delay_max = delay_max
        self.waiting_time = 0
        self.actual_delay = 0
        self.revenue = float(len(vnfs) * 10)
        self.link_bandwidth_reqs = [vnf.bw for vnf in vnfs]
        
    def __repr__(self):
        return f"SFCRequest(id={self.request_id}, num_vnfs={len(self.vnfs)})"


class NetworkGenerator:
    """Generate physical network topologies"""
    
    @staticmethod
    def generate_physical_network(num_nodes: int, 
                                 topology: str = 'scale_free',
                                 cpu_capacity: float = 200,
                                 ram_capacity: float = 256,
                                 storage_capacity: float = 512,
                                 bandwidth_capacity: float = 1000):
        """
        Generate a physical network with given topology
        
        Args:
            num_nodes: Number of nodes
            topology: 'scale_free', 'random', or 'grid'
            cpu_capacity: CPU capacity per node
            ram_capacity: RAM capacity per node
            storage_capacity: Storage capacity per node
            bandwidth_capacity: Bandwidth capacity per link
            
        Returns:
            NetworkX graph with node and edge attributes
        """
        
        if topology == 'scale_free':
            G = nx.barabasi_albert_graph(num_nodes, m=3)
        elif topology == 'random':
            G = nx.erdos_renyi_graph(num_nodes, p=0.3)
        elif topology == 'grid':
            side = int(np.sqrt(num_nodes))
            G = nx.grid_2d_graph(side, side)
            # Convert tuple nodes to integers
            G = nx.convert_node_labels_to_integers(G)
        else:
            raise ValueError(f"Unknown topology: {topology}")
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['cpu'] = cpu_capacity
            G.nodes[node]['ram'] = ram_capacity
            G.nodes[node]['storage'] = storage_capacity
            G.nodes[node]['cpu_residual'] = cpu_capacity
            G.nodes[node]['ram_residual'] = ram_capacity
            G.nodes[node]['storage_residual'] = storage_capacity
            G.nodes[node]['load'] = {}
        
        # Add edge attributes
        for u, v in G.edges():
            G.edges[u, v]['bandwidth'] = bandwidth_capacity
            G.edges[u, v]['bandwidth_residual'] = bandwidth_capacity
        
        # Add helper methods
        G.get_node_features = lambda: NetworkGenerator._get_node_features(G)
        G.get_edge_index = lambda: NetworkGenerator._get_edge_index(G)
        G.get_node_residual_resources = lambda node_id: {
            'cpu': G.nodes[node_id].get('cpu_residual', cpu_capacity),
            'ram': G.nodes[node_id].get('ram_residual', ram_capacity),
            'storage': G.nodes[node_id].get('storage_residual', storage_capacity),
        }
        G.allocate_resource = lambda node, cpu, ram, storage: NetworkGenerator._allocate_resource(G, node, cpu, ram, storage)
        G.deallocate_resource = lambda node, cpu, ram, storage: NetworkGenerator._deallocate_resource(G, node, cpu, ram, storage)
        
        return G
    
    @staticmethod
    def _get_node_features(G) -> np.ndarray:
        """Get node feature matrix (normalized residual resources)"""
        num_nodes = G.number_of_nodes()
        features = []
        
        for node in range(num_nodes):
            node_data = G.nodes[node]
            cpu_norm = node_data.get('cpu_residual', 200) / 200.0
            ram_norm = node_data.get('ram_residual', 256) / 256.0
            storage_norm = node_data.get('storage_residual', 512) / 512.0
            degree = G.degree(node) / num_nodes
            
            features.append([cpu_norm, ram_norm, storage_norm, degree])
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def _get_edge_index(G) -> Tuple[np.ndarray, np.ndarray]:
        """Get edge index in COO format for GNN"""
        edges = list(G.edges())
        if len(edges) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        
        src, dst = zip(*edges)
        src = np.array(src, dtype=np.int64)
        dst = np.array(dst, dtype=np.int64)
        
        # Make undirected
        src_undirected = np.concatenate([src, dst])
        dst_undirected = np.concatenate([dst, src])
        
        return src_undirected, dst_undirected
    
    @staticmethod
    def _allocate_resource(G, node_id: int, cpu: float, ram: float, storage: float) -> bool:
        """Allocate resources from a node"""
        node = G.nodes[node_id]
        
        if (node.get('cpu_residual', 0) >= cpu and
            node.get('ram_residual', 0) >= ram and
            node.get('storage_residual', 0) >= storage):
            
            node['cpu_residual'] = node.get('cpu_residual', 200) - cpu
            node['ram_residual'] = node.get('ram_residual', 256) - ram
            node['storage_residual'] = node.get('storage_residual', 512) - storage
            return True
        
        return False
    
    @staticmethod
    def _deallocate_resource(G, node_id: int, cpu: float, ram: float, storage: float):
        """Deallocate resources from a node"""
        node = G.nodes[node_id]
        node['cpu_residual'] = node.get('cpu_residual', 0) + cpu
        node['ram_residual'] = node.get('ram_residual', 0) + ram
        node['storage_residual'] = node.get('storage_residual', 0) + storage


class SFCRequestGenerator:
    """Generate SFC requests"""
    
    def __init__(self, 
                 cpu_range: Tuple[float, float] = (10, 50),
                 ram_range: Tuple[float, float] = (16, 128),
                 storage_range: Tuple[float, float] = (32, 256),
                 bw_range: Tuple[float, float] = (10, 100),
                 vnf_count_range: Tuple[int, int] = (3, 8)):
        """
        Initialize SFC request generator
        
        Args:
            cpu_range: (min, max) CPU requirement per VNF
            ram_range: (min, max) RAM requirement per VNF
            storage_range: (min, max) Storage requirement per VNF
            bw_range: (min, max) Bandwidth requirement per link
            vnf_count_range: (min, max) number of VNFs in SFC
        """
        self.cpu_range = cpu_range
        self.ram_range = ram_range
        self.storage_range = storage_range
        self.bw_range = bw_range
        self.vnf_count_range = vnf_count_range
        self.request_counter = 0
    
    def generate(self, num_requests: int) -> List[SFCRequest]:
        """Generate multiple SFC requests"""
        requests = []
        
        for _ in range(num_requests):
            request = self._generate_single()
            requests.append(request)
        
        return requests
    
    def _generate_single(self) -> SFCRequest:
        """Generate a single SFC request"""
        self.request_counter += 1
        
        # Generate number of VNFs
        num_vnfs = np.random.randint(self.vnf_count_range[0], self.vnf_count_range[1] + 1)
        
        # Generate VNFs
        vnfs = []
        for i in range(num_vnfs):
            cpu = np.random.uniform(self.cpu_range[0], self.cpu_range[1])
            ram = np.random.uniform(self.ram_range[0], self.ram_range[1])
            storage = np.random.uniform(self.storage_range[0], self.storage_range[1])
            bw = np.random.uniform(self.bw_range[0], self.bw_range[1])
            
            vnf = VNF(i, cpu, ram, storage, bw)
            vnfs.append(vnf)
        
        # Create SFC request
        request = SFCRequest(self.request_counter, vnfs, delay_max=100.0)
        
        return request
