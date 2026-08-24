import numpy as np
import torch
import networkx as nx
from typing import Tuple, List


class GraphUtils:

    @staticmethod
    def adjacency_to_edge_index(adj_matrix) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.FloatTensor(adj_matrix)

        edge_index = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()
        edge_attr = adj_matrix[adj_matrix > 0]

        return edge_index, edge_attr

    @staticmethod
    def edge_index_to_adjacency(edge_index, num_nodes, edge_attr=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu().numpy()

        adj_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            if edge_attr is not None:
                adj_matrix[u, v] = edge_attr[i].item() if isinstance(edge_attr[i], torch.Tensor) else edge_attr[i]
            else:
                adj_matrix[u, v] = 1

        return torch.FloatTensor(adj_matrix)

    @staticmethod
    def get_degree_sequence(adj_matrix):
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()

        degree = np.sum(adj_matrix, axis=1)
        return degree

    @staticmethod
    def get_laplacian(adj_matrix):
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()

        degree = np.diag(np.sum(adj_matrix, axis=1))
        laplacian = degree - adj_matrix

        return torch.FloatTensor(laplacian)

    @staticmethod
    def normalize_adjacency(adj_matrix):
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()

        degree = np.sum(adj_matrix, axis=1)
        degree = np.power(degree, -0.5)
        degree[np.isinf(degree)] = 0
        degree_matrix = np.diag(degree)

        normalized = degree_matrix @ adj_matrix @ degree_matrix

        return torch.FloatTensor(normalized)

    @staticmethod
    def k_hop_neighbors(adj_matrix, node_id, k):
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()

        neighbors = set([node_id])
        current_neighbors = set([node_id])

        for _ in range(k):
            next_neighbors = set()
            for node in current_neighbors:
                neighbors_of_node = np.where(adj_matrix[node] > 0)[0]
                next_neighbors.update(neighbors_of_node)

            current_neighbors = next_neighbors - neighbors
            neighbors.update(current_neighbors)

        neighbors.discard(node_id)
        return list(neighbors)

    @staticmethod
    def shortest_path_distance(adj_matrix, src, dst):
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()

        num_nodes = adj_matrix.shape[0]
        dist = [float('inf')] * num_nodes
        dist[src] = 0
        visited = [False] * num_nodes

        for _ in range(num_nodes):
            u = -1
            for v in range(num_nodes):
                if not visited[v] and (u == -1 or dist[v] < dist[u]):
                    u = v

            if dist[u] == float('inf'):
                break

            visited[u] = True

            for v in range(num_nodes):
                if adj_matrix[u][v] > 0 and dist[u] + adj_matrix[u][v] < dist[v]:
                    dist[v] = dist[u] + adj_matrix[u][v]

        return dist[dst] if dist[dst] != float('inf') else -1

    @staticmethod
    def get_strongly_connected_components(adj_matrix):
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()

        G = nx.DiGraph()
        num_nodes = adj_matrix.shape[0]

        for i in range(num_nodes):
            G.add_node(i)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i][j] > 0:
                    G.add_edge(i, j)

        sccs = list(nx.strongly_connected_components(G))
        return sccs

    @staticmethod
    def subgraph(adj_matrix, nodes):
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()

        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        sub_adj = np.zeros((len(nodes), len(nodes)))

        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                sub_adj[i][j] = adj_matrix[u][v]

        return torch.FloatTensor(sub_adj)