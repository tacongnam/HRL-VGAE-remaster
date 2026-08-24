import networkx as nx
from typing import List, Optional


class RoutingEngine:

    @staticmethod
    def find_feasible_path(p_net, src, dst, bw_required, latency_limit=None):
        if src == dst:
            return [src]

        feasible_graph = p_net.graph.copy()

        edges_to_remove = []
        for u, v in feasible_graph.edges():
            if p_net.get_link_residual_bw(u, v) < bw_required:
                edges_to_remove.append((u, v))

        for u, v in edges_to_remove:
            feasible_graph.remove_edge(u, v)

        try:
            path = nx.shortest_path(feasible_graph, src, dst)

            if latency_limit is not None:
                latency = RoutingEngine.calculate_path_latency(p_net, path)
                if latency > latency_limit:
                    return None

            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    @staticmethod
    def allocate_bandwidth(p_net, path, bw_required):
        if path is None or len(path) < 2:
            return True

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not p_net.allocate_link_bw(u, v, bw_required):
                for j in range(i):
                    p_net.deallocate_link_bw(path[j], path[j + 1], bw_required)
                return False
        return True

    @staticmethod
    def deallocate_bandwidth(p_net, path, bw_required):
        if path is None or len(path) < 2:
            return True

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            p_net.deallocate_link_bw(u, v, bw_required)
        return True

    @staticmethod
    def calculate_path_latency(p_net, path):
        if path is None or len(path) < 2:
            return 0

        total_latency = 0
        for i in range(len(path) - 1):
            total_latency += p_net.get_link_latency(path[i], path[i + 1])
        return total_latency

    @staticmethod
    def calculate_path_cost(p_net, path):
        if path is None or len(path) < 2:
            return 0

        return len(path) - 1