import networkx as nx
from typing import List


class ConstraintChecker:

    @staticmethod
    def check_resource_feasible(p_net, server_id, vnf):
        residual = p_net.get_node_residual_resources(server_id)
        return (residual['cpu'] >= vnf.cpu_req and
                residual['ram'] >= vnf.ram_req and
                residual['storage'] >= vnf.storage_req)

    @staticmethod
    def check_bandwidth_feasible(p_net, path, bw_req):
        if path is None or len(path) < 2:
            return True
        for i in range(len(path) - 1):
            if p_net.get_link_residual_bw(path[i], path[i + 1]) < bw_req:
                return False
        return True

    @staticmethod
    def check_latency_feasible(p_net, path, latency_req):
        if path is None or len(path) < 2:
            return True
        total_latency = sum(p_net.get_link_latency(path[i], path[i + 1])
                           for i in range(len(path) - 1))
        return total_latency <= latency_req

    @staticmethod
    def check_path_validity(p_net, path):
        if path is None or len(path) < 2:
            return True
        return p_net.is_valid_path(path)