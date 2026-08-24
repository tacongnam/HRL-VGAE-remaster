import numpy as np
from ..environment.constraints import ConstraintChecker
from .routing import RoutingEngine


class CandidateMasking:

    @staticmethod
    def get_resource_feasible_candidates(p_net, vnf):
        candidates = np.zeros(p_net.num_nodes, dtype=bool)

        for node_id in range(p_net.num_nodes):
            if ConstraintChecker.check_resource_feasible(p_net, node_id, vnf):
                candidates[node_id] = True

        return candidates

    @staticmethod
    def get_routing_feasible_candidates(p_net, prev_server, bw_req, latency_limit=None):
        candidates = np.zeros(p_net.num_nodes, dtype=bool)

        for node_id in range(p_net.num_nodes):
            path = RoutingEngine.find_feasible_path(p_net, prev_server, node_id, 
                                                    bw_req, latency_limit)
            if path is not None:
                candidates[node_id] = True

        return candidates

    @staticmethod
    def combine_masks(*masks):
        if len(masks) == 0:
            return None
        result = masks[0].copy()
        for mask in masks[1:]:
            result = np.logical_and(result, mask)
        return result

    @staticmethod
    def get_feasible_candidates(p_net, vnf, prev_server=None, bw_req=None, 
                               latency_limit=None):
        resource_mask = CandidateMasking.get_resource_feasible_candidates(p_net, vnf)

        if prev_server is None or bw_req is None:
            return resource_mask

        routing_mask = CandidateMasking.get_routing_feasible_candidates(
            p_net, prev_server, bw_req, latency_limit)

        return CandidateMasking.combine_masks(resource_mask, routing_mask)

    @staticmethod
    def get_feasible_server_list(mask):
        return np.where(mask)[0].tolist()