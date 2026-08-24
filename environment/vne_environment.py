import numpy as np
from typing import Dict, List, Tuple
from .network_state import NetworkState


class VNEEnvironment:

    def __init__(self, physical_network):
        self.p_net = physical_network
        self.current_request = None
        self.network_state = None
        self.step_count = 0

    def receive_request(self, sfc_request):
        self.current_request = sfc_request
        self.network_state = NetworkState(self.p_net)
        self.step_count = 0

    def step(self, vnf_index: int, selected_server: int, path=None):
        if self.current_request is None:
            return {'status': 'error', 'message': 'No request received'}, False

        vnf = self.current_request.get_vnf(vnf_index)
        if vnf is None:
            return {'status': 'error', 'message': 'Invalid VNF index'}, False

        if not self.p_net.allocate_resource(selected_server, vnf.cpu_req, vnf.ram_req, vnf.storage_req):
            return {'status': 'resource_failure'}, True

        self.network_state.add_placement(vnf_index, selected_server, path)
        self.network_state.record_resource_allocation(vnf.cpu_req, vnf.ram_req, vnf.storage_req)

        if path is not None:
            bw_req = self.current_request.link_bandwidth_reqs[vnf_index - 1] if vnf_index > 0 else 0
            self.network_state.record_bandwidth_allocation(bw_req)

        self.step_count += 1
        done = (vnf_index == len(self.current_request) - 1)

        return {'status': 'success', 'server': selected_server, 'path': path}, done

    def commit(self):
        self.network_state.commit()

    def rollback(self):
        self.network_state.rollback()

    def get_network_summary(self):
        total_cpu = sum(self.p_net.nodes[i].cpu_residual for i in range(self.p_net.num_nodes))
        total_bw = sum(self.p_net.edges[edge].bandwidth_residual 
                      for edge in self.p_net.edges if isinstance(edge, tuple))
        avg_load = sum(self.p_net.nodes[i].cpu_allocated for i in range(self.p_net.num_nodes)) / self.p_net.num_nodes
        
        return {
            'total_residual_cpu': total_cpu,
            'total_residual_bw': total_bw,
            'avg_node_load': avg_load,
        }

    def get_current_request_info(self):
        if self.current_request is None:
            return None
        return {
            'request_id': self.current_request.request_id,
            'num_vnfs': len(self.current_request),
            'num_links': self.current_request.get_num_links(),
        }