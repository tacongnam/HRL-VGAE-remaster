import numpy as np
from typing import Dict, List
from copy import deepcopy


class NetworkState:
    def __init__(self, p_net):
        self.p_net = deepcopy(p_net)
        self.placed_vnfs = []
        self.selected_servers = []
        self.allocated_paths = []
        self.allocated_resources = []

    def add_placement(self, vnf_index, server_id, path=None):
        self.placed_vnfs.append(vnf_index)
        self.selected_servers.append(server_id)
        if path is not None:
            self.allocated_paths.append(path)

    def rollback(self):
        for server_id, resources in zip(self.selected_servers, self.allocated_resources):
            cpu, ram, storage = resources
            self.p_net.deallocate_resource(server_id, cpu, ram, storage)

        for path, bw_req in zip(self.allocated_paths, 
                               self.allocated_resources_bw if hasattr(self, 'allocated_resources_bw') else []):
            for i in range(len(path) - 1):
                self.p_net.deallocate_link_bw(path[i], path[i + 1], bw_req)

        self.placed_vnfs = []
        self.selected_servers = []
        self.allocated_paths = []
        self.allocated_resources = []

    def commit(self):
        self.placed_vnfs = []
        self.selected_servers = []
        self.allocated_paths = []
        self.allocated_resources = []

    def record_resource_allocation(self, cpu, ram, storage):
        self.allocated_resources.append((cpu, ram, storage))

    def record_bandwidth_allocation(self, bw_req):
        if not hasattr(self, 'allocated_resources_bw'):
            self.allocated_resources_bw = []
        self.allocated_resources_bw.append(bw_req)