import numpy as np
from typing import List


class VNF:
    def __init__(self, vnf_id, cpu_req, ram_req, storage_req):
        self.vnf_id = vnf_id
        self.cpu_req = cpu_req
        self.ram_req = ram_req
        self.storage_req = storage_req

    def get_requirements(self):
        return {
            'cpu': self.cpu_req,
            'ram': self.ram_req,
            'storage': self.storage_req,
        }

    def __repr__(self):
        return f"VNF({self.vnf_id}, CPU:{self.cpu_req}, RAM:{self.ram_req}, Storage:{self.storage_req})"


class SFCRequest:
    def __init__(self, request_id, vnfs: List[VNF], link_bandwidth_reqs: List[int]):
        self.request_id = request_id
        self.vnfs = vnfs
        self.link_bandwidth_reqs = link_bandwidth_reqs
        self.arrival_time = None
        self.duration = None
        self.departure_time = None

    def __len__(self):
        return len(self.vnfs)

    def get_vnf(self, index):
        if 0 <= index < len(self.vnfs):
            return self.vnfs[index]
        return None

    def get_num_links(self):
        return len(self.link_bandwidth_reqs)

    def __repr__(self):
        return f"SFC({self.request_id}, {len(self.vnfs)} VNFs, {self.get_num_links()} links)"