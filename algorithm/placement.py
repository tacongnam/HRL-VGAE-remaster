import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from .candidate_mask import CandidateMasking
from .routing import RoutingEngine
from .reward import RewardCalculator


class PlacementEngine:

    def __init__(self, p_net, vgae, dqn_agent, routing_engine):
        self.p_net = p_net
        self.vgae = vgae
        self.dqn_agent = dqn_agent
        self.routing_engine = routing_engine
        self.current_request = None

    def place_sfc(self, sfc_request, training=True, use_epsilon_greedy=True):
        self.current_request = sfc_request
        
        result = {
            'success': False,
            'request_id': sfc_request.request_id,
            'placement': [],
            'paths': [],
            'total_reward': 0.0,
            'transitions': [],
            'failure_reason': None,
        }

        try:
            with torch.no_grad():
                z, _, _ = self.vgae.encode(
                    torch.FloatTensor(self.p_net.get_node_features()),
                    self.p_net.get_edge_index()
                )
        except Exception as e:
            result['failure_reason'] = f"VGAE encoding failed: {str(e)}"
            return result

        for vnf_index, vnf in enumerate(sfc_request.vnfs):
            resource_mask = CandidateMasking.get_resource_feasible_candidates(
                self.p_net, vnf)

            if vnf_index > 0:
                prev_server = result['placement'][-1]
                bw_req = sfc_request.link_bandwidth_reqs[vnf_index - 1]
                routing_mask = CandidateMasking.get_routing_feasible_candidates(
                    self.p_net, prev_server, bw_req)
                final_mask = CandidateMasking.combine_masks(resource_mask, routing_mask)
            else:
                final_mask = resource_mask

            if not final_mask.any():
                result['failure_reason'] = f"No feasible candidate for VNF {vnf_index}"
                self._rollback(result)
                return result

            state = self._build_state(z, vnf, result, vnf_index, sfc_request)

            if use_epsilon_greedy:
                selected_server = self.dqn_agent.select_action(state, final_mask)
            else:
                with torch.no_grad():
                    q_values = self.dqn_agent.q_network(state)
                    q_values[~final_mask] = -float('inf')
                    selected_server = q_values.argmax().item()

            if vnf_index > 0:
                prev_server = result['placement'][-1]
                bw_req = sfc_request.link_bandwidth_reqs[vnf_index - 1]
                path = self.routing_engine.find_feasible_path(
                    self.p_net, prev_server, selected_server, bw_req)

                if path is None:
                    result['failure_reason'] = f"No feasible path from {prev_server} to {selected_server}"
                    self._rollback(result)
                    return result

                if not self.routing_engine.allocate_bandwidth(self.p_net, path, bw_req):
                    result['failure_reason'] = f"Failed to allocate bandwidth for path"
                    self._rollback(result)
                    return result

                result['paths'].append(path)

            if not self.p_net.allocate_resource(selected_server, 
                                               vnf.cpu_req, vnf.ram_req, vnf.storage_req):
                result['failure_reason'] = f"Failed to allocate resource on server {selected_server}"
                self._rollback(result)
                return result

            reward = RewardCalculator.calculate_intermediate_reward(
                self.p_net, selected_server, vnf)

            if vnf_index > 0 and len(result['paths']) > 0:
                path_reward = RewardCalculator.calculate_path_reward(
                    self.p_net, result['paths'][-1])
                reward += path_reward

            result['placement'].append(selected_server)
            result['total_reward'] += reward

            if training:
                next_state = self._build_state(z, None, result, vnf_index + 1, sfc_request)
                done = (vnf_index == len(sfc_request.vnfs) - 1)

                transition = {
                    'state': state,
                    'action': selected_server,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done,
                    'candidate_mask': final_mask,
                }
                result['transitions'].append(transition)

        result['success'] = True
        result['total_reward'] += RewardCalculator.calculate_success_bonus()
        return result

    def _build_state(self, z, vnf, placement_result, vnf_index, sfc_request):
        z_np = z.detach().cpu().numpy().flatten()

        if vnf is not None:
            vnf_req = np.array([
                vnf.cpu_req / 100.0,
                vnf.ram_req / 256.0,
                vnf.storage_req / 512.0
            ], dtype=np.float32)
        else:
            vnf_req = np.zeros(3, dtype=np.float32)

        if vnf_index > 0:
            prev_server = placement_result['placement'][-1]
            prev_embedding = z[prev_server].detach().cpu().numpy()
        else:
            prev_embedding = np.zeros(z.shape[1], dtype=np.float32)

        network_summary = self._get_network_summary()

        state = np.concatenate([z_np, vnf_req, prev_embedding, network_summary])
        return torch.FloatTensor(state)

    def _get_network_summary(self):
        total_cpu = 0
        total_bw = 0
        node_count = 0

        for node_id in range(self.p_net.num_nodes):
            residual = self.p_net.get_node_residual_resources(node_id)
            total_cpu += residual['cpu']
            node_count += 1

        avg_cpu = total_cpu / node_count if node_count > 0 else 0

        valid_edges = 0
        for edge in self.p_net.edges:
            if isinstance(edge, tuple):
                total_bw += self.p_net.edges[edge].bandwidth_residual
                valid_edges += 1

        avg_bw = total_bw / valid_edges if valid_edges > 0 else 0

        return np.array([avg_cpu / 200.0, avg_bw / 1000.0], dtype=np.float32)

    def _rollback(self, result):
        for server, vnf in zip(result['placement'], self.current_request.vnfs):
            self.p_net.deallocate_resource(server, vnf.cpu_req, vnf.ram_req, vnf.storage_req)

        for path, bw_req in zip(result['paths'], self.current_request.link_bandwidth_reqs):
            self.routing_engine.deallocate_bandwidth(self.p_net, path, bw_req)