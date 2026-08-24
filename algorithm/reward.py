import numpy as np


class RewardCalculator:

    @staticmethod
    def calculate_intermediate_reward(p_net, selected_server, vnf, 
                                     lambda_c=0.5, lambda_f=0.1):
        resource_cost = vnf.cpu_req + vnf.ram_req / 10.0 + vnf.storage_req / 20.0

        residual = p_net.get_node_residual_resources(selected_server)
        fit_score = (residual['cpu'] + residual['ram'] / 10.0 + residual['storage'] / 20.0)

        reward = -lambda_c * resource_cost + lambda_f * fit_score
        return reward

    @staticmethod
    def calculate_path_reward(p_net, path, lambda_l=0.1, lambda_d=0.05):
        if path is None or len(path) < 2:
            return 0.0

        from .routing import RoutingEngine
        
        latency = RoutingEngine.calculate_path_latency(p_net, path)
        cost = RoutingEngine.calculate_path_cost(p_net, path)

        reward = -lambda_l * latency - lambda_d * cost
        return reward

    @staticmethod
    def calculate_total_placement_reward(sfc_request, p_net, placement_result,
                                        lambda_c=0.5, lambda_l=0.1, lambda_d=0.05,
                                        lambda_f=0.1):
        total_reward = 0.0

        for i, (server, vnf) in enumerate(zip(placement_result['placement'], 
                                              sfc_request.vnfs)):
            inter_reward = RewardCalculator.calculate_intermediate_reward(
                p_net, server, vnf, lambda_c, lambda_f)
            total_reward += inter_reward

            if i < len(placement_result['paths']):
                path = placement_result['paths'][i]
                path_reward = RewardCalculator.calculate_path_reward(
                    p_net, path, lambda_l, lambda_d)
                total_reward += path_reward

        return total_reward

    @staticmethod
    def calculate_failure_penalty(penalty_factor=10.0):
        return -penalty_factor

    @staticmethod
    def calculate_success_bonus(bonus_factor=5.0):
        return bonus_factor