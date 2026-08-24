import numpy as np


class MetricCalculator:

    @staticmethod
    def calculate_acceptance_ratio(results):
        if len(results) == 0:
            return 0.0
        success_count = sum(1 for r in results if r.get('success', False))
        return success_count / len(results)

    @staticmethod
    def calculate_resource_utilization(p_net):
        total_cpu = sum(node.cpu_capacity for node in p_net.nodes.values())
        used_cpu = sum(node.cpu_allocated for node in p_net.nodes.values())
        
        if total_cpu == 0:
            return 0.0
        return used_cpu / total_cpu

    @staticmethod
    def calculate_total_cost(p_net):
        total_cost = 0.0
        for node in p_net.nodes.values():
            cost = (node.cpu_allocated + 
                   node.ram_allocated / 10.0 + 
                   node.storage_allocated / 20.0)
            total_cost += cost
        return total_cost

    @staticmethod
    def calculate_load_imbalance(p_net):
        loads = []
        for node in p_net.nodes.values():
            load = (node.cpu_allocated / node.cpu_capacity)
            loads.append(load)

        if len(loads) == 0:
            return 0.0

        return float(np.var(loads))

    @staticmethod
    def calculate_avg_path_length(results):
        if len(results) == 0:
            return 0.0

        total_length = 0
        path_count = 0

        for result in results:
            if result.get('success', False):
                for path in result.get('paths', []):
                    if path is not None:
                        total_length += len(path)
                        path_count += 1

        if path_count == 0:
            return 0.0

        return total_length / path_count

    @staticmethod
    def calculate_avg_reward(results):
        if len(results) == 0:
            return 0.0

        total_reward = sum(r.get('total_reward', 0) for r in results)
        return total_reward / len(results)

    @staticmethod
    def get_detailed_metrics(results, p_net):
        return {
            'acceptance_ratio': MetricCalculator.calculate_acceptance_ratio(results),
            'resource_utilization': MetricCalculator.calculate_resource_utilization(p_net),
            'total_cost': MetricCalculator.calculate_total_cost(p_net),
            'load_imbalance': MetricCalculator.calculate_load_imbalance(p_net),
            'avg_path_length': MetricCalculator.calculate_avg_path_length(results),
            'avg_reward': MetricCalculator.calculate_avg_reward(results),
        }