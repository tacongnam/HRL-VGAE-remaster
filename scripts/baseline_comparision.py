import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import default_config as config
from data import NetworkGenerator, SFCRequestGenerator
from environment import SFCRequest
from utils import MetricCalculator
import numpy as np


class RandomPlacementSolver:
    def __init__(self, p_net):
        self.p_net = p_net

    def solve(self, sfc_request):
        placement = []
        paths = []

        for vnf_index, vnf in enumerate(sfc_request.vnfs):
            valid_servers = []
            for server_id in range(self.p_net.num_nodes):
                residual = self.p_net.get_node_residual_resources(server_id)
                if (residual['cpu'] >= vnf.cpu_req and
                    residual['ram'] >= vnf.ram_req and
                    residual['storage'] >= vnf.storage_req):
                    valid_servers.append(server_id)

            if not valid_servers:
                self._rollback(placement, paths, sfc_request)
                return {'success': False, 'placement': [], 'paths': []}

            selected_server = np.random.choice(valid_servers)

            if vnf_index > 0:
                prev_server = placement[-1]
                bw_req = sfc_request.link_bandwidth_reqs[vnf_index - 1]
                path = self._simple_path(prev_server, selected_server)
                if path is None:
                    self._rollback(placement, paths, sfc_request)
                    return {'success': False, 'placement': [], 'paths': []}
                paths.append(path)
                self._allocate_bandwidth(path, bw_req)

            self.p_net.allocate_resource(selected_server, vnf.cpu_req, 
                                        vnf.ram_req, vnf.storage_req)
            placement.append(selected_server)

        return {'success': True, 'placement': placement, 'paths': paths}

    def _simple_path(self, src, dst):
        if src == dst:
            return [src]
        if self.p_net.graph.has_edge(src, dst):
            return [src, dst]
        return None

    def _allocate_bandwidth(self, path, bw_req):
        for i in range(len(path) - 1):
            self.p_net.allocate_link_bw(path[i], path[i + 1], bw_req)

    def _rollback(self, placement, paths, sfc_request):
        for server, vnf in zip(placement, sfc_request.vnfs):
            self.p_net.deallocate_resource(server, vnf.cpu_req, 
                                          vnf.ram_req, vnf.storage_req)
        for path, bw_req in zip(paths, sfc_request.link_bandwidth_reqs):
            for i in range(len(path) - 1):
                self.p_net.deallocate_link_bw(path[i], path[i + 1], bw_req)


class FirstFitSolver:
    def __init__(self, p_net):
        self.p_net = p_net

    def solve(self, sfc_request):
        placement = []
        paths = []

        for vnf_index, vnf in enumerate(sfc_request.vnfs):
            selected_server = None
            for server_id in range(self.p_net.num_nodes):
                residual = self.p_net.get_node_residual_resources(server_id)
                if (residual['cpu'] >= vnf.cpu_req and
                    residual['ram'] >= vnf.ram_req and
                    residual['storage'] >= vnf.storage_req):
                    selected_server = server_id
                    break

            if selected_server is None:
                self._rollback(placement, paths, sfc_request)
                return {'success': False, 'placement': [], 'paths': []}

            if vnf_index > 0:
                prev_server = placement[-1]
                bw_req = sfc_request.link_bandwidth_reqs[vnf_index - 1]
                path = self._simple_path(prev_server, selected_server)
                if path is None:
                    self._rollback(placement, paths, sfc_request)
                    return {'success': False, 'placement': [], 'paths': []}
                paths.append(path)
                self._allocate_bandwidth(path, bw_req)

            self.p_net.allocate_resource(selected_server, vnf.cpu_req, 
                                        vnf.ram_req, vnf.storage_req)
            placement.append(selected_server)

        return {'success': True, 'placement': placement, 'paths': paths}

    def _simple_path(self, src, dst):
        if src == dst:
            return [src]
        if self.p_net.graph.has_edge(src, dst):
            return [src, dst]
        return None

    def _allocate_bandwidth(self, path, bw_req):
        for i in range(len(path) - 1):
            self.p_net.allocate_link_bw(path[i], path[i + 1], bw_req)

    def _rollback(self, placement, paths, sfc_request):
        for server, vnf in zip(placement, sfc_request.vnfs):
            self.p_net.deallocate_resource(server, vnf.cpu_req, 
                                          vnf.ram_req, vnf.storage_req)
        for path, bw_req in zip(paths, sfc_request.link_bandwidth_reqs):
            for i in range(len(path) - 1):
                self.p_net.deallocate_link_bw(path[i], path[i + 1], bw_req)


class BestFitSolver:
    def __init__(self, p_net):
        self.p_net = p_net

    def solve(self, sfc_request):
        placement = []
        paths = []

        for vnf_index, vnf in enumerate(sfc_request.vnfs):
            best_server = None
            best_fitness = float('inf')

            for server_id in range(self.p_net.num_nodes):
                residual = self.p_net.get_node_residual_resources(server_id)
                if (residual['cpu'] >= vnf.cpu_req and
                    residual['ram'] >= vnf.ram_req and
                    residual['storage'] >= vnf.storage_req):
                    fitness = (residual['cpu'] - vnf.cpu_req + 
                             residual['ram'] - vnf.ram_req + 
                             residual['storage'] - vnf.storage_req)
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_server = server_id

            if best_server is None:
                self._rollback(placement, paths, sfc_request)
                return {'success': False, 'placement': [], 'paths': []}

            if vnf_index > 0:
                prev_server = placement[-1]
                bw_req = sfc_request.link_bandwidth_reqs[vnf_index - 1]
                path = self._simple_path(prev_server, best_server)
                if path is None:
                    self._rollback(placement, paths, sfc_request)
                    return {'success': False, 'placement': [], 'paths': []}
                paths.append(path)
                self._allocate_bandwidth(path, bw_req)

            self.p_net.allocate_resource(best_server, vnf.cpu_req, 
                                        vnf.ram_req, vnf.storage_req)
            placement.append(best_server)

        return {'success': True, 'placement': placement, 'paths': paths}

    def _simple_path(self, src, dst):
        if src == dst:
            return [src]
        if self.p_net.graph.has_edge(src, dst):
            return [src, dst]
        return None

    def _allocate_bandwidth(self, path, bw_req):
        for i in range(len(path) - 1):
            self.p_net.allocate_link_bw(path[i], path[i + 1], bw_req)

    def _rollback(self, placement, paths, sfc_request):
        for server, vnf in zip(placement, sfc_request.vnfs):
            self.p_net.deallocate_resource(server, vnf.cpu_req, 
                                          vnf.ram_req, vnf.storage_req)
        for path, bw_req in zip(paths, sfc_request.link_bandwidth_reqs):
            for i in range(len(path) - 1):
                self.p_net.deallocate_link_bw(path[i], path[i + 1], bw_req)


def main():
    print("Starting Baseline Comparison")

    num_p_nodes = config.PHYSICAL_NETWORK['num_nodes']

    print(f"\nGenerating physical network...")
    p_net = NetworkGenerator.generate_physical_network(
        num_p_nodes,
        config.PHYSICAL_NETWORK['topology'],
        config.PHYSICAL_NETWORK['cpu_capacity'],
        config.PHYSICAL_NETWORK['ram_capacity'],
        config.PHYSICAL_NETWORK['storage_capacity'],
        config.PHYSICAL_NETWORK['bandwidth_capacity']
    )

    sfc_generator = SFCRequestGenerator(
        cpu_range=config.SFC['cpu_requirement'],
        ram_range=config.SFC['ram_requirement'],
        storage_range=config.SFC['storage_requirement'],
        bw_range=config.SFC['bandwidth_requirement'],
        vnf_count_range=(config.SFC['num_vnfs_min'], config.SFC['num_vnfs_max'])
    )

    baselines = {
        'Random': RandomPlacementSolver,
        'First-Fit': FirstFitSolver,
        'Best-Fit': BestFitSolver,
    }

    results_dict = {}

    for baseline_name, SolverClass in baselines.items():
        print(f"\n{'='*50}")
        print(f"Testing: {baseline_name}")
        print(f"{'='*50}")

        p_net_copy = p_net.copy()
        solver = SolverClass(p_net_copy)

        test_requests = sfc_generator.generate(50)
        results = []

        for sfc_request in test_requests:
            result = solver.solve(sfc_request)
            results.append(result)

        metrics = MetricCalculator.get_detailed_metrics(results, p_net_copy)
        results_dict[baseline_name] = metrics

        for key, value in metrics.items():
            print(f"{key:.<30} {value:.4f}")

    output_dir = Path('./outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / 'baseline_comparison.txt'
    with open(comparison_path, 'w') as f:
        f.write("Baseline Comparison Results\n")
        f.write("="*70 + "\n\n")

        for baseline_name, metrics in results_dict.items():
            f.write(f"Baseline: {baseline_name}\n")
            f.write("-"*70 + "\n")
            for key, value in metrics.items():
                f.write(f"  {key:.<40} {value:.4f}\n")
            f.write("\n")

    print(f"\n\nComparison results saved to {comparison_path}")


if __name__ == '__main__':
    from data import NetworkGenerator
    main()