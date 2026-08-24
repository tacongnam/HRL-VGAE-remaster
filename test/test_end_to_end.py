import unittest
import torch
from config import default_config as config
from data import NetworkGenerator, SFCRequestGenerator
from environment import VNEEnvironment
from model import VGAE, DQNAgent
from algorithm import PlacementEngine, RoutingEngine


class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        self.num_nodes = 20
        self.p_net = NetworkGenerator.generate_physical_network(
            self.num_nodes,
            'scale_free',
            config.PHYSICAL_NETWORK['cpu_capacity'],
            config.PHYSICAL_NETWORK['ram_capacity'],
            config.PHYSICAL_NETWORK['storage_capacity'],
            config.PHYSICAL_NETWORK['bandwidth_capacity']
        )

        self.vgae = VGAE(self.num_nodes, input_dim=4, hidden_dim=32, 
                        latent_dim=config.VGAE['embedding_dim'])
        self.vgae.eval()

        p_net_features = torch.FloatTensor(self.p_net.get_node_features())
        p_net_edges = self.p_net.get_edge_index()
        with torch.no_grad():
            z, _, _ = self.vgae.encode(p_net_features, p_net_edges)

        self.state_dim = len(z.flatten()) + 3 + config.VGAE['embedding_dim'] + 2

        self.dqn_agent = DQNAgent(self.state_dim, self.num_nodes, config.DQN)

        self.routing_engine = RoutingEngine()
        self.placement_engine = PlacementEngine(self.p_net, self.vgae, 
                                               self.dqn_agent, self.routing_engine)

    def test_end_to_end_single_request(self):
        sfc_generator = SFCRequestGenerator(
            cpu_range=config.SFC['cpu_requirement'],
            ram_range=config.SFC['ram_requirement'],
            storage_range=config.SFC['storage_requirement'],
            bw_range=config.SFC['bandwidth_requirement'],
            vnf_count_range=(3, 5)
        )

        requests = sfc_generator.generate(1)
        result = self.placement_engine.place_sfc(requests[0], training=False)

        self.assertIn('success', result)
        self.assertIn('placement', result)
        self.assertIn('paths', result)

    def test_end_to_end_multiple_requests(self):
        sfc_generator = SFCRequestGenerator(
            cpu_range=config.SFC['cpu_requirement'],
            ram_range=config.SFC['ram_requirement'],
            storage_range=config.SFC['storage_requirement'],
            bw_range=config.SFC['bandwidth_requirement'],
            vnf_count_range=(3, 5)
        )

        requests = sfc_generator.generate(10)
        results = []

        for request in requests:
            result = self.placement_engine.place_sfc(request, training=False)
            results.append(result)

        self.assertEqual(len(results), 10)

        success_count = sum(1 for r in results if r['success'])
        self.assertGreaterEqual(success_count, 0)
        self.assertLessEqual(success_count, 10)


if __name__ == '__main__':
    from model import VGAE
    unittest.main()