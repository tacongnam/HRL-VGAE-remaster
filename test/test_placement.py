import unittest
import torch
from environment import PhysicalNetwork, VNF, SFCRequest
from model import VGAE, DQNAgent
from algorithm import PlacementEngine, RoutingEngine


class TestPlacementEngine(unittest.TestCase):

    def setUp(self):
        self.num_nodes = 10
        self.p_net = PhysicalNetwork(self.num_nodes, topology='scale_free')

        self.vgae = VGAE(self.num_nodes, input_dim=4, hidden_dim=16, latent_dim=8)
        self.vgae.eval()

        self.state_dim = 8 * self.num_nodes + 3 + 8 + 2
        self.dqn_agent = DQNAgent(self.state_dim, self.num_nodes, {
            'learning_rate': 1e-3,
            'gamma': 0.95,
            'epsilon': 0.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.995,
            'target_update_freq': 500,
            'batch_size': 32,
            'replay_buffer_size': 10000,
        })

        self.routing_engine = RoutingEngine()
        self.placement_engine = PlacementEngine(self.p_net, self.vgae, 
                                               self.dqn_agent, self.routing_engine)

    def test_placement_single_vnf(self):
        vnf = VNF(0, 10, 16, 32)
        sfc_request = SFCRequest(0, [vnf], [])

        result = self.placement_engine.place_sfc(sfc_request, training=False)

        self.assertIn('success', result)
        self.assertIn('placement', result)

    def test_placement_multiple_vnfs(self):
        vnfs = [
            VNF(0, 10, 16, 32),
            VNF(1, 20, 32, 64),
            VNF(2, 15, 24, 48),
        ]
        sfc_request = SFCRequest(0, vnfs, [50, 50])

        result = self.placement_engine.place_sfc(sfc_request, training=False)

        self.assertIn('success', result)
        if result['success']:
            self.assertEqual(len(result['placement']), len(vnfs))

    def test_placement_with_training(self):
        vnfs = [
            VNF(0, 10, 16, 32),
            VNF(1, 20, 32, 64),
        ]
        sfc_request = SFCRequest(0, vnfs, [50])

        result = self.placement_engine.place_sfc(sfc_request, training=True)

        self.assertIn('transitions', result)


if __name__ == '__main__':
    unittest.main()