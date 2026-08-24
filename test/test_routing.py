import unittest
from environment import PhysicalNetwork, VNF
from algorithm import RoutingEngine


class TestRoutingEngine(unittest.TestCase):

    def setUp(self):
        self.p_net = PhysicalNetwork(10, topology='random')

    def test_find_feasible_path_same_node(self):
        path = RoutingEngine.find_feasible_path(self.p_net, 0, 0, 100)

        self.assertEqual(path, [0])

    def test_find_feasible_path_direct_edge(self):
        if self.p_net.graph.has_edge(0, 1):
            bw_cap = self.p_net.get_link_residual_bw(0, 1)
            path = RoutingEngine.find_feasible_path(self.p_net, 0, 1, bw_cap // 2)

            self.assertIsNotNone(path)
            self.assertIn(0, path)
            self.assertIn(1, path)

    def test_find_feasible_path_no_bandwidth(self):
        self.p_net.allocate_link_bw(0, 1, 1000)

        path = RoutingEngine.find_feasible_path(self.p_net, 0, 1, 100)

        self.assertIsNone(path)

    def test_allocate_bandwidth(self):
        path = [0, 1, 2]
        bw_req = 50

        success = RoutingEngine.allocate_bandwidth(self.p_net, path, bw_req)

        if success:
            self.assertLess(self.p_net.get_link_residual_bw(0, 1),
                           self.p_net.bw_capacity)

    def test_deallocate_bandwidth(self):
        path = [0, 1, 2]
        bw_req = 50

        RoutingEngine.allocate_bandwidth(self.p_net, path, bw_req)
        bw_before = self.p_net.get_link_residual_bw(0, 1)

        RoutingEngine.deallocate_bandwidth(self.p_net, path, bw_req)
        bw_after = self.p_net.get_link_residual_bw(0, 1)

        self.assertGreater(bw_after, bw_before)

    def test_calculate_path_latency(self):
        path = [0, 1, 2]

        latency = RoutingEngine.calculate_path_latency(self.p_net, path)

        self.assertGreaterEqual(latency, 0)

    def test_calculate_path_cost(self):
        path = [0, 1, 2]

        cost = RoutingEngine.calculate_path_cost(self.p_net, path)

        self.assertEqual(cost, len(path) - 1)


if __name__ == '__main__':
    unittest.main()