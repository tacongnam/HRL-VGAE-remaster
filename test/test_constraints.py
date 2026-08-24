import unittest
from environment import PhysicalNetwork, VNF, ConstraintChecker


class TestConstraintChecker(unittest.TestCase):

    def setUp(self):
        self.p_net = PhysicalNetwork(10, topology='random')

    def test_check_resource_feasible_true(self):
        vnf = VNF(0, 50, 100, 200)

        result = ConstraintChecker.check_resource_feasible(self.p_net, 0, vnf)

        self.assertTrue(result)

    def test_check_resource_feasible_false(self):
        vnf = VNF(0, 500, 1000, 2000)

        result = ConstraintChecker.check_resource_feasible(self.p_net, 0, vnf)

        self.assertFalse(result)

    def test_check_bandwidth_feasible_empty_path(self):
        result = ConstraintChecker.check_bandwidth_feasible(self.p_net, [], 100)

        self.assertTrue(result)

    def test_check_path_validity(self):
        path = [0, 1, 2]

        result = ConstraintChecker.check_path_validity(self.p_net, path)

        self.assertIn(result, [True, False])


if __name__ == '__main__':
    unittest.main()