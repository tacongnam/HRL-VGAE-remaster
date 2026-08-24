import unittest
import torch
import numpy as np
from model import DQNNetwork, DQNAgent, ReplayBuffer


class TestDQNNetwork(unittest.TestCase):

    def setUp(self):
        self.state_dim = 64
        self.action_dim = 50
        self.hidden_dim = 128

    def test_dqn_network_initialization(self):
        network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.assertIsNotNone(network.fc1)
        self.assertIsNotNone(network.fc2)
        self.assertIsNotNone(network.fc3)

    def test_dqn_network_forward(self):
        network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        state = torch.randn(self.state_dim)

        q_values = network(state)

        self.assertEqual(q_values.shape[0], self.action_dim)

    def test_dqn_network_batch_forward(self):
        network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        batch_size = 32
        states = torch.randn(batch_size, self.state_dim)

        q_values = network(states)

        self.assertEqual(q_values.shape, (batch_size, self.action_dim))


class TestReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.capacity = 1000
        self.buffer = ReplayBuffer(self.capacity)

    def test_replay_buffer_add(self):
        state = np.random.randn(64)
        action = 5
        reward = 1.0
        next_state = np.random.randn(64)
        done = False

        self.buffer.add(state, action, reward, next_state, done)

        self.assertEqual(len(self.buffer), 1)

    def test_replay_buffer_sample(self):
        for _ in range(100):
            state = np.random.randn(64)
            action = np.random.randint(0, 50)
            reward = np.random.randn()
            next_state = np.random.randn(64)
            done = False

            self.buffer.add(state, action, reward, next_state, done)

        batch_size = 32
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        self.assertEqual(states.shape, (batch_size, 64))
        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertEqual(next_states.shape, (batch_size, 64))
        self.assertEqual(dones.shape, (batch_size,))

    def test_replay_buffer_capacity(self):
        for i in range(1500):
            state = np.random.randn(64)
            action = 0
            reward = 0
            next_state = np.random.randn(64)
            done = False

            self.buffer.add(state, action, reward, next_state, done)

        self.assertEqual(len(self.buffer), self.capacity)


class TestDQNAgent(unittest.TestCase):

    def setUp(self):
        self.state_dim = 64
        self.num_servers = 50
        self.config = {
            'learning_rate': 1e-3,
            'gamma': 0.95,
            'epsilon': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.995,
            'target_update_freq': 500,
            'batch_size': 32,
            'replay_buffer_size': 10000,
            'hidden_dim': 128,
        }

    def test_dqn_agent_initialization(self):
        agent = DQNAgent(self.state_dim, self.num_servers, self.config)

        self.assertIsNotNone(agent.q_network)
        self.assertIsNotNone(agent.target_network)
        self.assertIsNotNone(agent.replay_buffer)

    def test_dqn_agent_select_action_random(self):
        agent = DQNAgent(self.state_dim, self.num_servers, self.config)
        agent.epsilon = 1.0

        state = np.random.randn(self.state_dim)
        mask = np.ones(self.num_servers, dtype=bool)

        action = agent.select_action(state, mask)

        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.num_servers)

    def test_dqn_agent_select_action_greedy(self):
        agent = DQNAgent(self.state_dim, self.num_servers, self.config)
        agent.epsilon = 0.0

        state = torch.randn(self.state_dim)
        mask = np.ones(self.num_servers, dtype=bool)

        action = agent.select_action(state, mask)

        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.num_servers)

    def test_dqn_agent_store_transition(self):
        agent = DQNAgent(self.state_dim, self.num_servers, self.config)

        state = np.random.randn(self.state_dim)
        action = 5
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        done = False

        agent.store_transition(state, action, reward, next_state, done)

        self.assertEqual(len(agent.replay_buffer), 1)

    def test_dqn_agent_train_step(self):
        agent = DQNAgent(self.state_dim, self.num_servers, self.config)

        for _ in range(50):
            state = np.random.randn(self.state_dim)
            action = np.random.randint(0, self.num_servers)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim)
            done = False

            agent.store_transition(state, action, reward, next_state, done)

        loss = agent.train_step(batch_size=32)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)


if __name__ == '__main__':
    unittest.main()