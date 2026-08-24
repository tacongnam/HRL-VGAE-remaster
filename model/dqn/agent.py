import numpy as np
import torch
from torch.optim import Adam
from .network import DQNNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(self, state_dim, num_servers, config):
        self.state_dim = state_dim
        self.num_servers = num_servers
        self.config = config

        self.q_network = DQNNetwork(state_dim, num_servers, config.get('hidden_dim', 128))
        self.target_network = DQNNetwork(state_dim, num_servers, config.get('hidden_dim', 128))
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = Adam(self.q_network.parameters(), 
                             lr=config.get('learning_rate', 1e-3))
        self.replay_buffer = ReplayBuffer(config.get('replay_buffer_size', 10000))

        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.gamma = config.get('gamma', 0.95)
        self.target_update_freq = config.get('target_update_freq', 500)
        self.batch_size = config.get('batch_size', 32)
        
        self.update_counter = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.q_network.to(self.device)
        self.target_network.to(self.device)

    def select_action(self, state, candidate_mask):
        if np.random.rand() < self.epsilon:
            candidates = np.where(candidate_mask)[0]
            if len(candidates) == 0:
                return np.random.randint(0, self.num_servers)
            return np.random.choice(candidates)
        else:
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state).to(self.device)
                elif isinstance(state, torch.Tensor):
                    state = state.to(self.device)

                q_values = self.q_network(state)
                
                if isinstance(q_values, torch.Tensor) and len(q_values.shape) > 1:
                    q_values = q_values.squeeze(0)
                
                q_values_np = q_values.cpu().numpy()
                q_values_np[~candidate_mask] = -float('inf')
                
                valid_actions = np.where(candidate_mask)[0]
                if len(valid_actions) == 0:
                    return np.random.randint(0, self.num_servers)
                
                return int(np.argmax(q_values_np))

    def store_transition(self, state, action, reward, next_state, done):
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            q_next = self.target_network(next_states)
            if len(q_next.shape) > 1:
                q_next = q_next.max(1)[0]
            else:
                q_next = q_next.max()
            
            target_q = rewards + self.gamma * q_next * (1 - dones)

        q_pred = self.q_network(states)
        if len(q_pred.shape) > 1:
            q_pred = q_pred[range(batch_size), actions]

        loss = torch.nn.functional.mse_loss(q_pred, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])