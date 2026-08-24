"""
Lower Level Agent: Load Balancer + Routing
Mô tả: Chọn đường routing hoặc slot cụ thể trong DC đã được chọn bởi Upper
"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque


class LowerMLP(nn.Module):
    """Multi-layer perceptron cho Lower Agent"""
    def __init__(self, input_dim, hidden_dim, output_dim, name="lower_net"):
        super().__init__()
        self.name = name
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class LowerAgent:
    """
    Hierarchical Agent cấp Lower: Load Balancer + Routing
    - Nhận DC đã chọn từ Upper
    - Quyết định route cụ thể hoặc slot cụ thể (K candidates)
    - Reward: negative của node pressure + link pressure (cân bằng tải)
    """
    
    def __init__(self, 
                 state_dim=20,
                 k_candidates=3,
                 hidden_dim=64,
                 gamma=0.9,
                 learning_rate=1e-3,
                 device='cpu'):
        self.state_dim = state_dim
        self.k_candidates = k_candidates  # K-shortest paths hoặc K slots
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = learning_rate
        self.device = device
        
        # Policy network
        self.policy_net = LowerMLP(state_dim, hidden_dim, k_candidates, "lower_policy").to(device)
        
        # Target network cho Double DQN
        self.target_net = LowerMLP(state_dim, hidden_dim, k_candidates, "lower_target").to(device)
        
        # Clone initial weights
        self._sync_target_network()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.buffer = deque(maxlen=10000)
        
        self.update_counter = 0
        self.target_sync_frequency = 20  # Lower sync nhanh hơn Upper (20 vs 50)
    
    def _sync_target_network(self):
        """Sao chép weights từ policy sang target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def act(self, state, epsilon=0.1):
        """
        Chọn route/slot từ K candidates
        Args:
            state: ndarray [batch_size, state_dim] hoặc [state_dim]
            epsilon: exploration rate
        Returns:
            action: int từ 0 đến k_candidates-1
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Epsilon-greedy
        if np.random.random() < epsilon:
            return np.random.randint(0, self.k_candidates)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Lưu transition vào buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=32):
        """Train policy network với Double DQN"""
        if len(self.buffer) < batch_size:
            return None
        
        # Sample batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Double DQN: argmax from policy net, evaluate with target net
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1)
            target_q = self.target_net(next_states)[range(batch_size), best_actions]
            target_q = rewards + self.gamma * target_q * (1 - dones)
        
        current_q = self.policy_net(states)[range(batch_size), actions]
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.target_sync_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save_weights(self, filepath):
        """Lưu weights"""
        torch.save(self.policy_net.state_dict(), filepath)
    
    def load_weights(self, filepath):
        """Nạp weights"""
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self._sync_target_network()
    
    def freeze(self):
        """Đóng băng weights (cho pretraining phase)"""
        for param in self.policy_net.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Mở khóa weights"""
        for param in self.policy_net.parameters():
            param.requires_grad = True
