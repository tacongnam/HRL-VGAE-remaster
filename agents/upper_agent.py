"""
Upper Level Agent: Request Scheduler + Coarse DC Placement
Mô tả: Chọn request tiếp theo từ queue + chọn DC cho từng VNF trong SFC
"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque


class UpperMLP(nn.Module):
    """Multi-layer perceptron cho Upper Agent"""
    def __init__(self, input_dim, hidden_dim, output_dim, name="upper_net"):
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


class UpperAgent:
    """
    Hierarchical Agent cấp Upper: Scheduler + Placer
    - schedulenet: chọn request từ queue (output: K candidates)
    - placenet: chọn DC cho VNF (output: MAXDCS)
    """
    
    def __init__(self, 
                 latent_dim=32, 
                 max_dcs=10,
                 top_k_candidates=8,
                 hidden_dim=128,
                 gamma=0.95,
                 learning_rate=1e-3,
                 device='cpu'):
        self.latent_dim = latent_dim
        self.max_dcs = max_dcs
        self.top_k = top_k_candidates
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = learning_rate
        self.device = device
        
        # Tính input dim cho schedule network
        # globalz (latent_dim) + queuepressure + avgurgency + K requests feat (K * 5)
        self.schedule_input_dim = latent_dim + 2 + (top_k_candidates * 5)
        
        # Tính input dim cho placement network
        # globalz + vnf_feat (5) + locz + nodepressure
        self.place_input_dim = latent_dim + 5 + 1 + 1
        
        # Policy networks
        self.schedule_net = UpperMLP(self.schedule_input_dim, hidden_dim, top_k_candidates, "upper_schedule").to(device)
        self.place_net = UpperMLP(self.place_input_dim, hidden_dim, max_dcs, "upper_place").to(device)
        
        # Target networks cho Double DQN
        self.target_schedule_net = UpperMLP(self.schedule_input_dim, hidden_dim, top_k_candidates, "target_schedule").to(device)
        self.target_place_net = UpperMLP(self.place_input_dim, hidden_dim, max_dcs, "target_place").to(device)
        
        # Clone initial weights
        self._sync_target_networks()
        
        # Optimizers
        self.schedule_optimizer = torch.optim.Adam(self.schedule_net.parameters(), lr=learning_rate)
        self.place_optimizer = torch.optim.Adam(self.place_net.parameters(), lr=learning_rate)
        
        # Replay buffers
        self.schedule_buffer = deque(maxlen=10000)
        self.place_buffer = deque(maxlen=10000)
        
        self.update_counter_schedule = 0
        self.update_counter_place = 0
        self.target_sync_frequency_schedule = 50
        self.target_sync_frequency_place = 50
    
    def _sync_target_networks(self):
        """Sao chép weights từ policy sang target network"""
        self.target_schedule_net.load_state_dict(self.schedule_net.state_dict())
        self.target_place_net.load_state_dict(self.place_net.state_dict())
    
    def act_schedule(self, state, epsilon=0.1):
        """
        Chọn request từ queue
        Args:
            state: ndarray [batch_size, schedule_input_dim] hoặc [schedule_input_dim]
            epsilon: exploration rate
        Returns:
            action: int từ 0 đến top_k-1
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Epsilon-greedy
        if np.random.random() < epsilon:
            return np.random.randint(0, self.top_k)
        
        with torch.no_grad():
            q_values = self.schedule_net(state)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def act_place(self, state, epsilon=0.1):
        """
        Chọn DC cho VNF
        Args:
            state: ndarray [batch_size, place_input_dim] hoặc [place_input_dim]
            epsilon: exploration rate
        Returns:
            action: int từ 0 đến max_dcs-1
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Epsilon-greedy
        if np.random.random() < epsilon:
            return np.random.randint(0, self.max_dcs)
        
        with torch.no_grad():
            q_values = self.place_net(state)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def remember_schedule(self, state, action, reward, next_state, done):
        """Lưu transition cho scheduling vào buffer"""
        self.schedule_buffer.append((state, action, reward, next_state, done))
    
    def remember_place(self, state, action, reward, next_state, done):
        """Lưu transition cho placement vào buffer"""
        self.place_buffer.append((state, action, reward, next_state, done))
    
    def train_schedule(self, batch_size=32):
        """Train scheduling network với Double DQN"""
        if len(self.schedule_buffer) < batch_size:
            return None
        
        # Sample batch
        indices = np.random.choice(len(self.schedule_buffer), batch_size, replace=False)
        batch = [self.schedule_buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Double DQN: argmax from policy net, evaluate with target net
        with torch.no_grad():
            best_actions = self.schedule_net(next_states).argmax(dim=1)
            target_q = self.target_schedule_net(next_states)[range(batch_size), best_actions]
            target_q = rewards + self.gamma * target_q * (1 - dones)
        
        current_q = self.schedule_net(states)[range(batch_size), actions]
        loss = nn.MSELoss()(current_q, target_q)
        
        self.schedule_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.schedule_net.parameters(), max_norm=1.0)
        self.schedule_optimizer.step()
        
        self.update_counter_schedule += 1
        if self.update_counter_schedule % self.target_sync_frequency_schedule == 0:
            self.target_schedule_net.load_state_dict(self.schedule_net.state_dict())
        
        return loss.item()
    
    def train_place(self, batch_size=32):
        """Train placement network với Double DQN"""
        if len(self.place_buffer) < batch_size:
            return None
        
        # Sample batch
        indices = np.random.choice(len(self.place_buffer), batch_size, replace=False)
        batch = [self.place_buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Double DQN
        with torch.no_grad():
            best_actions = self.place_net(next_states).argmax(dim=1)
            target_q = self.target_place_net(next_states)[range(batch_size), best_actions]
            target_q = rewards + self.gamma * target_q * (1 - dones)
        
        current_q = self.place_net(states)[range(batch_size), actions]
        loss = nn.MSELoss()(current_q, target_q)
        
        self.place_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.place_net.parameters(), max_norm=1.0)
        self.place_optimizer.step()
        
        self.update_counter_place += 1
        if self.update_counter_place % self.target_sync_frequency_place == 0:
            self.target_place_net.load_state_dict(self.place_net.state_dict())
        
        return loss.item()
    
    def save_weights(self, filepath_schedule, filepath_place):
        """Lưu weights"""
        torch.save(self.schedule_net.state_dict(), filepath_schedule)
        torch.save(self.place_net.state_dict(), filepath_place)
    
    def load_weights(self, filepath_schedule, filepath_place):
        """Nạp weights"""
        self.schedule_net.load_state_dict(torch.load(filepath_schedule, map_location=self.device))
        self.place_net.load_state_dict(torch.load(filepath_place, map_location=self.device))
        self._sync_target_networks()
    
    def freeze(self):
        """Đóng băng weights (cho pretraining phase)"""
        for param in self.schedule_net.parameters():
            param.requires_grad = False
        for param in self.place_net.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Mở khóa weights"""
        for param in self.schedule_net.parameters():
            param.requires_grad = True
        for param in self.place_net.parameters():
            param.requires_grad = True
