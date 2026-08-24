import numpy as np
import torch
from collections import deque
from typing import Tuple


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([torch.FloatTensor(s) if isinstance(s, np.ndarray) else s 
                             for s in states])
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack([torch.FloatTensor(s) if isinstance(s, np.ndarray) else s 
                                  for s in next_states])
        dones = torch.FloatTensor(dones)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()