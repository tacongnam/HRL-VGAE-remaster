import torch
import torch.nn as nn


class DQNNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if isinstance(x, list) or (isinstance(x, torch.Tensor) and x.dim() == 1):
            x = torch.FloatTensor(x) if not isinstance(x, torch.Tensor) else x
            x = x.unsqueeze(0) if x.dim() == 1 else x

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze(0) if x.shape[0] == 1 and len(x.shape) > 1 else x