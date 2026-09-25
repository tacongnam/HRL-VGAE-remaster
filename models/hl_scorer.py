import torch
import torch.nn as nn
from config import Config

N_OBJ_HL = 4


class HLSharedQScorer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d_in = cfg.d_hl_input
        self.trunk = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.q_heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(N_OBJ_HL)])

    def forward(self, z_global, sfc_features, pareto_w):
        x = torch.cat([z_global, sfc_features, pareto_w], dim=-1)
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.q_heads], dim=-1)


HLSharedQScorerTarget = HLSharedQScorer
