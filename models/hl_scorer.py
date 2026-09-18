import torch
import torch.nn as nn
from config import Config

class HLSharedQScorer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d_in = cfg.d_hl_input
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, z_global: torch.Tensor, sfc_features: torch.Tensor, pareto_w: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_global, sfc_features, pareto_w], dim=-1)
        return self.net(x)

class HLSharedQScorerTarget(HLSharedQScorer):
    pass
