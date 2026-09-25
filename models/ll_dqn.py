import torch
import torch.nn as nn
from config import Config

N_OBJ = 4


class LLNodeScorer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d_in = cfg.d_ll_input
        self.shared = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
        )
        self.q_heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(N_OBJ)])

    def forward(
        self, z_global, z_prev_node, z_cand_node, vnf_features, sfc_features, pareto_w
    ):
        x = torch.cat(
            [z_global, z_prev_node, z_cand_node, vnf_features, sfc_features, pareto_w],
            dim=-1,
        )
        h = self.shared(x)
        return torch.cat([head(h) for head in self.q_heads], dim=-1)
