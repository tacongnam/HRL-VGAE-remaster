import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from config import Config


class MNVGAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d_in = cfg.vgae.d_in
        d_h = cfg.vgae.d_hidden
        d_z = cfg.vgae.d_latent

        self.gcn0 = GCNConv(d_in, d_h)
        self.skip = nn.Linear(d_in, d_h, bias=False)
        self.gcn_mu = GCNConv(d_h, d_z)
        self.gcn_logvar = GCNConv(d_h, d_z)

        self.temporal_enabled = cfg.vgae.temporal
        d_g = 2 * d_z
        if self.temporal_enabled:
            self.temporal_cell = nn.GRUCell(d_g, d_g)
            self._graph_states = {}

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        h1 = F.relu(self.gcn0(x, edge_index)) + self.skip(x)
        mu = self.gcn_mu(h1, edge_index)
        logvar = self.gcn_logvar(h1, edge_index)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def reset_temporal_state(self, graph_id):
        if self.temporal_enabled and graph_id in self._graph_states:
            del self._graph_states[graph_id]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        graph_id=None,
        update_temporal: bool = True,
    ):
        mu, logvar = self.encode(x, edge_index)
        z_nodes = self.reparameterize(mu, logvar)

        z_mean = z_nodes.mean(dim=0)
        z_max = z_nodes.max(dim=0).values
        z_global_static = torch.cat([z_mean, z_max], dim=-1)

        z_global = z_global_static
        if self.temporal_enabled and graph_id is not None:
            prev = self._graph_states.get(graph_id)
            if prev is None:
                prev = z_global_static.detach()
            new_state = self.temporal_cell(
                z_global_static.unsqueeze(0), prev.unsqueeze(0)
            ).squeeze(0)
            z_global = new_state
            if update_temporal:
                self._graph_states[graph_id] = new_state.detach()

        return z_nodes, z_global, mu, logvar

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
