import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar


class InnerProductDecoder(nn.Module):

    def __init__(self, latent_dim):
        super(InnerProductDecoder, self).__init__()

    def forward(self, z):
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj


class VGAE(nn.Module):

    def __init__(self, num_nodes, input_dim, hidden_dim, latent_dim):
        super(VGAE, self).__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = InnerProductDecoder(latent_dim)
        self.num_nodes = num_nodes

    def encode(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        z, mu, logvar = self.encode(x, edge_index)
        return z, mu, logvar

    def compute_loss(self, x, edge_index, adj):
        z, mu, logvar = self.forward(x, edge_index)
        adj_pred = self.decode(z)

        recon_loss = F.binary_cross_entropy(adj_pred, adj)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss, recon_loss, kl_loss