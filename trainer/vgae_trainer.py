import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm


class VGAETrainer:

    def __init__(self, vgae_model, config):
        self.vgae = vgae_model
        self.config = config
        self.optimizer = Adam(vgae_model.parameters(), 
                             lr=config.get('learning_rate', 1e-3))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vgae.to(self.device)

    def train_step(self, x, edge_index, adj):
        self.vgae.train()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        adj = adj.to(self.device)

        z, mu, logvar = self.vgae(x, edge_index)
        adj_pred = self.vgae.decode(z)

        recon_loss = F.binary_cross_entropy(adj_pred, adj)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), recon_loss.item(), kl_loss.item()

    def train_epoch(self, p_net_samples):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for p_net in p_net_samples:
            x = torch.FloatTensor(p_net.get_node_features())
            edge_index = p_net.get_edge_index()
            adj = p_net.get_adjacency_matrix()

            loss, recon, kl = self.train_step(x, edge_index, adj)

            total_loss += loss
            total_recon += recon
            total_kl += kl

        avg_loss = total_loss / len(p_net_samples)
        avg_recon = total_recon / len(p_net_samples)
        avg_kl = total_kl / len(p_net_samples)

        return avg_loss, avg_recon, avg_kl

    def train(self, p_net_dataset, num_epochs):
        for epoch in range(num_epochs):
            avg_loss, avg_recon, avg_kl = self.train_epoch(p_net_dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")

    def save(self, path):
        torch.save({
            'model': self.vgae.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.vgae.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def eval(self):
        self.vgae.eval()