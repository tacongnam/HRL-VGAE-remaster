import unittest
import torch
from model import VGAE, GCNEncoder, InnerProductDecoder


class TestVGAE(unittest.TestCase):

    def setUp(self):
        self.num_nodes = 10
        self.input_dim = 4
        self.hidden_dim = 16
        self.latent_dim = 8

    def test_vgae_initialization(self):
        vgae = VGAE(self.num_nodes, self.input_dim, self.hidden_dim, self.latent_dim)
        self.assertIsNotNone(vgae.encoder)
        self.assertIsNotNone(vgae.decoder)

    def test_gcn_encoder_forward(self):
        encoder = GCNEncoder(self.input_dim, self.hidden_dim, self.latent_dim)
        x = torch.randn(self.num_nodes, self.input_dim)
        edge_index = torch.LongTensor([[0, 1, 2], [1, 2, 3]])

        mu, logvar = encoder(x, edge_index)

        self.assertEqual(mu.shape, (self.num_nodes, self.latent_dim))
        self.assertEqual(logvar.shape, (self.num_nodes, self.latent_dim))

    def test_inner_product_decoder_forward(self):
        decoder = InnerProductDecoder(self.latent_dim)
        z = torch.randn(self.num_nodes, self.latent_dim)

        adj_pred = decoder(z)

        self.assertEqual(adj_pred.shape, (self.num_nodes, self.num_nodes))
        self.assertTrue(torch.all(adj_pred >= 0) and torch.all(adj_pred <= 1))

    def test_vgae_reparameterize(self):
        vgae = VGAE(self.num_nodes, self.input_dim, self.hidden_dim, self.latent_dim)
        mu = torch.randn(self.num_nodes, self.latent_dim)
        logvar = torch.randn(self.num_nodes, self.latent_dim)

        z = vgae.reparameterize(mu, logvar)

        self.assertEqual(z.shape, (self.num_nodes, self.latent_dim))

    def test_vgae_forward(self):
        vgae = VGAE(self.num_nodes, self.input_dim, self.hidden_dim, self.latent_dim)
        x = torch.randn(self.num_nodes, self.input_dim)
        edge_index = torch.LongTensor([[0, 1, 2], [1, 2, 3]])

        z, mu, logvar = vgae(x, edge_index)

        self.assertEqual(z.shape, (self.num_nodes, self.latent_dim))
        self.assertEqual(mu.shape, (self.num_nodes, self.latent_dim))
        self.assertEqual(logvar.shape, (self.num_nodes, self.latent_dim))


if __name__ == '__main__':
    unittest.main()