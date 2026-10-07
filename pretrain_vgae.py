import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from config import Config
from models import MNVGAE
from data import parse_episode, get_node_features_from_graph


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def recon_loss_sampled(z_nodes, edge_index, num_nodes, neg_ratio, device):
    if edge_index.shape[1] == 0:
        return torch.tensor(0.0, device=device)
    pos_score = (z_nodes[edge_index[0]] * z_nodes[edge_index[1]]).sum(dim=-1)
    num_neg = max(1, int(edge_index.shape[1] * neg_ratio))
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_score = (z_nodes[neg_src] * z_nodes[neg_dst]).sum(dim=-1)
    return F.binary_cross_entropy_with_logits(
        pos_score, torch.ones_like(pos_score)
    ) + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="pretrained_vgae.pt")
    args = parser.parse_args()

    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    G, _, _, topology_id = parse_episode(args.episode)
    num_nodes = G.number_of_nodes()

    x = torch.tensor(
        get_node_features_from_graph(G, cfg.vgae.d_in),
        dtype=torch.float32,
        device=device,
    )
    edges = list(G.edges())
    ei = (
        torch.tensor(
            [
                [u for u, v in edges] + [v for u, v in edges],
                [v for u, v in edges] + [u for u, v in edges],
            ],
            dtype=torch.long,
            device=device,
        )
        if edges
        else torch.zeros((2, 0), dtype=torch.long, device=device)
    )

    vgae = MNVGAE(cfg).to(device)
    static_params = [
        p for name, p in vgae.named_parameters() if not name.startswith("temporal_cell")
    ]
    optimizer = optim.Adam(static_params, lr=args.lr)

    vgae.train()
    best_loss = float("inf")
    best_state_dict = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        z_nodes, _, mu, logvar = vgae(x, ei, graph_id=None, update_temporal=False)
        recon = recon_loss_sampled(z_nodes, ei, num_nodes, args.neg_ratio, device)
        kl = vgae.kl_loss(mu, logvar)
        loss = recon + kl
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in vgae.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{args.epochs} | loss={loss.item():.6f} (recon={recon.item():.6f} kl={kl.item():.6f})"
            )

    torch.save(
        {
            "vgae": (
                best_state_dict if best_state_dict is not None else vgae.state_dict()
            ),
            "pretrain_cfg": {
                "d_in": cfg.vgae.d_in,
                "d_hidden": cfg.vgae.d_hidden,
                "d_latent": cfg.vgae.d_latent,
                "temporal": cfg.vgae.temporal,
            },
            "topology_id": topology_id,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
        },
        args.out,
    )
    print(f"Pretrained VGAE saved → {args.out}")


if __name__ == "__main__":
    main()
