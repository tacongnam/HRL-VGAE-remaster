"""
Pre-train static VGAE encoder from topology in a single train.json.

Only the graph topology (V, E) is used — requests (R) are ignored.
Temporal GRU cell is NOT trained here; it learns during HRL training.

Usage:
    python pretrain_vgae.py \\
        --episode data/train.json \\
        --epochs 100 \\
        --lr 1e-3 \\
        --neg-ratio 1.0 \\
        --seed 42 \\
        --out pretrained_vgae.pt \\
        --device cpu
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.vgae import MNVGAE
from data.loader import parse_episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_graph_tensors(G, cfg: Config, device: torch.device):
    """Extract node feature tensor and edge_index from nx.Graph."""
    from data.loader import get_node_features_from_graph
    x_np = get_node_features_from_graph(G, cfg.vgae.d_in)
    x = torch.tensor(x_np, dtype=torch.float32, device=device)

    edges = list(G.edges())
    if edges:
        src = [u for u, v in edges] + [v for u, v in edges]
        dst = [v for u, v in edges] + [u for u, v in edges]
        ei = torch.tensor([src, dst], dtype=torch.long, device=device)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long, device=device)
    return x, ei


def recon_loss_sampled(z_nodes, edge_index, num_nodes, neg_ratio, device):
    num_pos = edge_index.shape[1]
    if num_pos == 0:
        return torch.tensor(0.0, device=device)
    pos_score = (z_nodes[edge_index[0]] * z_nodes[edge_index[1]]).sum(dim=-1)
    num_neg = max(1, int(num_pos * neg_ratio))
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_score = (z_nodes[neg_src] * z_nodes[neg_dst]).sum(dim=-1)
    return (
        F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-train VGAE static encoder")
    parser.add_argument("--episode", type=str, required=True,
                        help="Path to train.json (topology + requests; requests are ignored)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default matches cfg.vgae.lr=1e-3)")
    parser.add_argument("--neg-ratio", type=float, default=1.0,
                        help="Negative edge sampling ratio relative to positive edges")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="pretrained_vgae.pt",
                        help="Output checkpoint path")
    parser.add_argument("--device", type=str, default="",
                        help="'cpu' or 'cuda'. Auto-detect if empty.")
    args = parser.parse_args()

    set_seeds(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    cfg = Config()

    # Load topology only — requests are ignored
    print(f"Loading topology from: {args.episode}")
    G, _requests, _F_raw, topology_id = parse_episode(args.episode)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"  Topology id : {topology_id}")
    print(f"  Nodes       : {num_nodes}")
    print(f"  Edges       : {num_edges}")

    x, ei = build_graph_tensors(G, cfg, device)

    # Build model — temporal disabled during pre-training
    vgae = MNVGAE(cfg).to(device)

    # Only optimize static encoder parameters; exclude temporal_cell
    static_params = [
        p for name, p in vgae.named_parameters()
        if not name.startswith("temporal_cell")
    ]
    optimizer = optim.Adam(static_params, lr=args.lr)

    print(f"\nPre-training static encoder for {args.epochs} epochs ...")
    print(f"  Trainable params : {sum(p.numel() for p in static_params):,}")
    temporal_params = sum(p.numel() for name, p in vgae.named_parameters()
                          if name.startswith("temporal_cell"))
    print(f"  temporal_cell params (excluded) : {temporal_params:,}")

    vgae.train()
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # Forward without graph_id so GRU is bypassed completely
        z_nodes, _z_global, mu, logvar = vgae(x, ei, graph_id=None, update_temporal=False)

        recon = recon_loss_sampled(z_nodes, ei, num_nodes, args.neg_ratio, device)
        kl = vgae.kl_loss(mu, logvar)
        loss = recon + kl

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs} | loss={loss_val:.6f} "
                  f"(recon={recon.item():.6f} kl={kl.item():.6f})")

    print(f"\nBest loss: {best_loss:.6f}")

    # Save checkpoint
    # temporal_cell state is saved too (random-init) so HRL can load full model
    checkpoint = {
        "vgae": vgae.state_dict(),          # full state dict including temporal_cell
        "pretrain_cfg": {
            "d_in": cfg.vgae.d_in,
            "d_hidden": cfg.vgae.d_hidden,
            "d_latent": cfg.vgae.d_latent,
            "temporal": cfg.vgae.temporal,
        },
        "topology_id": topology_id,
        "epochs": args.epochs,
        "best_loss": best_loss,
        "seed": args.seed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(checkpoint, args.out)
    print(f"Checkpoint saved → {args.out}")


if __name__ == "__main__":
    main()
