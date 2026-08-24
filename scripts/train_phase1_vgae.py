import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import default_config as config
from trainer import VGAETrainer
from data import NetworkGenerator
import torch


def main():
    print("Starting VGAE Pretraining Phase 1")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_p_nodes = config.PHYSICAL_NETWORK['num_nodes']
    topology = config.PHYSICAL_NETWORK['topology']

    print(f"\nGenerating pretraining dataset...")
    num_pretrain_nets = 5
    p_net_template = NetworkGenerator.generate_physical_network(
        num_p_nodes, topology,
        config.PHYSICAL_NETWORK['cpu_capacity'],
        config.PHYSICAL_NETWORK['ram_capacity'],
        config.PHYSICAL_NETWORK['storage_capacity'],
        config.PHYSICAL_NETWORK['bandwidth_capacity']
    )

    pretrain_nets = [p_net_template.copy() for _ in range(num_pretrain_nets)]

    from model import VGAE

    print(f"Initializing VGAE model...")
    vgae_model = VGAE(
        num_p_nodes,
        input_dim=4,
        hidden_dim=32,
        latent_dim=config.VGAE['embedding_dim']
    )
    vgae_model.to(device)

    trainer = VGAETrainer(vgae_model, config.VGAE)

    print(f"Starting VGAE training...")
    trainer.train(pretrain_nets, config.VGAE['num_epochs'])

    output_dir = Path('./outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'vgae_pretrained.pth'
    trainer.save(str(checkpoint_path))

    print(f"\nVGAE pretraining completed!")
    print(f"Model saved to {checkpoint_path}")


if __name__ == '__main__':
    main()