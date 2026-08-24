import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import default_config as config
from data import NetworkGenerator, SFCRequestGenerator
from model import VGAE, DQNAgent
from algorithm import PlacementEngine, RoutingEngine
from trainer import DQNTrainer
import torch


def main():
    print("Starting Fine-tuning Phase 3")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_p_nodes = config.PHYSICAL_NETWORK['num_nodes']

    print(f"\nGenerating physical network...")
    p_net = NetworkGenerator.generate_physical_network(
        num_p_nodes,
        config.PHYSICAL_NETWORK['topology'],
        config.PHYSICAL_NETWORK['cpu_capacity'],
        config.PHYSICAL_NETWORK['ram_capacity'],
        config.PHYSICAL_NETWORK['storage_capacity'],
        config.PHYSICAL_NETWORK['bandwidth_capacity']
    )

    print(f"Loading VGAE model...")
    vgae_model = VGAE(
        num_p_nodes,
        input_dim=4,
        hidden_dim=32,
        latent_dim=config.VGAE['embedding_dim']
    )

    vgae_checkpoint_path = Path('./outputs/vgae_pretrained.pth')
    if vgae_checkpoint_path.exists():
        checkpoint = torch.load(str(vgae_checkpoint_path))
        vgae_model.load_state_dict(checkpoint['model'])
        print(f"VGAE model loaded from {vgae_checkpoint_path}")

    vgae_model.to(device)
    vgae_model.eval()

    p_net_features = torch.FloatTensor(p_net.get_node_features()).to(device)
    p_net_edges = p_net.get_edge_index().to(device)

    with torch.no_grad():
        z, _, _ = vgae_model.encode(p_net_features, p_net_edges)

    state_dim = len(z.flatten()) + 3 + config.VGAE['embedding_dim'] + 2

    print(f"Initializing DQN Agent...")
    dqn_agent = DQNAgent(state_dim, p_net.num_nodes, config.DQN)
    dqn_agent.q_network.to(device)
    dqn_agent.target_network.to(device)

    dqn_checkpoint_path = Path('./outputs/dqn_trained.pth')
    if dqn_checkpoint_path.exists():
        dqn_agent.load(str(dqn_checkpoint_path))
        print(f"DQN model loaded from {dqn_checkpoint_path}")

    routing_engine = RoutingEngine()
    placement_engine = PlacementEngine(p_net, vgae_model, dqn_agent, routing_engine)

    sfc_generator = SFCRequestGenerator(
        cpu_range=config.SFC['cpu_requirement'],
        ram_range=config.SFC['ram_requirement'],
        storage_range=config.SFC['storage_requirement'],
        bw_range=config.SFC['bandwidth_requirement'],
        vnf_count_range=(config.SFC['num_vnfs_min'], config.SFC['num_vnfs_max'])
    )

    num_finetune_episodes = config.TRAINING.get('num_finetune_episodes', 50)

    print(f"Starting fine-tuning for {num_finetune_episodes} episodes...")
    dqn_trainer = DQNTrainer(placement_engine, dqn_agent, config.TRAINING)
    dqn_trainer.train(num_finetune_episodes, sfc_generator)

    output_dir = Path('./outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'dqn_finetuned.pth'
    dqn_trainer.save_model(str(checkpoint_path))

    print(f"\nFine-tuning completed!")
    print(f"Model saved to {checkpoint_path}")


if __name__ == '__main__':
    main()