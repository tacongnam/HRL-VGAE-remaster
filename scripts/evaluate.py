import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import default_config as config
from data import NetworkGenerator, SFCRequestGenerator
from model import VGAE, DQNAgent
from algorithm import PlacementEngine, RoutingEngine
from utils import MetricCalculator, Visualizer
import torch


def main():
    print("Starting Evaluation")

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
        checkpoint = torch.load(str(vgae_checkpoint_path), map_location=device)
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

    dqn_checkpoint_path = Path('./outputs/dqn_finetuned.pth')
    if not dqn_checkpoint_path.exists():
        dqn_checkpoint_path = Path('./outputs/dqn_trained.pth')

    if dqn_checkpoint_path.exists():
        dqn_agent.load(str(dqn_checkpoint_path))
        print(f"DQN model loaded from {dqn_checkpoint_path}")
    else:
        print("Warning: DQN checkpoint not found")

    routing_engine = RoutingEngine()
    placement_engine = PlacementEngine(p_net, vgae_model, dqn_agent, routing_engine)

    sfc_generator = SFCRequestGenerator(
        cpu_range=config.SFC['cpu_requirement'],
        ram_range=config.SFC['ram_requirement'],
        storage_range=config.SFC['storage_requirement'],
        bw_range=config.SFC['bandwidth_requirement'],
        vnf_count_range=(config.SFC['num_vnfs_min'], config.SFC['num_vnfs_max'])
    )

    print(f"\nGenerating test requests...")
    num_test_requests = 100
    test_requests = sfc_generator.generate(num_test_requests)

    print(f"Running evaluation on {num_test_requests} requests...")
    results = []
    for i, sfc_request in enumerate(test_requests):
        result = placement_engine.place_sfc(sfc_request, training=False, 
                                           use_epsilon_greedy=False)
        results.append(result)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_test_requests} requests")

    print(f"\nCalculating metrics...")
    metrics = MetricCalculator.get_detailed_metrics(results, p_net)

    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:.<30} {value:.4f}")

    output_dir = Path('./outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'evaluation_results.txt'
    with open(results_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("="*50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key:.<30} {value:.4f}\n")

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()