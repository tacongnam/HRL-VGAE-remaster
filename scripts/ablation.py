import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import default_config as config
from data import NetworkGenerator, SFCRequestGenerator
from model import VGAE, DQNAgent
from algorithm import PlacementEngine, RoutingEngine
from utils import MetricCalculator
import torch
import numpy as np


def run_variant(variant_name, p_net, use_vgae=True, num_test_requests=50):
    print(f"\n{'='*50}")
    print(f"Running variant: {variant_name}")
    print(f"{'='*50}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_p_nodes = p_net.num_nodes

    if use_vgae:
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

        vgae_model.to(device)
        vgae_model.eval()

        p_net_features = torch.FloatTensor(p_net.get_node_features()).to(device)
        p_net_edges = p_net.get_edge_index().to(device)

        with torch.no_grad():
            z, _, _ = vgae_model.encode(p_net_features, p_net_edges)

        state_dim = len(z.flatten()) + 3 + config.VGAE['embedding_dim'] + 2
    else:
        vgae_model = None
        p_net_features = torch.FloatTensor(p_net.get_node_features()).to(device)
        state_dim = len(p_net_features.flatten()) + 3 + 32 + 2

    dqn_agent = DQNAgent(state_dim, num_p_nodes, config.DQN)
    dqn_agent.q_network.to(device)
    dqn_agent.target_network.to(device)

    dqn_checkpoint_path = Path('./outputs/dqn_trained.pth')
    if dqn_checkpoint_path.exists():
        dqn_agent.load(str(dqn_checkpoint_path))

    routing_engine = RoutingEngine()
    placement_engine = PlacementEngine(p_net, vgae_model, dqn_agent, routing_engine)

    sfc_generator = SFCRequestGenerator(
        cpu_range=config.SFC['cpu_requirement'],
        ram_range=config.SFC['ram_requirement'],
        storage_range=config.SFC['storage_requirement'],
        bw_range=config.SFC['bandwidth_requirement'],
        vnf_count_range=(config.SFC['num_vnfs_min'], config.SFC['num_vnfs_max'])
    )

    test_requests = sfc_generator.generate(num_test_requests)

    results = []
    for sfc_request in test_requests:
        result = placement_engine.place_sfc(sfc_request, training=False, 
                                           use_epsilon_greedy=False)
        results.append(result)

    metrics = MetricCalculator.get_detailed_metrics(results, p_net)

    for key, value in metrics.items():
        print(f"{key:.<30} {value:.4f}")

    return metrics


def main():
    print("Starting Ablation Study")

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

    variants = {
        'DQN + Raw Features': False,
        'DQN + VGAE': True,
    }

    results_dict = {}
    for variant_name, use_vgae in variants.items():
        metrics = run_variant(variant_name, p_net.copy(), use_vgae=use_vgae, 
                            num_test_requests=50)
        results_dict[variant_name] = metrics

    output_dir = Path('./outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_path = output_dir / 'ablation_results.txt'
    with open(ablation_path, 'w') as f:
        f.write("Ablation Study Results\n")
        f.write("="*70 + "\n\n")

        for variant_name, metrics in results_dict.items():
            f.write(f"Variant: {variant_name}\n")
            f.write("-"*70 + "\n")
            for key, value in metrics.items():
                f.write(f"  {key:.<40} {value:.4f}\n")
            f.write("\n")

    print(f"\n\nAblation results saved to {ablation_path}")


if __name__ == '__main__':
    main()