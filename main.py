import sys
import numpy as np
import torch
from pathlib import Path

from config import default_config as config
from data import NetworkGenerator, SFCRequestGenerator
from environment import PhysicalNetwork, VNEEnvironment
from model import VGAE, DQNAgent
from algorithm import PlacementEngine, RoutingEngine
from trainer import VGAETrainer, DQNTrainer
from utils import MetricCalculator, Logger


def create_vgae_model(num_nodes, input_dim=4, hidden_dim=32, latent_dim=16):
    vgae = VGAE(num_nodes, input_dim, hidden_dim, latent_dim)
    return vgae


def create_dqn_agent(state_dim, num_servers, dqn_config):
    agent = DQNAgent(state_dim, num_servers, dqn_config)
    return agent


def pretrain_vgae(vgae_model, p_net_dataset, vgae_config):
    print("\n" + "="*50)
    print("Phase 1: VGAE Pretraining")
    print("="*50)

    trainer = VGAETrainer(vgae_model, vgae_config)
    trainer.train(p_net_dataset, vgae_config['num_epochs'])

    return vgae_model


def train_dqn(placement_engine, dqn_agent, sfc_generator, training_config, 
             num_episodes=None):
    print("\n" + "="*50)
    print("Phase 2: DQN Training")
    print("="*50)

    if num_episodes is None:
        num_episodes = training_config.get('num_episodes', 100)

    trainer = DQNTrainer(placement_engine, dqn_agent, training_config)
    trainer.train(num_episodes, sfc_generator)

    return trainer


def evaluate(placement_engine, p_net, num_test_requests=100):
    print("\n" + "="*50)
    print("Evaluation")
    print("="*50)

    sfc_generator = SFCRequestGenerator()
    test_requests = sfc_generator.generate(num_test_requests)

    results = []
    for sfc_request in test_requests:
        result = placement_engine.place_sfc(sfc_request, training=False, use_epsilon_greedy=False)
        results.append(result)

    metrics = MetricCalculator.get_detailed_metrics(results, p_net)

    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    return metrics, results


def main():
    print("Starting VGAE-DQN-SFCP Training Pipeline")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    logger = Logger('./logs')

    num_p_nodes = config.PHYSICAL_NETWORK['num_nodes']
    topology = config.PHYSICAL_NETWORK['topology']

    print(f"\nInitializing Physical Network: {num_p_nodes} nodes, topology: {topology}")
    p_net = NetworkGenerator.generate_physical_network(
        num_p_nodes, topology,
        config.PHYSICAL_NETWORK['cpu_capacity'],
        config.PHYSICAL_NETWORK['ram_capacity'],
        config.PHYSICAL_NETWORK['storage_capacity'],
        config.PHYSICAL_NETWORK['bandwidth_capacity']
    )

    print("Generating pretraining dataset for VGAE...")
    num_pretrain_nets = 5
    pretrain_nets = [p_net.copy() for _ in range(num_pretrain_nets)]

    print(f"Initializing VGAE model...")
    vgae_model = create_vgae_model(
        num_p_nodes,
        input_dim=4,
        hidden_dim=32,
        latent_dim=config.VGAE['embedding_dim']
    )
    vgae_model.to(device)

    vgae_model = pretrain_vgae(vgae_model, pretrain_nets, config.VGAE)
    vgae_model.eval()

    p_net_features = torch.FloatTensor(p_net.get_node_features())
    p_net_edges = p_net.get_edge_index()
    with torch.no_grad():
        z, _, _ = vgae_model.encode(p_net_features, p_net_edges)
    state_dim = len(z.flatten()) + 3 + config.VGAE['embedding_dim'] + 2
    
    print(f"State dimension: {state_dim}")

    print(f"\nInitializing DQN Agent...")
    dqn_agent = create_dqn_agent(state_dim, num_p_nodes, config.DQN)

    routing_engine = RoutingEngine()

    placement_engine = PlacementEngine(p_net, vgae_model, dqn_agent, routing_engine)

    sfc_generator = SFCRequestGenerator(
        cpu_range=config.SFC['cpu_requirement'],
        ram_range=config.SFC['ram_requirement'],
        storage_range=config.SFC['storage_requirement'],
        bw_range=config.SFC['bandwidth_requirement'],
        vnf_count_range=(config.SFC['num_vnfs_min'], config.SFC['num_vnfs_max'])
    )

    dqn_trainer = train_dqn(placement_engine, dqn_agent, sfc_generator, 
                           config.TRAINING, config.TRAINING['num_episodes'])

    metrics, eval_results = evaluate(placement_engine, p_net, num_test_requests=100)

    for key, value in metrics.items():
        logger.log(metric=key, value=value)

    logger.save()

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)

    return {
        'vgae_model': vgae_model,
        'dqn_agent': dqn_agent,
        'placement_engine': placement_engine,
        'trainer': dqn_trainer,
        'metrics': metrics,
        'eval_results': eval_results,
    }


if __name__ == '__main__':
    result = main()