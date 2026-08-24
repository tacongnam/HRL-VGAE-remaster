
import sys
import numpy as np
import torch
from pathlib import Path

# Import config, data, environment, models
from config import default_config as config
from data import NetworkGenerator, SFCRequestGenerator
from environment import PhysicalNetwork, VNEEnvironment

# Import new HRL components
from agents.upper_agent import UpperAgent
from agents.lower_agent import LowerAgent
from strategy.hrl_strategy import HRLStrategy
from trainers.hrl_trainer import HRLTrainer

# Import existing components
from model import VGAE
from algorithm import RoutingEngine
from utils import MetricCalculator, Logger


def create_vgae_model(num_nodes, input_dim=4, hidden_dim=32, latent_dim=16, device='cpu'):
    """Khởi tạo VGAE model"""
    vgae = VGAE(num_nodes, input_dim, hidden_dim, latent_dim)
    vgae.to(device)
    return vgae


def evaluate_strategy(strategy, p_net, sfc_generator, num_test_requests=100, device='cpu'):
    """
    Đánh giá chiến lược sau khi training
    """
    print("\n" + "="*60)
    print("Evaluation Phase")
    print("="*60)
    
    strategy.upper_agent.schedule_net.eval()
    strategy.upper_agent.place_net.eval()
    strategy.lower_agent.policy_net.eval()
    
    test_requests = sfc_generator.generate(num_test_requests)
    
    results = {
        'acceptance_ratio': 0,
        'resource_utilization': 0,
        'total_cost': 0,
        'load_imbalance': 0,
        'avg_path_length': 0,
        'avg_reward': 0,
    }
    
    accepted = 0
    total_cost = 0
    rewards = []
    
    for req in test_requests:
        # Tạo environment test
        test_env = VNEEnvironment(p_net)
        test_env.reset()
        
        # Build state cho scheduling (dummy queue với 1 request)
        node_pressures = test_env.get_node_pressures()
        schedule_state = strategy.build_schedule_state([req], node_pressures)
        
        # Upper: Schedule (lấy request)
        req_idx = strategy.upper_agent.act_schedule(schedule_state, epsilon=0.0)
        
        placed_vnfs = []
        success = True
        request_cost = 0
        
        # Process VNFs
        locz = 0.0
        for vnf_idx, vnf in enumerate(req.vnfs):
            node_pressure = node_pressures.get(0, 0.5)
            place_state = strategy.build_place_state(vnf, locz, node_pressure)
            
            # Upper: Placement
            dc_idx = strategy.upper_agent.act_place(place_state, epsilon=0.0)
            dc_idx = min(dc_idx, strategy.max_dcs - 1)
            
            # Try placement
            place_success, placement_cost = test_env.place_vnf(vnf, dc_idx)
            
            if place_success:
                placed_vnfs.append(vnf)
                request_cost += placement_cost
                locz = (vnf_idx + 1) / max(len(req.vnfs), 1)
                
                # Lower: Routing (inference only, no training)
                link_pressures = test_env.get_link_pressures()
                lower_state = strategy.build_lower_state(dc_idx, vnf, node_pressures, link_pressures)
                
                route_idx = strategy.lower_agent.act(lower_state, epsilon=0.0)
                test_env.execute_routing(dc_idx, vnf, route_idx)
            else:
                success = False
                break
        
        if success and placed_vnfs:
            accepted += 1
            total_cost += request_cost
            reward = strategy.compute_upper_reward(req, placed_vnfs, True)
            rewards.append(reward)
    
    results['acceptance_ratio'] = accepted / max(num_test_requests, 1)
    results['total_cost'] = total_cost
    results['avg_reward'] = np.mean(rewards) if rewards else 0.0
    results['resource_utilization'] = np.mean([np.mean(list(p_net.nodes[n].get('load', {}).values() or [0.5])) 
                                               for n in p_net.nodes()])
    
    return results


def main():
    """
    Main training pipeline với các pha tuần tự
    """
    print("="*60)
    print("HRL-VGAE-Remaster: Hierarchical RL for VNF Placement & Routing")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    logger = Logger('./logs')
    
    # ============================================================================
    # STEP 1: Khởi tạo Physical Network
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 1: Initialize Physical Network")
    print("="*60)
    
    num_p_nodes = config['PHYSICAL_NETWORK']['num_nodes']
    topology = config['PHYSICAL_NETWORK']['topology']
    
    print(f"Generating physical network: {num_p_nodes} nodes, topology: {topology}")
    p_net = NetworkGenerator.generate_physical_network(
        num_p_nodes, topology,
        config['PHYSICAL_NETWORK']['cpu_capacity'],
        config['PHYSICAL_NETWORK']['ram_capacity'],
        config['PHYSICAL_NETWORK']['storage_capacity'],
        config['PHYSICAL_NETWORK']['bandwidth_capacity']
    )
    print(f"✓ Physical network created with {p_net.num_nodes} nodes\n")
    
    # ============================================================================
    # STEP 2: VGAE Pretraining (Phase 0)
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 2: Create and Pretrain VGAE (Phase 0)")
    print("="*60)
    
    vgae_model = create_vgae_model(
        num_p_nodes,
        input_dim=4,
        hidden_dim=32,
        latent_dim=config['VGAE']['embedding_dim'],
        device=device
    )
    
    # Tạo pretraining dataset
    num_pretrain_nets = config['VGAE'].get('num_pretrain_nets', 5)
    pretrain_nets = [p_net.copy() for _ in range(num_pretrain_nets)]
    
    # Initialize HRLStrategy với VGAE
    strategy = HRLStrategy(
        p_net=p_net,
        vgae_model=vgae_model,
        latent_dim=config['VGAE']['embedding_dim'],
        max_dcs=config['PHYSICAL_NETWORK']['num_nodes'],
        top_k_candidates=config['HRL'].get('top_k_candidates', 8),
        k_routes=config['HRL'].get('k_routes', 3),
        upper_hidden_dim=config['HRL'].get('upper_hidden_dim', 128),
        lower_hidden_dim=config['HRL'].get('lower_hidden_dim', 64),
        device=device
    )
    
    # Initialize trainer
    trainer = HRLTrainer(strategy, config, device=device)
    
    # Phase 0: Pretrain VGAE
    trainer.pretrain_vgae(pretrain_nets, num_epochs=config['VGAE'].get('num_epochs', 100))
    
    # ============================================================================
    # STEP 3: Environment Setup
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 3: Setup Environment and SFC Generator")
    print("="*60)
    
    env = VNEEnvironment(p_net)
    
    sfc_generator = SFCRequestGenerator(
        cpu_range=config['SFC']['cpu_requirement'],
        ram_range=config['SFC']['ram_requirement'],
        storage_range=config['SFC']['storage_requirement'],
        bw_range=config['SFC']['bandwidth_requirement'],
        vnf_count_range=(config['SFC']['num_vnfs_min'], config['SFC']['num_vnfs_max'])
    )
    print(f"✓ Environment and SFC generator initialized\n")
    
    # ============================================================================
    # STEP 4: Lower Agent Pretraining (Phase 1)
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 4: Lower Agent Pretraining (Phase 1)")
    print("="*60)
    
    num_lower_episodes = config['TRAINING'].get('num_lower_pretrain_episodes', 500)
    trainer.pretrain_lower(env, num_episodes=num_lower_episodes)
    
    # ============================================================================
    # STEP 5: Upper Agent Pretraining (Phase 2)
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 5: Upper Agent Pretraining (Phase 2)")
    print("="*60)
    
    num_upper_episodes = config['TRAINING'].get('num_upper_pretrain_episodes', 1000)
    trainer.pretrain_upper(env, sfc_generator, num_episodes=num_upper_episodes)
    
    # ============================================================================
    # STEP 6: Main Training Loop (Phase 3)
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 6: Main Training Loop (Phase 3)")
    print("="*60)
    
    num_main_episodes = config['TRAINING'].get('num_episodes', 2000)
    trainer.train_main(env, sfc_generator, num_episodes=num_main_episodes)
    
    # ============================================================================
    # STEP 7: Evaluation
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 7: Final Evaluation")
    print("="*60)
    
    num_eval_requests = config['EVALUATION'].get('num_test_requests', 100)
    eval_results = evaluate_strategy(strategy, p_net, sfc_generator, 
                                     num_test_requests=num_eval_requests, 
                                     device=device)
    
    print("\nFinal Evaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
        logger.log(metric=f'eval_{key}', value=value)
    
    # ============================================================================
    # STEP 8: Save Logs and Checkpoints
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 8: Save Logs and Checkpoints")
    print("="*60)
    
    # Lưu training logs
    logs = trainer.get_logs()
    for metric_name, metric_values in logs.items():
        if metric_values:
            avg_value = np.mean(metric_values)
            logger.log(metric=f'train_{metric_name}', value=avg_value)
    
    logger.save()
    strategy.save_all_checkpoints('./checkpoints')
    
    print("✓ Training completed successfully!")
    
    return {
        'strategy': strategy,
        'trainer': trainer,
        'vgae_model': vgae_model,
        'upper_agent': strategy.upper_agent,
        'lower_agent': strategy.lower_agent,
        'p_net': p_net,
        'eval_results': eval_results,
        'logs': logs,
    }


if __name__ == '__main__':
    result = main()
