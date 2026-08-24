import torch
from pathlib import Path
from datetime import datetime
from data import NetworkGenerator, SFCRequestGenerator
from model import VGAE, DQNAgent
from algorithm import PlacementEngine, RoutingEngine
from utils import Logger
from .vgae_trainer import VGAETrainer
from .dqn_trainer import DQNTrainer


class TrainingPipeline:

    def __init__(self, config, output_dir='./outputs'):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(str(self.output_dir / 'logs'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def phase1_pretrain_vgae(self, num_p_nodes, num_pretrain_nets=5):
        print("\n" + "="*60)
        print("PHASE 1: VGAE Pretraining")
        print("="*60)

        print(f"Generating {num_pretrain_nets} physical networks for pretraining...")
        p_net_template = NetworkGenerator.generate_physical_network(
            num_p_nodes,
            self.config.PHYSICAL_NETWORK['topology'],
            self.config.PHYSICAL_NETWORK['cpu_capacity'],
            self.config.PHYSICAL_NETWORK['ram_capacity'],
            self.config.PHYSICAL_NETWORK['storage_capacity'],
            self.config.PHYSICAL_NETWORK['bandwidth_capacity']
        )

        pretrain_nets = [p_net_template.copy() for _ in range(num_pretrain_nets)]

        print(f"Initializing VGAE model...")
        vgae_model = VGAE(
            num_p_nodes,
            input_dim=4,
            hidden_dim=32,
            latent_dim=self.config.VGAE['embedding_dim']
        )
        vgae_model.to(self.device)

        trainer = VGAETrainer(vgae_model, self.config.VGAE)

        print(f"Starting VGAE training for {self.config.VGAE['num_epochs']} epochs...")
        trainer.train(pretrain_nets, self.config.VGAE['num_epochs'])

        checkpoint_path = self.output_dir / 'vgae_pretrained.pth'
        trainer.save(str(checkpoint_path))
        print(f"VGAE model saved to {checkpoint_path}")

        vgae_model.eval()
        return vgae_model, p_net_template

    def phase2_train_dqn(self, vgae_model, p_net, num_episodes=None):
        print("\n" + "="*60)
        print("PHASE 2: DQN Training")
        print("="*60)

        if num_episodes is None:
            num_episodes = self.config.TRAINING['num_episodes']

        p_net_features = torch.FloatTensor(p_net.get_node_features()).to(self.device)
        p_net_edges = p_net.get_edge_index().to(self.device)

        with torch.no_grad():
            z, _, _ = vgae_model.encode(p_net_features, p_net_edges)

        state_dim = len(z.flatten()) + 3 + self.config.VGAE['embedding_dim'] + 2

        print(f"State dimension: {state_dim}")
        print(f"Number of servers: {p_net.num_nodes}")

        dqn_agent = DQNAgent(state_dim, p_net.num_nodes, self.config.DQN)
        dqn_agent.q_network.to(self.device)
        dqn_agent.target_network.to(self.device)

        routing_engine = RoutingEngine()
        placement_engine = PlacementEngine(p_net, vgae_model, dqn_agent, routing_engine)

        sfc_generator = SFCRequestGenerator(
            cpu_range=self.config.SFC['cpu_requirement'],
            ram_range=self.config.SFC['ram_requirement'],
            storage_range=self.config.SFC['storage_requirement'],
            bw_range=self.config.SFC['bandwidth_requirement'],
            vnf_count_range=(self.config.SFC['num_vnfs_min'], self.config.SFC['num_vnfs_max'])
        )

        dqn_trainer = DQNTrainer(placement_engine, dqn_agent, self.config.TRAINING)

        print(f"Starting DQN training for {num_episodes} episodes...")
        dqn_trainer.train(num_episodes, sfc_generator)

        checkpoint_path = self.output_dir / 'dqn_trained.pth'
        dqn_trainer.save_model(str(checkpoint_path))
        print(f"DQN model saved to {checkpoint_path}")

        return dqn_agent, placement_engine, dqn_trainer

    def phase3_finetune(self, vgae_model, dqn_agent, placement_engine, 
                       num_episodes=None):
        print("\n" + "="*60)
        print("PHASE 3: Fine-tuning")
        print("="*60)

        if num_episodes is None:
            num_episodes = self.config.TRAINING.get('num_finetune_episodes', 50)

        sfc_generator = SFCRequestGenerator(
            cpu_range=self.config.SFC['cpu_requirement'],
            ram_range=self.config.SFC['ram_requirement'],
            storage_range=self.config.SFC['storage_requirement'],
            bw_range=self.config.SFC['bandwidth_requirement'],
            vnf_count_range=(self.config.SFC['num_vnfs_min'], self.config.SFC['num_vnfs_max'])
        )

        dqn_trainer = DQNTrainer(placement_engine, dqn_agent, self.config.TRAINING)

        print(f"Starting fine-tuning for {num_episodes} episodes...")
        dqn_trainer.train(num_episodes, sfc_generator)

        checkpoint_path = self.output_dir / 'dqn_finetuned.pth'
        dqn_trainer.save_model(str(checkpoint_path))
        print(f"Fine-tuned DQN model saved to {checkpoint_path}")

        return dqn_trainer

    def run_full_pipeline(self):
        print("\nStarting VGAE-DQN-SFCP Full Pipeline")
        print(f"Device: {self.device}")

        num_p_nodes = self.config.PHYSICAL_NETWORK['num_nodes']

        vgae_model, p_net = self.phase1_pretrain_vgae(num_p_nodes)

        dqn_agent, placement_engine, dqn_trainer = self.phase2_train_dqn(
            vgae_model, p_net)

        dqn_trainer_finetuned = self.phase3_finetune(
            vgae_model, dqn_agent, placement_engine)

        print("\n" + "="*60)
        print("Pipeline Complete!")
        print("="*60)

        return {
            'vgae_model': vgae_model,
            'dqn_agent': dqn_agent,
            'placement_engine': placement_engine,
            'dqn_trainer': dqn_trainer_finetuned,
            'p_net': p_net,
        }