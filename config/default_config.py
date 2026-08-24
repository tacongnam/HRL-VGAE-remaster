"""
Default Configuration for HRL-VGAE-Remaster
"""

PHYSICAL_NETWORK = {
    'num_nodes': 50,
    'topology': 'scale_free',  # 'scale_free', 'random', 'grid'
    'cpu_capacity': 200,
    'ram_capacity': 256,
    'storage_capacity': 512,
    'bandwidth_capacity': 1000,
}

SFC = {
    'num_vnfs_min': 3,
    'num_vnfs_max': 8,
    'cpu_requirement': (10, 50),
    'ram_requirement': (16, 128),
    'storage_requirement': (32, 256),
    'bandwidth_requirement': (10, 100),
}

VGAE = {
    'embedding_dim': 32,
    'gnn_layer': 2,
    'hidden_dim': 32,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'num_epochs': 100,
    'num_pretrain_nets': 5,
    'freeze_after_pretrain': True,
}

# Hierarchical RL Configuration
HRL = {
    # Upper Agent (Scheduler + Placer)
    'top_k_candidates': 8,           # Top-K requests from queue
    'max_queue_size': 100,           # Max pending requests
    'upper_hidden_dim': 128,         # Upper network hidden dimension
    'upper_gamma': 0.95,             # Discount factor
    'upper_lr_pretrain': 1e-3,       # Learning rate for pretraining
    'upper_lr_online': 1e-4,         # Learning rate for online training
    'upper_batch_size': 32,
    'upper_target_sync_freq': 50,    # Sync target network every N steps
    
    # Lower Agent (Load Balancer + Routing)
    'k_routes': 3,                   # K-shortest paths candidates
    'lower_hidden_dim': 64,          # Lower network hidden dimension
    'lower_gamma': 0.9,              # Discount factor (shorter horizon)
    'lower_lr_pretrain': 1e-3,       # Learning rate for pretraining
    'lower_lr_online': 5e-5,         # Learning rate for online training (finetune)
    'lower_batch_size': 32,
    'lower_target_sync_freq': 20,    # Sync faster (local problem)
    
    # Epsilon schedule (global, not reset per file)
    'epsilon_max': 0.5,
    'epsilon_min': 0.05,
    
    # Reward weights (FIXED, not learned)
    'w1_r2c': 1.0,                   # Revenue/cost weight
    'w2_sla_penalty': 0.3,           # SLA violation penalty
    'w3_reject_penalty': 1.0,        # Rejection penalty
}

# Training Configuration
TRAINING = {
    'num_lower_pretrain_episodes': 500,      # Phase 1
    'num_upper_pretrain_episodes': 1000,     # Phase 2
    'num_episodes': 2000,                    # Phase 3 (main training)
    'train_frequency': 10,                   # Train every N steps
    'checkpoint_dir': './checkpoints',
    'arrival_rate': 0.8,                     # Poisson arrival
}

# Evaluation Configuration
EVALUATION = {
    'num_test_requests': 100,
    'evaluation_frequency': 100,  # Evaluate every N episodes
}

# Old DQN config (kept for backward compatibility, not used in HRL)
DQN = {
    'embedding_dim': 64,
    'hidden_dim': 128,
    'learning_rate': 1e-3,
    'gamma': 0.95,
    'epsilon': 1.0,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.995,
    'target_update_freq': 500,
    'batch_size': 32,
    'replay_buffer_size': 10000,
}

# Logging and checkpoint
LOGGING = {
    'log_dir': './logs',
    'checkpoint_dir': './checkpoints',
    'verbose': True,
    'save_frequency': 100,  # Save checkpoint every N episodes
}

default_config = {
    'PHYSICAL_NETWORK': PHYSICAL_NETWORK,
    'SFC': SFC,
    'VGAE': VGAE,
    'HRL': HRL,
    'TRAINING': TRAINING,
    'EVALUATION': EVALUATION,
    'DQN': DQN,
    'LOGGING': LOGGING,
}
