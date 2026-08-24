PHYSICAL_NETWORK = {
    'num_nodes': 50,
    'topology': 'scale_free', #'scale_free', 'random', 'grid'
    'cpu_capacity': 200,
    'ram_capacity': 256,
    'storage_capacity': 512,
    'bandwidth_capacity': 1000,
}

SFC = {
    'num_vnfs_min': 3,
    'num_vnfs_max': 8,
    'cpu_requirements': (10, 50),
    'ram_requirements': (16, 128),
    'storage_requirements': (32, 256),
    'bandwidth_requirements': (10, 100),
}

VGAE = {
    'embedding_dim': 64,
    'gnn_layer': 2,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'num_epochs': 50,
}

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

TRAINING = {
    'num_episodes': 1000,
    'num_requests_per_episode': 100,
    'arrival_rate': 0.8,  # Poisson arrival
    'duration_distribution': 'exponential',  # Holding time
}