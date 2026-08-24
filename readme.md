# README.md

```markdown
# HRL-VGAE: Deep Reinforcement Learning for Online VNF Placement and Routing

## Overview

HRL-VGAE is a deep reinforcement learning approach for solving the online Virtual Network Function (VNF) placement and routing problem in NFV networks. The system combines:

- **VGAE (Variational Graph Auto-Encoder)**: Learns topology-aware representations of the physical network
- **DQN (Deep Q-Network)**: Makes sequential server selection decisions for VNF placement
- **Dijkstra Algorithm**: Finds feasible routing paths between selected servers

### Key Features

- Online SFC placement without offline precomputation
- Topology-aware network representation using VGAE
- Scalable action space through candidate masking
- Resource and bandwidth constraint handling
- Automatic rollback on placement failure
- Modular architecture for easy extension

### Architecture

```
Physical Network + Resources
        ↓
    VGAE Encoding
        ↓
Topology-Aware Latent Representation
        ↓
    DQN Agent
        ↓
  Server Selection
        ↓
    Dijkstra Routing
        ↓
Feasibility Checking
        ↓
Success/Failure
```

## Installation

### Requirements

- Python 3.7+
- PyTorch >= 1.9.0
- PyTorch Geometric >= 2.0.0
- NetworkX >= 2.6
- NumPy, SciPy, Pandas, Matplotlib

### Setup

```bash
# Clone repository
git clone <repository-url>
cd HRL-VGAE

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import torch_geometric; print('Installation successful')"
```

## Quick Start

### Full Pipeline Training

Run the complete training pipeline (VGAE pretraining → DQN training → Fine-tuning):

```bash
python main.py
```

### Individual Training Phases

**Phase 1: VGAE Pretraining**
```bash
python scripts/train_phase1_vgae.py
```

**Phase 2: DQN Training**
```bash
python scripts/train_phase2_dqn.py
```

**Phase 3: Fine-tuning**
```bash
python scripts/train_phase3_finetune.py
```

### Evaluation

```bash
python scripts/evaluate.py
```

### Ablation Study

Compare VGAE-DQN with DQN using raw features:

```bash
python scripts/ablation.py
```

### Baseline Comparison

Compare against heuristic baselines (Random, First-Fit, Best-Fit):

```bash
python scripts/baseline_comparison.py
```

## Configuration

Edit `config/default_config.py` to customize parameters:

```python
# Physical Network
PHYSICAL_NETWORK = {
    'num_nodes': 50,
    'topology': 'scale_free',  # 'scale_free', 'random', 'grid'
    'cpu_capacity': 200,
    'ram_capacity': 256,
    'storage_capacity': 512,
    'bandwidth_capacity': 1000,
}

# SFC Requests
SFC = {
    'num_vnfs_min': 3,
    'num_vnfs_max': 8,
    'cpu_requirement': (10, 50),
    'ram_requirement': (16, 128),
    'storage_requirement': (32, 256),
    'bandwidth_requirement': (10, 100),
}

# VGAE Model
VGAE = {
    'embedding_dim': 64,
    'gnn_layer': 2,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'num_epochs': 50,
}

# DQN Agent
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

# Training
TRAINING = {
    'num_episodes': 100,
    'num_requests_per_episode': 50,
    'arrival_rate': 0.8,
}
```

## Project Structure

```
HRL-VGAE/
├── config/                      # Configuration files
│   ├── default_config.py
│   └── __init__.py
│
├── data/                        # Data generation and loading
│   ├── generator.py            # Network and SFC generation
│   ├── loader.py               # File I/O
│   ├── simulator.py            # Event simulator
│   └── __init__.py
│
├── environment/                 # Environment components
│   ├── physical_network.py      # Physical network model
│   ├── sfc_request.py           # SFC request model
│   ├── network_state.py         # Network state management
│   ├── vne_environment.py       # Main environment
│   ├── constraints.py           # Constraint checking
│   └── __init__.py
│
├── model/                       # Neural network models
│   ├── vgae/                    # VGAE model
│   │   ├── vgae.py
│   │   └── __init__.py
│   ├── dqn/                     # DQN model
│   │   ├── network.py
│   │   ├── agent.py
│   │   ├── replay_buffer.py
│   │   └── __init__.py
│   ├── utils/                   # Model utilities
│   │   ├── graph_utils.py
│   │   ├── normalization.py
│   │   └── __init__.py
│   └── __init__.py
│
├── algorithm/                   # Placement and routing algorithms
│   ├── routing.py              # Dijkstra routing
│   ├── placement.py            # Sequential placement
│   ├── candidate_mask.py       # Candidate filtering
│   ├── reward.py               # Reward computation
│   └── __init__.py
│
├── trainer/                     # Training modules
│   ├── vgae_trainer.py         # VGAE trainer
│   ├── dqn_trainer.py          # DQN trainer
│   ├── pipeline.py             # Training pipeline
│   └── __init__.py
│
├── utils/                       # Utility functions
│   ├── metric.py               # Metrics calculation
│   ├── logger.py               # Logging
│   ├── checkpoint.py           # Model checkpointing
│   ├── visualization.py        # Plotting and visualization
│   └── __init__.py
│
├── scripts/                     # Training and evaluation scripts
│   ├── train_phase1_vgae.py
│   ├── train_phase2_dqn.py
│   ├── train_phase3_finetune.py
│   ├── evaluate.py
│   ├── ablation.py
│   ├── baseline_comparison.py
│   └── __init__.py
│
├── test/                        # Unit and integration tests
│   ├── test_vgae.py
│   ├── test_dqn.py
│   ├── test_routing.py
│   ├── test_placement.py
│   ├── test_constraints.py
│   ├── test_end_to_end.py
│   └── __init__.py
│
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE
```

## Core Components

### 1. Physical Network (`environment/physical_network.py`)

Represents the substrate network with:
- Node resources (CPU, RAM, Storage)
- Link resources (Bandwidth, Latency)
- Resource allocation/deallocation

```python
from environment import PhysicalNetwork

p_net = PhysicalNetwork(num_nodes=50, topology='scale_free')
p_net.allocate_resource(node_id=0, cpu_req=20, ram_req=32, storage_req=64)
```

### 2. SFC Request (`environment/sfc_request.py`)

Represents a service function chain request:
- VNF list with resource requirements
- Virtual link bandwidth requirements
- Temporal information (arrival, duration, departure)

```python
from environment import VNF, SFCRequest

vnfs = [VNF(0, cpu_req=20, ram_req=32, storage_req=64),
        VNF(1, cpu_req=30, ram_req=48, storage_req=96)]
sfc = SFCRequest(request_id=0, vnfs=vnfs, link_bandwidth_reqs=[50, 75])
```

### 3. VGAE Model (`model/vgae/vgae.py`)

Encodes physical network topology and resources into latent embeddings:

```python
from model import VGAE

vgae = VGAE(num_nodes=50, input_dim=4, hidden_dim=32, latent_dim=64)
z, mu, logvar = vgae.encode(node_features, edge_index)
```

### 4. DQN Agent (`model/dqn/agent.py`)

Selects servers for VNF placement using Q-learning:

```python
from model import DQNAgent

dqn_agent = DQNAgent(state_dim=256, num_servers=50, config=dqn_config)
action = dqn_agent.select_action(state, candidate_mask)
```

### 5. Placement Engine (`algorithm/placement.py`)

Orchestrates sequential VNF placement:

```python
from algorithm import PlacementEngine

placement_engine = PlacementEngine(p_net, vgae, dqn_agent, routing_engine)
result = placement_engine.place_sfc(sfc_request, training=True)
```

## Usage Examples

### Example 1: Train Full Pipeline

```python
from trainer.pipeline import TrainingPipeline
from config import default_config

pipeline = TrainingPipeline(default_config)
result = pipeline.run_full_pipeline()

vgae_model = result['vgae_model']
dqn_agent = result['dqn_agent']
placement_engine = result['placement_engine']
```

### Example 2: Evaluate Model

```python
from data import SFCRequestGenerator
from utils import MetricCalculator

sfc_generator = SFCRequestGenerator()
requests = sfc_generator.generate(100)

results = []
for req in requests:
    result = placement_engine.place_sfc(req, training=False, use_epsilon_greedy=False)
    results.append(result)

metrics = MetricCalculator.get_detailed_metrics(results, p_net)
print(f"Acceptance Ratio: {metrics['acceptance_ratio']:.4f}")
print(f"Resource Utilization: {metrics['resource_utilization']:.4f}")
```

### Example 3: Custom Training

```python
from trainer import VGAETrainer, DQNTrainer
from config import default_config

# Train VGAE
vgae_trainer = VGAETrainer(vgae_model, default_config.VGAE)
vgae_trainer.train(pretrain_nets, num_epochs=50)

# Train DQN
dqn_trainer = DQNTrainer(placement_engine, dqn_agent, default_config.TRAINING)
dqn_trainer.train(num_episodes=100, sfc_generator=sfc_generator)
```

## Key Algorithms

### VGAE Encoding

```
Input: Node features X, Edge index E
  ↓
GCN Encoder: μ = GCN_μ(X, E), log σ = GCN_σ(X, E)
  ↓
Reparameterization: z_i = μ_i + σ_i ⊙ ε, ε ~ N(0, I)
  ↓
Output: Latent embeddings Z
```

### DQN Training

```
for each episode:
  for each SFC request:
    for each VNF:
      1. Get feasible candidates
      2. Compute state representation
      3. Select action via ε-greedy: a = argmax Q(s, a) or random
      4. Execute action (allocate resources, find path)
      5. Receive reward
      6. Store transition in replay buffer
      7. Sample minibatch and update Q-network
      8. Update target network periodically
```

### Sequential Placement

```
For each VNF in SFC:
  1. Check resource feasibility → Create resource mask
  2. If VNF > 0: Check routing feasibility → Create routing mask
  3. Combine masks (AND operation)
  4. If no feasible candidate → Rollback and reject
  5. DQN selects server from feasible set
  6. Dijkstra finds path (if needed)
  7. Allocate resources and bandwidth
  8. Proceed to next VNF
If all VNFs placed → Accept request
Else → Rollback entire request
```

## Performance Metrics

The system computes the following metrics:

- **Acceptance Ratio (SAR)**: Percentage of accepted requests
- **Resource Utilization**: CPU/RAM/Storage usage across network
- **Deployment Cost**: Total resource consumption
- **Load Imbalance**: Variance in node loads
- **Average Path Length**: Mean path length for routing
- **Average Reward**: Mean reward per placement

## Comparison with Baselines

The system is compared against:

1. **Random Placement**: Randomly select feasible servers
2. **First-Fit**: Select first feasible server
3. **Best-Fit**: Select server with minimum remaining resources
4. **DRL-SFCP (A3C)**: Original implementation with Actor-Critic
5. **HRL-VGAE Original**: Hierarchical admission + placement

## Testing

Run all tests:

```bash
python -m pytest test/ -v
```

Run specific test module:

```bash
python -m pytest test/test_vgae.py -v
python -m pytest test/test_dqn.py -v
python -m pytest test/test_placement.py -v
python -m pytest test/test_end_to_end.py -v
```

Run with coverage:

```bash
python -m pytest test/ --cov=. --cov-report=html
```

## Outputs

After running, the system generates:

- **Checkpoints**: `outputs/vgae_pretrained.pth`, `outputs/dqn_trained.pth`
- **Logs**: `outputs/logs/` directory with training curves
- **Results**: `outputs/evaluation_results.txt`
- **Comparison**: `outputs/baseline_comparison.txt`
- **Ablation**: `outputs/ablation_results.txt`

## Visualization

Generate training curves:

```python
from utils import Visualizer
from trainer import DQNTrainer

trainer = DQNTrainer(...)
trainer.train(num_episodes, sfc_generator)

metrics = trainer.get_metrics()
Visualizer.plot_training_curves(metrics, save_path='outputs/training_curves.png')
```

Plot network topology with placement:

```python
Visualizer.plot_network_graph(p_net, placement_result, 
                             save_path='outputs/network.png')
```

Plot resource utilization:

```python
Visualizer.plot_resource_utilization(p_net, 
                                     save_path='outputs/resource_util.png')
```

## Hyperparameter Tuning

Key hyperparameters to tune:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| embedding_dim | 64 | 32-128 | VGAE/DQN embedding dimension |
| learning_rate (VGAE) | 1e-3 | 1e-5 - 1e-2 | VGAE optimizer learning rate |
| learning_rate (DQN) | 1e-3 | 1e-5 - 1e-2 | DQN optimizer learning rate |
| gamma | 0.95 | 0.9-0.99 | Discount factor |
| epsilon_decay | 0.995 | 0.99-0.999 | Exploration decay rate |
| batch_size | 32 | 16-64 | Training batch size |
| target_update_freq | 500 | 100-1000 | Target network update frequency |

## Known Limitations

1. **Scalability**: Current implementation tested up to 100 nodes
2. **Routing Complexity**: Dijkstra adds O(E log V) per placement
3. **State Representation**: High-dimensional state might need feature selection
4. **Cold Start**: VGAE requires pretraining before DQN training

## Future Improvements

- [ ] Support for VNF migration
- [ ] Multi-objective optimization (cost, latency, load balance)
- [ ] Attention-based path selection
- [ ] Graph neural networks for end-to-end learning
- [ ] Distributed training for large networks
- [ ] Real-world dataset benchmarks

## License

This project is licensed under the Apache License 2.0 - see LICENSE file for details.

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the maintainers.

## Acknowledgments

- Original DRL-SFCP paper: Wang et al., ICC 2021
- VGAE inspiration from Kipf & Welling, VGAE paper
- Implementation based on PyTorch and PyTorch Geometric

## References

1. Wang, T., et al. "DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning." ICC 2021.
2. Kipf, T., & Welling, M. "Variational Graph Auto-Encoders." ICLR 2016 Workshop.
3. Mnih, V., et al. "Human-level control through deep reinforcement learning." Nature 2015.
4. Kipf, T., & Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017.

---

**Last Updated**: 2024
**Version**: 1.0.0
**Status**: Active Development
```