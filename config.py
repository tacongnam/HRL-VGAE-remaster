from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class NetworkConfig:
    num_nodes: int = 50
    function_node_ratio: float = 0.30
    cpu_capacity_mips: float = 8000.0
    bw_min_mbps: float = 1000.0
    bw_max_mbps: float = 10000.0
    delay_min_ms: float = 2.0
    delay_max_ms: float = 5.0


@dataclass
class SFCConfig:
    arrival_rate: float = 20.0
    arrival_interval: int = 100
    vnf_len_min: int = 2
    vnf_len_max: int = 8
    cpu_req_min: float = 100.0
    cpu_req_max: float = 500.0
    bw_req_min: float = 10.0
    bw_req_max: float = 100.0
    deadline_min: int = 20
    deadline_max: int = 100


@dataclass
class VGAEConfig:
    d_in: int = 16
    d_hidden: int = 64
    d_latent: int = 32
    lr: float = 1e-3
    neg_sample_ratio: float = 1.0
    train_every_steps: int = 20
    temporal: bool = True
    temporal_hidden: int = 64


@dataclass
class QNetConfig:
    d_sfc: int = 7
    d_vnf: int = 2
    lr: float = 5e-4
    gamma: float = 0.95
    target_update_freq: int = 50
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995
    hl_buffer_size: int = 15_000
    ll_buffer_size: int = 72_000
    batch_size: int = 64
    max_q_vectors_per_action: int = 10


@dataclass
class DeployCostConfig:
    w_cpu: float = 1.0
    w_ram: float = 1.0
    w_storage: float = 1.0
    w_bw: float = 1.0
    w_init: float = 1.0
    ram_per_cpu_unit: float = 1.0
    storage_per_vnf: float = 1.0
    init_cost_per_hop: float = 0.5


@dataclass
class ParetoConfig:
    enabled: bool = True
    num_weight_bins: int = 11
    chebyshev_rho: float = 0.05
    accept_weight_init: float = 0.5
    cost_weight_init: float = 0.5
    weight_adapt_rate: float = 0.01
    utopia_momentum: float = 0.98
    front_log_every: int = 25
    hv_ref_cost: float = -1.0
    hv_ref_delay: float = -1.0
    hv_ref_balance: float = -1.0
    hv_ref_success: float = -1.0
    failure_penalty_cost: float = 10.0
    failure_penalty_delay: float = 10.0
    failure_penalty_balance: float = 10.0
    failure_penalty_success: float = 10.0

    def hv_reference_point(self) -> np.ndarray:
        return np.array(
            [
                self.hv_ref_cost,
                self.hv_ref_delay,
                self.hv_ref_balance,
                self.hv_ref_success,
            ],
            dtype=np.float64,
        )

    def failure_penalty_vector(self) -> np.ndarray:
        return np.array(
            [
                -self.failure_penalty_cost,
                -self.failure_penalty_delay,
                -self.failure_penalty_balance,
                -self.failure_penalty_success,
            ],
            dtype=np.float32,
        )


@dataclass
class RewardConfig:
    mu_cpu: float = 0.001
    mu_bw: float = 0.001
    theta: float = 2.0
    lambda_L: float = 1.0
    r_fail: float = 10.0
    alpha: float = 1.0
    beta: float = 0.5
    gamma_load: float = 0.5
    omega_bw: float = 100.0
    lambda_penalty: float = 1.0
    mu_deploy_cost: float = 0.001


@dataclass
class TrainConfig:
    total_steps: int = 200_000
    episode_horizon: int = 500
    log_interval: int = 1
    seed: int = 42
    max_sfc_per_timestep: "Optional[int]" = None


@dataclass
class Config:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sfc: SFCConfig = field(default_factory=SFCConfig)
    vgae: VGAEConfig = field(default_factory=VGAEConfig)
    qnet: QNetConfig = field(default_factory=QNetConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    deploy_cost: DeployCostConfig = field(default_factory=DeployCostConfig)
    pareto: ParetoConfig = field(default_factory=ParetoConfig)

    @property
    def d_latent(self):
        return self.vgae.d_latent

    @property
    def d_global(self):
        return 2 * self.vgae.d_latent

    @property
    def d_ll_input(self):
        return (
            self.d_global
            + 2 * self.vgae.d_latent
            + self.qnet.d_vnf
            + self.qnet.d_sfc
            + 2
        )

    @property
    def d_hl_input(self):
        return self.d_global + self.qnet.d_sfc + 2

    @property
    def sfc_quota_per_timestep(self) -> int:
        if self.train.max_sfc_per_timestep is not None:
            return max(1, self.train.max_sfc_per_timestep)
        import math

        return max(1, math.ceil(self.sfc.arrival_rate / self.sfc.arrival_interval))
