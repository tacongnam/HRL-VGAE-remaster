from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class EpisodeMetrics:
    episode: int = 0
    hl_reward: float = 0.0
    ll_reward: float = 0.0
    vgae_loss: float = 0.0
    hl_loss: float = 0.0
    ll_loss: float = 0.0
    accepted: int = 0
    rejected: int = 0
    acceptance_ratio: float = 0.0
    avg_hops: float = 0.0
    avg_link_util: float = 0.0
    avg_cpu_util: float = 0.0
    total_revenue: float = 0.0
    total_deploy_cost: float = 0.0
    epsilon: float = 1.0
    w_accept: float = 0.5
    w_cost: float = 0.5


class MetricsTracker:
    def __init__(self):
        self.history: List[EpisodeMetrics] = []
        self._hop_buf: List[float] = []
        self._revenue_buf: List[float] = []

    def record_sfc_success(self, num_hops: int, revenue: float):
        self._hop_buf.append(float(num_hops))
        self._revenue_buf.append(revenue)

    def flush_episode(self, ep: int, env, stats: dict) -> EpisodeMetrics:
        link_utils = []
        for u, v, e in env.G.edges(data=True):
            link_utils.append(1.0 - e["bw_free"] / e["bw_total"])
        cpu_utils = []
        for u in env.G.nodes():
            nd = env.G.nodes[u]
            if nd.get("cpu_total", 0) > 0:
                cpu_utils.append(1.0 - nd["cpu_free"] / nd["cpu_total"])
        m = EpisodeMetrics(
            episode=ep,
            hl_reward=stats["hl_reward"],
            ll_reward=stats["ll_reward"],
            vgae_loss=stats["vgae_loss"],
            hl_loss=stats["hl_loss"],
            ll_loss=stats["ll_loss"],
            accepted=stats["accepted"],
            rejected=stats["rejected"],
            acceptance_ratio=stats["acceptance_ratio"],
            avg_hops=float(np.mean(self._hop_buf)) if self._hop_buf else 0.0,
            avg_link_util=float(np.mean(link_utils)) if link_utils else 0.0,
            avg_cpu_util=float(np.mean(cpu_utils)) if cpu_utils else 0.0,
            total_revenue=float(np.sum(self._revenue_buf)),
            total_deploy_cost=stats.get("total_deploy_cost", 0.0),
            epsilon=stats.get("epsilon_hl", 1.0),
            w_accept=stats.get("w_accept", 0.5),
            w_cost=stats.get("w_cost", 0.5),
        )
        self.history.append(m)
        self._hop_buf.clear()
        self._revenue_buf.clear()
        return m

    def last_n_mean(self, key: str, n: int = 50) -> float:
        if not self.history:
            return 0.0
        window = self.history[-n:]
        return float(np.mean([getattr(m, key) for m in window]))

    CSV_HEADER = (
        "episode,hl_reward,ll_reward,vgae_loss,hl_loss,ll_loss,accepted,rejected,acceptance_ratio,"
        "avg_hops,avg_link_util,avg_cpu_util,total_revenue,total_deploy_cost,epsilon,w_accept,w_cost\n"
    )
