import os
import csv
from utils.metrics import EpisodeMetrics

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

class TrainingLogger:
    def __init__(self, log_dir: str = "runs", csv_path: str = "training_log.csv", use_tb: bool = True):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = csv_path
        self.tb_writer = None
        if use_tb and _TB_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=log_dir)
        self._csv_file = open(csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(['episode', 'hl_reward', 'll_reward', 'vgae_loss', 'hl_loss', 'll_loss', 'accepted', 'rejected',
            'acceptance_ratio', 'avg_hops', 'avg_link_util', 'avg_cpu_util', 'total_revenue', 'total_deploy_cost', 'epsilon', 'w_accept', 'w_cost'])

    def log(self, m: EpisodeMetrics):
        self._csv_writer.writerow([m.episode, round(m.hl_reward, 4), round(m.ll_reward, 4), round(m.vgae_loss, 6), round(m.hl_loss, 6),
            round(m.ll_loss, 6), m.accepted, m.rejected, round(m.acceptance_ratio, 4), round(m.avg_hops, 2), round(m.avg_link_util, 4),
            round(m.avg_cpu_util, 4), round(m.total_revenue, 4), round(m.total_deploy_cost, 4), round(m.epsilon, 4),
            round(m.w_accept, 4), round(m.w_cost, 4)])
        self._csv_file.flush()
        if self.tb_writer is not None:
            ep = m.episode
            self.tb_writer.add_scalar('Reward/HL', m.hl_reward, ep)
            self.tb_writer.add_scalar('Reward/LL', m.ll_reward, ep)
            self.tb_writer.add_scalar('Loss/VGAE', m.vgae_loss, ep)
            self.tb_writer.add_scalar('Loss/HL_DQN', m.hl_loss, ep)
            self.tb_writer.add_scalar('Loss/LL_DQN', m.ll_loss, ep)
            self.tb_writer.add_scalar('SFC/AcceptanceRatio', m.acceptance_ratio, ep)
            self.tb_writer.add_scalar('SFC/Accepted', m.accepted, ep)
            self.tb_writer.add_scalar('SFC/Rejected', m.rejected, ep)
            self.tb_writer.add_scalar('Network/AvgLinkUtil', m.avg_link_util, ep)
            self.tb_writer.add_scalar('Network/AvgCpuUtil', m.avg_cpu_util, ep)
            self.tb_writer.add_scalar('Revenue/Total', m.total_revenue, ep)
            self.tb_writer.add_scalar('Cost/Deploy', m.total_deploy_cost, ep)
            self.tb_writer.add_scalar('Train/Epsilon', m.epsilon, ep)
            self.tb_writer.add_scalar('Pareto/WAccept', m.w_accept, ep)
            self.tb_writer.add_scalar('Pareto/WCost', m.w_cost, ep)

    def close(self):
        self._csv_file.close()
        if self.tb_writer is not None:
            self.tb_writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
