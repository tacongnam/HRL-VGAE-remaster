import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import argparse
import random
import time
import numpy as np
import torch
import torch.optim as optim
from typing import Optional, List
from config import Config
from env.nfv_env import NFVEnvironment
from models.vgae import MNVGAE
from agents.hl_agent import HLAgent
from agents.ll_agent import LLAgent
from utils.dijkstra import custom_dijkstra, compute_path_cost_delay
from utils.deploy_cost import total_deploy_cost
from utils.pareto import ParetoScalarizer
from models.ll_dqn import N_OBJ
from utils.metrics import MetricsTracker
from utils.logger import TrainingLogger
from data.loader import discover_episodes, parse_episode

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def encode_graph(vgae: MNVGAE, env: NFVEnvironment, device: torch.device, update_temporal: bool = True):
    x, ei = env.get_node_features_tensor(device)
    with torch.no_grad():
        z_nodes, z_global, mu, logvar = vgae(x, ei, graph_id=env.topology_id, update_temporal=update_temporal)
    return z_nodes.detach(), z_global.detach(), x, ei, mu, logvar

def build_vnf_feature(vnf, cfg: Config, device: torch.device) -> torch.Tensor:
    return torch.tensor([vnf.cpu_req / cfg.network.cpu_capacity_mips, 1.0], dtype=torch.float32, device=device)

def _compute_recon_loss_sampled(z_nodes: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, neg_ratio: float, device) -> torch.Tensor:
    num_pos = edge_index.shape[1]
    if num_pos == 0:
        return torch.tensor(0.0, device=device)
    pos_src = z_nodes[edge_index[0]]
    pos_dst = z_nodes[edge_index[1]]
    pos_score = (pos_src * pos_dst).sum(dim=-1)
    num_neg = max(1, int(num_pos * neg_ratio))
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_score = (z_nodes[neg_src] * z_nodes[neg_dst]).sum(dim=-1)
    pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    return pos_loss + neg_loss

class StepTimer:
    def __init__(self, log_every: int = 500):
        self.log_every = log_every
        self.counts = {}
        self.totals = {}
        self.sfc_count = 0
        self._t = {}

    def start(self, key: str):
        self._t[key] = time.perf_counter()

    def end(self, key: str):
        elapsed = time.perf_counter() - self._t.get(key, time.perf_counter())
        self.totals[key] = self.totals.get(key, 0.0) + elapsed
        self.counts[key] = self.counts.get(key, 0) + 1

    def tick_sfc(self):
        self.sfc_count += 1
        if self.sfc_count % self.log_every == 0:
            self.report()

    def report(self):
        parts = []
        for k in self.totals:
            avg_ms = self.totals[k] / max(1, self.counts[k]) * 1000
            parts.append(f"{k}={avg_ms:.2f}ms(x{self.counts[k]})")
        print(f"  [TIMER] SFC#{self.sfc_count} | " + " | ".join(parts), flush=True)

def run_episode(env: NFVEnvironment, vgae: MNVGAE, hl_agent: HLAgent, ll_agent: LLAgent, vgae_optimizer: Optional[optim.Optimizer],
    scalarizer: ParetoScalarizer, cfg: Config, device: torch.device, tracker: Optional[MetricsTracker] = None, train: bool = True,
    verbose: bool = False, fixed_weight: Optional[tuple] = None) -> dict:
    episode_hl_reward = 0.0
    episode_ll_reward = 0.0
    vgae_loss_acc = 0.0
    hl_loss_acc = 0.0
    ll_loss_acc = 0.0
    vgae_updates = 0
    hl_updates = 0
    ll_updates = 0
    sfc_processed_since_vgae_train = 0
    done = False
    timer = StepTimer(log_every=500) if verbose else None
    blocked_until: dict = {}
    if verbose:
        print(f"  [EP] start | queue={len(env.queue)} total_reqs={len(env._all_requests)}", flush=True)
    if fixed_weight is not None:
        w_accept, w_cost = fixed_weight
    elif train:
        w_accept, w_cost = scalarizer.sample_weight()
    else:
        w_accept, w_cost = scalarizer.w_accept, scalarizer.w_cost
    pareto_w = torch.tensor([w_accept, w_cost], dtype=torch.float32, device=device)
    if verbose: timer.start('encode_init')
    z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device)
    if verbose: timer.end('encode_init')
    graph_dirty = False
    while not done:
        if not env.queue:
            if verbose: timer.start('step_time')
            done = env.step_time()
            if verbose: timer.end('step_time')
            if not done and (graph_dirty or (env._dataset_mode and env._req_cursor < len(env._all_requests))):
                if verbose: timer.start('encode')
                z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device)
                if verbose: timer.end('encode')
                graph_dirty = False
            continue
        if train and vgae_optimizer is not None and sfc_processed_since_vgae_train >= cfg.vgae.train_every_steps:
            if verbose: timer.start('vgae_train')
            vgae_optimizer.zero_grad()
            vgae.train()
            z_nodes_tr, _, mu_tr, logvar_tr = vgae(x, ei, graph_id=env.topology_id, update_temporal=False)
            kl = vgae.kl_loss(mu_tr, logvar_tr)
            recon_loss = _compute_recon_loss_sampled(z_nodes_tr, ei, env.num_nodes, cfg.vgae.neg_sample_ratio, device)
            v_loss = recon_loss + kl
            v_loss.backward()
            vgae_optimizer.step()
            vgae_loss_acc += v_loss.item()
            vgae_updates += 1
            vgae.eval()
            with torch.no_grad():
                z_nodes, z_global, mu, logvar = vgae(x, ei, graph_id=env.topology_id, update_temporal=True)
            z_nodes = z_nodes.detach()
            z_global = z_global.detach()
            sfc_processed_since_vgae_train = 0
            if verbose: timer.end('vgae_train')
        if verbose: timer.start('hl_select')
        hl_result = hl_agent.select_sfc(z_global, env.queue, env.t, pareto_w, blocked_until)
        if verbose: timer.end('hl_select')
        if hl_result is None:
            if verbose: timer.start('step_time')
            done = env.step_time()
            if verbose: timer.end('step_time')
            if not done and graph_dirty:
                if verbose: timer.start('encode')
                z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device)
                if verbose: timer.end('encode')
                graph_dirty = False
            continue
        sfc_idx, selected_sfc = hl_result
        sfc_feat = torch.tensor(selected_sfc.to_feature_vector(env.t), dtype=torch.float32, device=device)
        ll_records = []
        partial_allocations = []
        embedding_failed = False
        current_source = selected_sfc.source
        for i in range(selected_sfc.F_k):
            vnf = selected_sfc.vnf_sequence[i]
            if verbose: timer.start('ll_mask')
            cpu_mask = env.build_ll_mask(vnf.cpu_req)
            if verbose: timer.end('ll_mask')
            z_prev = z_nodes[current_source].detach()
            vnf_feat = build_vnf_feature(vnf, cfg, device)
            if verbose: timer.start('ll_select')
            chosen_node = ll_agent.select_node(z_global, z_prev, z_nodes, vnf_feat, sfc_feat, cpu_mask, pareto_w)
            if verbose: timer.end('ll_select')
            if chosen_node is None:
                embedding_failed = True
                break
            if verbose: timer.start('dijkstra')
            path = custom_dijkstra(env.G, current_source, chosen_node, selected_sfc.bandwidth, cfg.reward.omega_bw, cfg.reward.lambda_penalty)
            if verbose: timer.end('dijkstra')
            if path is None or env.G.nodes[chosen_node]['cpu_free'] < vnf.cpu_req:
                embedding_failed = True
                break
            if verbose: timer.start('allocate')
            env.allocate(chosen_node, path, vnf.cpu_req, selected_sfc.bandwidth)
            if verbose: timer.end('allocate')
            partial_allocations.append((chosen_node, path, vnf.cpu_req, selected_sfc.bandwidth))
            path_cost, path_delay = compute_path_cost_delay(env.G, path)
            state_tuple = (z_global.cpu(), z_prev.cpu(), z_nodes[chosen_node].detach().cpu(), vnf_feat.cpu(), sfc_feat.cpu(), pareto_w.cpu())
            ll_records.append({'state': state_tuple, 'path_cost': path_cost, 'path_delay': path_delay, 'target_node': chosen_node, 'path': path, 'cpu_req': vnf.cpu_req, 'vnf_idx': i})
            current_source = chosen_node
        dest_path = None
        dest_cost = 0.0
        dest_delay = 0.0
        if not embedding_failed:
            if verbose: timer.start('dijkstra')
            dest_path = custom_dijkstra(env.G, current_source, selected_sfc.dest, selected_sfc.bandwidth, cfg.reward.omega_bw, cfg.reward.lambda_penalty)
            if verbose: timer.end('dijkstra')
            if dest_path is None:
                embedding_failed = True
            else:
                env.allocate(selected_sfc.dest, dest_path, 0.0, selected_sfc.bandwidth)
                partial_allocations.append((selected_sfc.dest, dest_path, 0.0, selected_sfc.bandwidth))
                dest_cost, dest_delay = compute_path_cost_delay(env.G, dest_path)
        if embedding_failed:
            if verbose: timer.start('rollback')
            env._rollback_resources(partial_allocations)
            if verbose: timer.end('rollback')
            r_H = scalarizer.scalarize(-cfg.reward.r_fail, 0.0, w_accept, w_cost)
            env.remove_sfc_from_queue(selected_sfc)
            if not selected_sfc.is_expired(env.t + 1):
                blocked_until[selected_sfc.sfc_id] = env.t + 1
                if not any(q.sfc_id == selected_sfc.sfc_id for q in env.queue):
                    env.queue.append(selected_sfc)
            else:
                env.rejected += 1
                env.total_attempted += 1
                blocked_until.pop(selected_sfc.sfc_id, None)
            if train and ll_records:
                J = len(ll_records)
                penalty_vec = np.array([-cfg.reward.r_fail / max(1, J), 0.0], dtype=np.float32)
                if verbose: timer.start('ll_train')
                for rec in ll_records:
                    ll_agent.store_transition(rec['state'], None, penalty_vec, 1.0)
                    loss = ll_agent.train_step(pareto_w)
                    if loss is not None:
                        ll_loss_acc += loss
                        ll_updates += 1
                if verbose: timer.end('ll_train')
        else:
            deploy_cost = total_deploy_cost(env.G, partial_allocations, cfg)
            if verbose: timer.start('commit')
            env.commit_sfc(selected_sfc, partial_allocations, deploy_cost=deploy_cost)
            if verbose: timer.end('commit')
            env.remove_sfc_from_queue(selected_sfc)
            graph_dirty = True
            sfc_processed_since_vgae_train += 1
            load_std = env.get_load_std()
            n_steps = selected_sfc.F_k + 1
            for rec in ll_records:
                r_quality = -(cfg.reward.alpha * rec['path_cost'] + cfg.reward.beta * rec['path_delay'] + cfg.reward.gamma_load * load_std)
                rec['r_quality'] = r_quality
            r_quality_dest = -(cfg.reward.alpha * dest_cost + cfg.reward.beta * dest_delay + cfg.reward.gamma_load * load_std)
            total_r_quality = sum(rec['r_quality'] for rec in ll_records) + r_quality_dest
            r_bar_quality = total_r_quality / n_steps
            revenue = env.norm_revenue(selected_sfc)
            r_cost_component = -cfg.reward.mu_deploy_cost * deploy_cost
            r_accept_component = revenue + cfg.reward.theta * selected_sfc.urgency(env.t) + cfg.reward.lambda_L * r_bar_quality
            r_H = scalarizer.scalarize(r_accept_component, r_cost_component, w_accept, w_cost)
            scalarizer.update_utopia(r_accept_component, r_cost_component)
            episode_ll_reward += r_bar_quality
            if tracker is not None:
                tracker.record_sfc_success(selected_sfc.F_k + 1, revenue)
            if train:
                if verbose: timer.start('ll_train')
                cost_per_step = deploy_cost / n_steps
                for k in range(len(ll_records)):
                    cur_rec = ll_records[k]
                    is_terminal = (k == len(ll_records) - 1)
                    reward_vec = np.array([cur_rec['r_quality'], -cfg.reward.mu_deploy_cost * cost_per_step], dtype=np.float32)
                    if is_terminal:
                        next_state = None
                    else:
                        next_vnf = selected_sfc.vnf_sequence[k + 1]
                        next_mask = env.build_ll_mask(next_vnf.cpu_req)
                        next_zp = z_nodes[cur_rec['target_node']].detach().cpu()
                        next_vf = build_vnf_feature(next_vnf, cfg, device).cpu()
                        next_state = (z_global.cpu(), next_zp, z_nodes.cpu(), next_vf, sfc_feat.cpu(), next_mask, pareto_w.cpu())
                    ll_agent.store_transition(cur_rec['state'], next_state, reward_vec, float(is_terminal))
                    loss = ll_agent.train_step(pareto_w)
                    if loss is not None:
                        ll_loss_acc += loss
                        ll_updates += 1
                if verbose: timer.end('ll_train')
        episode_hl_reward += r_H
        if env._dataset_mode:
            if verbose: timer.start('flush_arrivals')
            env._flush_arrivals()
            if verbose: timer.end('flush_arrivals')
        if train:
            if verbose: timer.start('hl_train')
            next_valid_sfcs = [q for q in env.queue if not q.is_expired(env.t)]
            next_sfc_feats = [torch.tensor(q.to_feature_vector(env.t), dtype=torch.float32, device=device) for q in next_valid_sfcs]
            hl_done = not env.queue and (not hasattr(env, '_req_cursor') or env._req_cursor >= len(env._all_requests))
            hl_agent.store_transition(z_global.cpu(), sfc_feat.cpu(), pareto_w.cpu(), sfc_idx, r_H, z_global.cpu(), [f.cpu() for f in next_sfc_feats], pareto_w.cpu(), hl_done)
            hl_loss = hl_agent.train_step()
            if hl_loss is not None:
                hl_loss_acc += hl_loss
                hl_updates += 1
            if verbose: timer.end('hl_train')
        if verbose:
            timer.tick_sfc()
        if env.queue:
            continue
        if verbose: timer.start('step_time')
        done = env.step_time()
        if verbose: timer.end('step_time')
        if not done and graph_dirty:
            if verbose: timer.start('encode')
            z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device)
            if verbose: timer.end('encode')
            graph_dirty = False
    if train and not fixed_weight:
        cost_norm = env.total_deploy_cost / max(1.0, env.accepted)
        cost_norm_01 = min(1.0, cost_norm / 1000.0)
        scalarizer.adapt_weights(env.acceptance_ratio(), cost_norm_01)
        scalarizer.w_accept = w_accept * 0.9 + scalarizer.w_accept * 0.1
        scalarizer.w_cost = 1.0 - scalarizer.w_accept
    if verbose:
        print(f"  [EP] done | acc={env.accepted} rej={env.rejected} t={env.t}", flush=True)
        if timer:
            timer.report()
    return {'hl_reward': episode_hl_reward, 'll_reward': episode_ll_reward, 'acceptance_ratio': env.acceptance_ratio(),
        'accepted': env.accepted, 'rejected': env.rejected, 'vgae_loss': vgae_loss_acc / max(1, vgae_updates),
        'hl_loss': hl_loss_acc / max(1, hl_updates), 'll_loss': ll_loss_acc / max(1, ll_updates),
        'epsilon_hl': hl_agent.epsilon, 'epsilon_ll': ll_agent.epsilon, 'total_deploy_cost': env.total_deploy_cost,
        'w_accept': w_accept, 'w_cost': w_cost}

def _build_env_and_agents(cfg, device):
    vgae = MNVGAE(cfg).to(device)
    hl_agent = HLAgent(cfg, device)
    ll_agent = LLAgent(cfg, device)
    vgae_optimizer = optim.Adam(vgae.parameters(), lr=cfg.vgae.lr)
    scalarizer = ParetoScalarizer(cfg)
    return vgae, hl_agent, ll_agent, vgae_optimizer, scalarizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, default='data/train')
    parser.add_argument('--test-dir', type=str, default='data/test')
    parser.add_argument('--checkpoint', type=str, default='vgae_hrl_ql_checkpoint.pt')
    parser.add_argument('--log-dir', type=str, default='runs')
    parser.add_argument('--csv', type=str, default='training_log.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--passes-per-file', type=int, default=5)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    cfg = Config()
    set_seeds(cfg.train.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_paths = discover_episodes(args.train_dir)
    test_paths = discover_episodes(args.test_dir) if args.test_dir else []
    if not train_paths:
        raise FileNotFoundError(f"Không tìm thấy episode nào trong: {args.train_dir}")
    env = NFVEnvironment(cfg)
    vgae, hl_agent, ll_agent, vgae_optimizer, scalarizer = _build_env_and_agents(cfg, device)
    tracker = MetricsTracker()
    global_ep = 0
    warmup_episodes = args.warmup_epochs * len(train_paths) * args.passes_per_file
    total_train_episodes = args.epochs * len(train_paths) * args.passes_per_file
    decay_episodes = int((total_train_episodes - warmup_episodes) * 0.75)
    eps_decay_per_ep = (cfg.qnet.eps_end / cfg.qnet.eps_start) ** (1.0 / max(1, decay_episodes))
    with TrainingLogger(log_dir=args.log_dir, csv_path=args.csv) as logger:
        for epoch in range(1, args.epochs + 1):
            epoch_paths = train_paths.copy()
            random.shuffle(epoch_paths)
            is_warmup = (epoch <= args.warmup_epochs)
            for path in epoch_paths:
                G, reqs, _, topo_id = parse_episode(path)
                for pass_idx in range(args.passes_per_file):
                    global_ep += 1
                    if is_warmup:
                        hl_agent.epsilon = 1.0
                        ll_agent.epsilon = 1.0
                    env.reset(G, reqs, topology_id=topo_id)
                    ep_start = time.perf_counter()
                    verbose_this = args.verbose and global_ep == 1
                    print(f"  [EP {global_ep}] starting... reqs={len(reqs)}", flush=True)
                    stats = run_episode(env, vgae, hl_agent, ll_agent, vgae_optimizer, scalarizer, cfg, device, tracker=tracker, train=True, verbose=verbose_this)
                    ep_time = time.perf_counter() - ep_start
                    if not is_warmup:
                        hl_agent.epsilon = max(cfg.qnet.eps_end, hl_agent.epsilon * eps_decay_per_ep)
                        ll_agent.epsilon = max(cfg.qnet.eps_end, ll_agent.epsilon * eps_decay_per_ep)
                    stats['epsilon_hl'] = hl_agent.epsilon
                    stats['epsilon_ll'] = ll_agent.epsilon
                    m = tracker.flush_episode(global_ep, env, stats)
                    logger.log(m)
                    phase = 'WARMUP' if is_warmup else 'TRAIN'
                    print(f"[{phase}] Ep {global_ep:5d} | Epoch {epoch}/{args.epochs} | {os.path.basename(path)} p{pass_idx+1} | "
                        f"AccRatio={m.acceptance_ratio:.3f} | acc={m.accepted} rej={m.rejected} | HL_R={m.hl_reward:.2f} LL_R={m.ll_reward:.2f} | "
                        f"eps={m.epsilon:.3f} | Rev={m.total_revenue:.2f} | DeployCost={m.total_deploy_cost:.2f} | "
                        f"w=({m.w_accept:.2f},{m.w_cost:.2f}) | time={ep_time:.1f}s", flush=True)
    torch.save({'vgae': vgae.state_dict(), 'hl_q': hl_agent.q_net.state_dict(), 'll_q': ll_agent.q_net.state_dict(),
        'hl_eps': hl_agent.epsilon, 'll_eps': ll_agent.epsilon, 'w_accept': scalarizer.w_accept, 'w_cost': scalarizer.w_cost}, args.checkpoint)
    if test_paths:
        hl_agent.epsilon = 0.0
        ll_agent.epsilon = 0.0  
        vgae.eval()
        ratios, revs, costs = [], [], []
        for ep, path in enumerate(test_paths, 1):
            G, reqs, _, topo_id = parse_episode(path)
            env.reset(G, reqs, topology_id=topo_id)
            stats = run_episode(env, vgae, hl_agent, ll_agent, None, scalarizer, cfg, device, tracker=None, train=False)
            ratios.append(stats['acceptance_ratio'])
            revs.append(stats['hl_reward'])
            costs.append(stats['total_deploy_cost'])
            print(f"  Test Ep {ep:3d} [{os.path.basename(path)}] | AccRatio={stats['acceptance_ratio']:.3f} | acc={stats['accepted']} rej={stats['rejected']} | Cost={stats['total_deploy_cost']:.2f}")
        print(f"\n=== Test Summary ===")
        print(f"  AccRatio: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
        print(f"  HLReward: {np.mean(revs):.4f} ± {np.std(revs):.4f}")
        print(f"  DeployCost: {np.mean(costs):.4f} ± {np.std(costs):.4f}")

if __name__ == '__main__':
    main()