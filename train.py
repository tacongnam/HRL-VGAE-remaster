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
from nfv_env import NFVEnvironment
from vgae import MNVGAE
from hl_agent import HLAgent
from ll_agent import LLAgent
from dijkstra import custom_dijkstra, compute_path_cost_delay
from deploy_cost import total_deploy_cost
from pareto import ParetoScalarizer, ParetoArchive
from ll_dqn import N_OBJ
from metrics import MetricsTracker
from logger import TrainingLogger


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_graph(vgae, env, device, update_temporal=True):
    x, ei = env.get_node_features_tensor(device)
    with torch.no_grad():
        z_nodes, z_global, mu, logvar = vgae(
            x, ei, graph_id=env.topology_id, update_temporal=update_temporal)
    return z_nodes.detach(), z_global.detach(), x, ei, mu, logvar


def build_vnf_feature(vnf, cfg, device):
    return torch.tensor(
        [vnf.cpu_req / cfg.network.cpu_capacity_mips, 1.0],
        dtype=torch.float32, device=device)


def _compute_recon_loss_sampled(z_nodes, edge_index, num_nodes, neg_ratio, device):
    import torch.nn.functional as F
    num_pos = edge_index.shape[1]
    if num_pos == 0:
        return torch.tensor(0.0, device=device)
    pos_score = (z_nodes[edge_index[0]] * z_nodes[edge_index[1]]).sum(dim=-1)
    num_neg = max(1, int(num_pos * neg_ratio))
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_score = (z_nodes[neg_src] * z_nodes[neg_dst]).sum(dim=-1)
    return (F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
            + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score)))


def _build_reward_vec_4obj(r_cost: float, r_delay: float,
                            r_balance: float, r_success: float) -> np.ndarray:
    return np.array([r_cost, r_delay, r_balance, r_success], dtype=np.float32)


def run_episode(env: NFVEnvironment,
                vgae,
                hl_agent: HLAgent,
                ll_agent: LLAgent,
                vgae_optimizer,
                scalarizer: ParetoScalarizer,
                cfg: Config,
                device: torch.device,
                pareto_archive=None,
                tracker=None,
                train: bool = True,
                verbose: bool = False,
                fixed_weight=None) -> dict:

    episode_hl_reward = 0.0
    episode_ll_reward = 0.0
    episode_r_accept = 0.0
    episode_r_cost = 0.0
    vgae_loss_acc = 0.0
    hl_loss_acc = 0.0
    ll_loss_acc = 0.0
    vgae_updates = 0
    hl_updates = 0
    ll_updates = 0
    sfc_processed_since_vgae_train = 0
    done = False
    blocked_until: dict = {}
    failure_vec = cfg.pareto.failure_penalty_vector()
    ref_point = cfg.pareto.hv_reference_point()

    if fixed_weight is not None:
        w_accept, w_cost = fixed_weight
    elif train:
        w_accept, w_cost = scalarizer.sample_weight()
    else:
        w_accept, w_cost = scalarizer.w_accept, scalarizer.w_cost
    pareto_w = torch.tensor([w_accept, w_cost], dtype=torch.float32, device=device)

    z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device)
    graph_dirty = False

    while not done:
        if not env.queue:
            done = env.step_time()
            if not done and (graph_dirty or (env._dataset_mode
                             and env._req_cursor < len(env._all_requests))):
                z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device)
                graph_dirty = False
            continue

        if (train and vgae_optimizer is not None
                and sfc_processed_since_vgae_train >= cfg.vgae.train_every_steps):
            vgae_optimizer.zero_grad()
            vgae.train()
            z_nodes_tr, _, mu_tr, logvar_tr = vgae(
                x, ei, graph_id=env.topology_id, update_temporal=False)
            kl = vgae.kl_loss(mu_tr, logvar_tr)
            recon_loss = _compute_recon_loss_sampled(
                z_nodes_tr, ei, env.num_nodes, cfg.vgae.neg_sample_ratio, device)
            v_loss = recon_loss + kl
            v_loss.backward()
            vgae_optimizer.step()
            vgae_loss_acc += v_loss.item()
            vgae_updates += 1
            vgae.eval()
            with torch.no_grad():
                z_nodes, z_global, mu, logvar = vgae(
                    x, ei, graph_id=env.topology_id, update_temporal=True)
            z_nodes = z_nodes.detach()
            z_global = z_global.detach()
            sfc_processed_since_vgae_train = 0

        hl_result = hl_agent.select_sfc(
            z_global, env.queue, env.t, pareto_w, blocked_until, verbose=verbose)
        if hl_result is None:
            done = env.step_time()
            if not done and graph_dirty:
                z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device)
                graph_dirty = False
            continue

        sfc_idx, selected_sfc = hl_result
        sfc_feat = torch.tensor(
            selected_sfc.to_feature_vector(env.t),
            dtype=torch.float32, device=device)

        ll_records = []
        partial_allocations = []
        embedding_failed = False
        current_source = selected_sfc.source

        for i in range(selected_sfc.F_k):
            vnf = selected_sfc.vnf_sequence[i]
            cpu_mask = env.build_ll_mask(vnf.cpu_req)
            z_prev = z_nodes[current_source].detach()
            vnf_feat = build_vnf_feature(vnf, cfg, device)

            chosen_node = ll_agent.select_node(
                z_global, z_prev, z_nodes, vnf_feat, sfc_feat,
                cpu_mask, pareto_w, verbose=verbose)
            if chosen_node is None:
                embedding_failed = True
                break

            path = custom_dijkstra(
                env.G, current_source, chosen_node,
                selected_sfc.bandwidth,
                cfg.reward.omega_bw, cfg.reward.lambda_penalty)
            if path is None or env.G.nodes[chosen_node]['cpu_free'] < vnf.cpu_req:
                embedding_failed = True
                break

            env.allocate(chosen_node, path, vnf.cpu_req, selected_sfc.bandwidth)
            partial_allocations.append(
                (chosen_node, path, vnf.cpu_req, selected_sfc.bandwidth))

            path_cost, path_delay = compute_path_cost_delay(env.G, path)
            state_tuple = (
                z_global.cpu(), z_prev.cpu(),
                z_nodes[chosen_node].detach().cpu(),
                vnf_feat.cpu(), sfc_feat.cpu(), pareto_w.cpu()
            )
            ll_records.append({
                'state': state_tuple,
                'path_cost': path_cost,
                'path_delay': path_delay,
                'target_node': chosen_node,
                'path': path,
                'cpu_req': vnf.cpu_req,
                'vnf_idx': i
            })
            current_source = chosen_node

        dest_path = None
        dest_cost = 0.0
        dest_delay = 0.0
        if not embedding_failed:
            dest_path = custom_dijkstra(
                env.G, current_source, selected_sfc.dest,
                selected_sfc.bandwidth,
                cfg.reward.omega_bw, cfg.reward.lambda_penalty)
            if dest_path is None:
                embedding_failed = True
            else:
                env.allocate(selected_sfc.dest, dest_path, 0.0, selected_sfc.bandwidth)
                partial_allocations.append(
                    (selected_sfc.dest, dest_path, 0.0, selected_sfc.bandwidth))
                dest_cost, dest_delay = compute_path_cost_delay(env.G, dest_path)

        if embedding_failed:
            env._rollback_resources(partial_allocations)
            hl_reward_vec = failure_vec.copy()
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
                ll_fail_vec = failure_vec / max(1, J)
                for rec in ll_records:
                    ll_agent.store_transition(rec['state'], None, ll_fail_vec, 1.0)
                    loss = ll_agent.train_step(pareto_w)
                    if loss is not None:
                        ll_loss_acc += loss
                        ll_updates += 1

            if verbose:
                print(f"  [FAIL] sfc={selected_sfc.sfc_id} "
                      f"reward_vec={hl_reward_vec} r_H={r_H:.4f}")

        else:
            deploy_cost = total_deploy_cost(env.G, partial_allocations, cfg)
            env.commit_sfc(selected_sfc, partial_allocations, deploy_cost=deploy_cost)
            env.remove_sfc_from_queue(selected_sfc)
            graph_dirty = True
            sfc_processed_since_vgae_train += 1

            load_std = env.get_load_std()
            n_steps = selected_sfc.F_k + 1

            for rec in ll_records:
                rec['r_quality'] = -(cfg.reward.alpha * rec['path_cost']
                                     + cfg.reward.beta * rec['path_delay']
                                     + cfg.reward.gamma_load * load_std)
            r_quality_dest = -(cfg.reward.alpha * dest_cost
                               + cfg.reward.beta * dest_delay
                               + cfg.reward.gamma_load * load_std)
            total_r_quality = (sum(rec['r_quality'] for rec in ll_records)
                               + r_quality_dest)
            r_bar_quality = total_r_quality / n_steps

            revenue = env.norm_revenue(selected_sfc)
            r_accept_component = (revenue
                                  + cfg.reward.theta * selected_sfc.urgency(env.t)
                                  + cfg.reward.lambda_L * r_bar_quality)
            r_cost_component = -cfg.reward.mu_deploy_cost * deploy_cost
            r_balance_component = -load_std
            r_success_component = 1.0

            hl_reward_vec = _build_reward_vec_4obj(
                r_cost_component, r_bar_quality,
                r_balance_component, r_success_component)

            r_H = scalarizer.scalarize(r_accept_component, r_cost_component,
                                       w_accept, w_cost)
            scalarizer.update_utopia(r_accept_component, r_cost_component)

            episode_r_accept += r_accept_component
            episode_r_cost += r_cost_component
            episode_ll_reward += r_bar_quality

            if verbose:
                print(f"  [OK] sfc={selected_sfc.sfc_id} "
                      f"cost={deploy_cost:.2f} "
                      f"reward_vec={np.round(hl_reward_vec, 4)} "
                      f"r_H={r_H:.4f} ref={ref_point}")

            if tracker is not None:
                tracker.record_sfc_success(selected_sfc.F_k + 1, revenue)

            if train:
                cost_per_step = deploy_cost / n_steps
                for k in range(len(ll_records)):
                    cur_rec = ll_records[k]
                    is_terminal = (k == len(ll_records) - 1)
                    step_cost = -cfg.reward.mu_deploy_cost * cost_per_step
                    reward_vec = _build_reward_vec_4obj(
                        step_cost,
                        cur_rec['r_quality'],
                        -load_std,
                        1.0 if is_terminal else 0.0
                    )
                    if is_terminal:
                        next_state = None
                    else:
                        next_vnf = selected_sfc.vnf_sequence[k + 1]
                        next_mask = env.build_ll_mask(next_vnf.cpu_req)
                        next_zp = z_nodes[cur_rec['target_node']].detach().cpu()
                        next_vf = build_vnf_feature(next_vnf, cfg, device).cpu()
                        next_state = (
                            z_global.cpu(), next_zp, z_nodes.cpu(),
                            next_vf, sfc_feat.cpu(), next_mask, pareto_w.cpu()
                        )
                    ll_agent.store_transition(
                        cur_rec['state'], next_state, reward_vec, float(is_terminal))
                    loss = ll_agent.train_step(pareto_w)
                    if loss is not None:
                        ll_loss_acc += loss
                        ll_updates += 1

        episode_hl_reward += r_H

        if env._dataset_mode:
            env._flush_arrivals()

        if train:
            next_valid_sfcs = [q for q in env.queue if not q.is_expired(env.t)]
            next_sfc_feats = [
                torch.tensor(q.to_feature_vector(env.t),
                             dtype=torch.float32, device=device)
                for q in next_valid_sfcs
            ]
            next_valid_mask = np.ones(len(next_sfc_feats), dtype=bool)
            hl_done = (not env.queue
                       and (not hasattr(env, '_req_cursor')
                            or env._req_cursor >= len(env._all_requests)))

            hl_agent.store_transition(
                z_global.cpu(), sfc_feat.cpu(), pareto_w.cpu(),
                sfc_idx,
                hl_reward_vec,
                z_global.cpu(),
                [f.cpu() for f in next_sfc_feats],
                pareto_w.cpu(),
                float(hl_done),
                valid_action_mask=next_valid_mask,
            )
            hl_loss = hl_agent.train_step()
            if hl_loss is not None:
                hl_loss_acc += hl_loss
                hl_updates += 1

        if env.queue:
            continue

        done = env.step_time()
        if not done and graph_dirty:
            z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device)
            graph_dirty = False

    if pareto_archive is not None and env.accepted > 0:
        avg_r_accept = episode_r_accept / max(1, env.accepted)
        avg_r_cost = episode_r_cost / max(1, env.accepted)
        pareto_archive.add(
            np.array([avg_r_accept, avg_r_cost], dtype=np.float64),
            episode=getattr(env, '_episode_id', 0),
            w_accept=w_accept, w_cost=w_cost,
            acceptance_ratio=env.acceptance_ratio(),
            total_deploy_cost=env.total_deploy_cost,
        )

    if train and not fixed_weight:
        cost_norm = env.total_deploy_cost / max(1.0, env.accepted)
        cost_norm_01 = min(1.0, cost_norm / 1000.0)
        scalarizer.adapt_weights(env.acceptance_ratio(), cost_norm_01)
        scalarizer.w_accept = w_accept * 0.9 + scalarizer.w_accept * 0.1
        scalarizer.w_cost = 1.0 - scalarizer.w_accept

    return {
        'hl_reward': episode_hl_reward,
        'll_reward': episode_ll_reward,
        'r_accept': episode_r_accept,
        'r_cost': episode_r_cost,
        'acceptance_ratio': env.acceptance_ratio(),
        'accepted': env.accepted,
        'rejected': env.rejected,
        'vgae_loss': vgae_loss_acc / max(1, vgae_updates),
        'hl_loss': hl_loss_acc / max(1, hl_updates),
        'll_loss': ll_loss_acc / max(1, ll_updates),
        'epsilon_hl': hl_agent.epsilon,
        'epsilon_ll': ll_agent.epsilon,
        'total_deploy_cost': env.total_deploy_cost,
        'w_accept': w_accept,
        'w_cost': w_cost,
        'pareto_archive_size': len(pareto_archive) if pareto_archive else 0,
        'hv_ref_point': ref_point.tolist(),
    }


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
    parser.add_argument('--test-dir', type=str, default='data/test1')
    parser.add_argument('--checkpoint', type=str, default='vgae_hrl_ql_checkpoint.pt')
    parser.add_argument('--log-dir', type=str, default='runs')
    parser.add_argument('--csv', type=str, default='training_log.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--passes-per-file', type=int, default=5)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--archive-size', type=int, default=200)
    args = parser.parse_args()

    cfg = Config()
    set_seeds(cfg.train.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        from data.loader import discover_episodes, parse_episode
        train_paths = discover_episodes(args.train_dir)
        test_paths = discover_episodes(args.test_dir) if args.test_dir else []
    except Exception:
        train_paths = []
        test_paths = []

    if not train_paths:
        raise FileNotFoundError(f"No episodes found: {args.train_dir}")

    env = NFVEnvironment(cfg)
    vgae, hl_agent, ll_agent, vgae_optimizer, scalarizer = _build_env_and_agents(cfg, device)
    tracker = MetricsTracker()
    pareto_archive = ParetoArchive(max_size=args.archive_size)

    global_ep = 0
    warmup_episodes = args.warmup_epochs * len(train_paths) * args.passes_per_file
    total_train_episodes = args.epochs * len(train_paths) * args.passes_per_file
    decay_episodes = int((total_train_episodes - warmup_episodes) * 0.75)
    eps_decay_per_ep = (cfg.qnet.eps_end / cfg.qnet.eps_start) ** (
        1.0 / max(1, decay_episodes))

    ref_point_str = str(cfg.pareto.hv_reference_point().tolist())
    print(f"HV reference_point = {ref_point_str}")
    print(f"Objectives: [r_cost, r_delay, r_balance, r_success] — all maximize")
    print(f"Placement: Pareto Q-set + Hypervolume action selection")
    print(f"Admission: Scalar scalarizer (Chebyshev) — unchanged")

    with TrainingLogger(log_dir=args.log_dir, csv_path=args.csv) as logger:
        for epoch in range(1, args.epochs + 1):
            epoch_paths = train_paths.copy()
            random.shuffle(epoch_paths)
            is_warmup = (epoch <= args.warmup_epochs)

            for path in epoch_paths:
                from data.loader import parse_episode
                G, reqs, _, topo_id = parse_episode(path)
                for pass_idx in range(args.passes_per_file):
                    global_ep += 1
                    if is_warmup:
                        hl_agent.epsilon = 1.0
                        ll_agent.epsilon = 1.0

                    env.reset(G, reqs, topology_id=topo_id)
                    env._episode_id = global_ep

                    ep_start = time.perf_counter()
                    verbose_this = args.verbose and (global_ep <= 2)

                    stats = run_episode(
                        env, vgae, hl_agent, ll_agent, vgae_optimizer,
                        scalarizer, cfg, device,
                        pareto_archive=pareto_archive,
                        tracker=tracker, train=True, verbose=verbose_this)

                    ep_time = time.perf_counter() - ep_start
                    if not is_warmup:
                        hl_agent.epsilon = max(cfg.qnet.eps_end,
                                               hl_agent.epsilon * eps_decay_per_ep)
                        ll_agent.epsilon = max(cfg.qnet.eps_end,
                                               ll_agent.epsilon * eps_decay_per_ep)

                    stats['epsilon_hl'] = hl_agent.epsilon
                    stats['epsilon_ll'] = ll_agent.epsilon
                    m = tracker.flush_episode(global_ep, env, stats)
                    logger.log(m)

                    phase = 'WARMUP' if is_warmup else 'TRAIN'
                    print(
                        f"[{phase}] Ep {global_ep:5d} | Epoch {epoch}/{args.epochs} | "
                        f"AccRatio={m.acceptance_ratio:.3f} | "
                        f"acc={m.accepted} rej={m.rejected} | "
                        f"HL_R={m.hl_reward:.2f} LL_R={m.ll_reward:.2f} | "
                        f"eps={m.epsilon:.3f} | DeployCost={m.total_deploy_cost:.2f} | "
                        f"Archive={stats['pareto_archive_size']} | time={ep_time:.1f}s",
                        flush=True)

                    if global_ep % cfg.pareto.front_log_every == 0:
                        print(f"  {pareto_archive.summary()}", flush=True)

    torch.save({
        'vgae': vgae.state_dict(),
        'hl_q': hl_agent.q_net.state_dict(),
        'll_q': ll_agent.q_net.state_dict(),
        'hl_eps': hl_agent.epsilon,
        'll_eps': ll_agent.epsilon,
        'w_accept': scalarizer.w_accept,
        'w_cost': scalarizer.w_cost,
        'hv_ref_point': cfg.pareto.hv_reference_point().tolist(),
        'pareto_archive': [
            {'obj': e['obj'].tolist(),
             **{k: v for k, v in e.items() if k != 'obj'}}
            for e in pareto_archive.get_front()
        ],
    }, args.checkpoint)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Final {pareto_archive.summary()}")


if __name__ == '__main__':
    main()