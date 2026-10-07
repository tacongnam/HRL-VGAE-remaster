import argparse
import random
import time
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional

from config import Config
from env import NFVEnvironment
from models import MNVGAE
from agents import HLAgent, LLAgent
from utils import (
    custom_dijkstra,
    compute_path_cost_delay,
    total_deploy_cost,
    ParetoScalarizer,
    ParetoArchive,
    MetricsTracker,
    TrainingLogger,
)
from data import discover_episodes, parse_episode


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_graph(vgae, env, device, update_temporal=True):
    x, ei = env.get_node_features_tensor(device)
    with torch.inference_mode():
        z_nodes, z_global, mu, logvar = vgae(
            x, ei, graph_id=env.topology_id, update_temporal=update_temporal
        )
    return z_nodes.detach(), z_global.detach(), x, ei, mu, logvar


def build_vnf_feature(vnf, cfg, device):
    return torch.tensor(
        [vnf.cpu_req / cfg.network.cpu_capacity_mips, 1.0],
        dtype=torch.float32,
        device=device,
    )


def _compute_recon_loss_sampled(z_nodes, edge_index, num_nodes, neg_ratio, device):
    num_pos = edge_index.shape[1]
    if num_pos == 0:
        return torch.tensor(0.0, device=device)
    pos_score = (z_nodes[edge_index[0]] * z_nodes[edge_index[1]]).sum(dim=-1)
    num_neg = max(1, int(num_pos * neg_ratio))
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_score = (z_nodes[neg_src] * z_nodes[neg_dst]).sum(dim=-1)
    return F.binary_cross_entropy_with_logits(
        pos_score, torch.ones_like(pos_score)
    ) + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))


def _build_reward_vec_4obj(
    r_cost: float, r_delay: float, r_balance: float, r_success: float
) -> np.ndarray:
    return np.array([r_cost, r_delay, r_balance, r_success], dtype=np.float32)


def load_pretrained_vgae(
    vgae: MNVGAE, checkpoint_path: str, device: torch.device, cfg: Config = None
) -> Optional[optim.Optimizer]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["vgae"] if "vgae" in ckpt else ckpt
    vgae.load_state_dict(state_dict)

    _STATIC_PREFIXES = ("gcn0.", "skip.", "gcn_mu.", "gcn_logvar.")
    temporal_params = []
    for name, param in vgae.named_parameters():
        if any(name.startswith(p) for p in _STATIC_PREFIXES):
            param.requires_grad_(False)
        elif name.startswith("temporal_cell"):
            temporal_params.append(param)

    print(
        f"[pretrain] Loaded VGAE from {checkpoint_path} | Trainable temporal params: {len(temporal_params)}"
    )
    return optim.Adam(temporal_params, lr=1e-3) if temporal_params else None


def run_episode(
    env: NFVEnvironment,
    vgae: MNVGAE,
    hl_agent: HLAgent,
    ll_agent: LLAgent,
    vgae_optimizer: Optional[optim.Optimizer],
    scalarizer: ParetoScalarizer,
    cfg: Config,
    device: torch.device,
    pareto_archive: Optional[ParetoArchive] = None,
    tracker: Optional[MetricsTracker] = None,
    train: bool = True,
    verbose: bool = False,
    fixed_weight: Optional[tuple] = None,
) -> dict:
    ep_hl_r, ep_ll_r, ep_r_acc, ep_r_cost = 0.0, 0.0, 0.0, 0.0
    vgae_loss_acc, hl_loss_acc, ll_loss_acc = 0.0, 0.0, 0.0
    vgae_updates, hl_updates, ll_updates = 0, 0, 0
    sfc_vgae_count, sfc_total_count, sfc_step_count = 0, 0, 0
    done = False
    blocked_until = {}
    failure_vec = cfg.pareto.failure_penalty_vector()

    if vgae is not None:
        vgae.reset_temporal_state(env.topology_id)

    w_accept, w_cost = (
        fixed_weight
        if fixed_weight
        else (
            scalarizer.sample_weight()
            if train
            else (scalarizer.w_accept, scalarizer.w_cost)
        )
    )
    pareto_w = torch.tensor([w_accept, w_cost], dtype=torch.float32, device=device)

    z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device)
    graph_dirty = False
    z_nodes_cpu = z_nodes.cpu()

    while not done:
        if not env.queue:
            done = env.step_time()
            if not done and (
                graph_dirty
                or (env._dataset_mode and env._req_cursor < len(env._all_requests))
            ):
                z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device)
                z_nodes_cpu, graph_dirty = z_nodes.cpu(), False
            continue

        if (
            train
            and vgae_optimizer is not None
            and sfc_vgae_count >= cfg.vgae.train_every_steps
        ):
            vgae_optimizer.zero_grad()
            vgae.train()
            _is_pretrained = not any(
                p.requires_grad
                for n, p in vgae.named_parameters()
                if any(
                    n.startswith(s)
                    for s in ("gcn0.", "skip.", "gcn_mu.", "gcn_logvar.")
                )
            )
            z_nodes_tr, z_global_tr, mu_tr, logvar_tr = vgae(
                x, ei, graph_id=env.topology_id, update_temporal=_is_pretrained
            )
            v_loss = _compute_recon_loss_sampled(
                z_nodes_tr, ei, env.num_nodes, cfg.vgae.neg_sample_ratio, device
            ) + vgae.kl_loss(mu_tr, logvar_tr)
            if _is_pretrained:
                v_loss += z_global_tr.pow(2).mean() * 1e-4
            v_loss.backward()
            vgae_optimizer.step()
            vgae_loss_acc += v_loss.item()
            vgae_updates += 1
            vgae.eval()
            with torch.inference_mode():
                z_nodes, z_global, mu, logvar = vgae(
                    x, ei, graph_id=env.topology_id, update_temporal=True
                )
            z_nodes, z_global = z_nodes.detach(), z_global.detach()
            z_nodes_cpu, sfc_vgae_count = z_nodes.cpu(), 0

        hl_res = hl_agent.select_sfc(
            z_global, env.queue, env.t, pareto_w, blocked_until, verbose=verbose
        )
        if hl_res is None:
            done = env.step_time()
            if not done and graph_dirty:
                z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device)
                z_nodes_cpu, graph_dirty = z_nodes.cpu(), False
            continue

        sfc_idx, selected_sfc = hl_res
        sfc_feat = torch.tensor(
            selected_sfc.to_feature_vector(env.t), dtype=torch.float32, device=device
        )
        ll_records, partial_allocations, embedding_failed = [], [], False
        curr_src = selected_sfc.source
        ll_masks = [env.build_ll_mask(v.cpu_req) for v in selected_sfc.vnf_sequence]

        for i in range(selected_sfc.F_k):
            vnf = selected_sfc.vnf_sequence[i]
            z_prev = z_nodes[curr_src].detach()
            vnf_feat = build_vnf_feature(vnf, cfg, device)

            chosen_node = ll_agent.select_node(
                z_global,
                z_prev,
                z_nodes,
                vnf_feat,
                sfc_feat,
                ll_masks[i],
                pareto_w,
                current_source=curr_src,
                G=env.G,
                k_hop=cfg.train.k_hop_explore,
                verbose=verbose,
            )
            if chosen_node is None:
                embedding_failed = True
                break

            path = custom_dijkstra(
                env.G,
                curr_src,
                chosen_node,
                selected_sfc.bandwidth,
                cfg.reward.omega_bw,
                cfg.reward.lambda_penalty,
            )
            if path is None or env.G.nodes[chosen_node]["cpu_free"] < vnf.cpu_req:
                embedding_failed = True
                break

            env.allocate(chosen_node, path, vnf.cpu_req, selected_sfc.bandwidth)
            partial_allocations.append(
                (chosen_node, path, vnf.cpu_req, selected_sfc.bandwidth)
            )
            p_cost, p_delay = compute_path_cost_delay(env.G, path)
            ll_records.append(
                {
                    "state": (
                        z_global.cpu(),
                        z_prev.cpu(),
                        z_nodes_cpu[chosen_node],
                        vnf_feat.cpu(),
                        sfc_feat.cpu(),
                        pareto_w.cpu(),
                    ),
                    "path_cost": p_cost,
                    "path_delay": p_delay,
                    "target_node": chosen_node,
                }
            )
            curr_src = chosen_node

            path = custom_dijkstra(
                env.G,
                curr_src,
                chosen_node,
                selected_sfc.bandwidth,
                cfg.reward.omega_bw,
                cfg.reward.lambda_penalty,
            )
            if path is None or env.G.nodes[chosen_node]["cpu_free"] < vnf.cpu_req:
                embedding_failed = True
                break

            p_cost, p_delay = compute_path_cost_delay(env.G, path)
            ll_records[-1]["path_cost"] = p_cost
            ll_records[-1]["path_delay"] = p_delay
            env.allocate(chosen_node, path, vnf.cpu_req, selected_sfc.bandwidth)
            partial_allocations.append(
                (chosen_node, path, vnf.cpu_req, selected_sfc.bandwidth)
            )
            curr_src = chosen_node

        dest_cost, dest_delay = 0.0, 0.0
        if not embedding_failed:
            dest_path = custom_dijkstra(
                env.G,
                curr_src,
                selected_sfc.dest,
                selected_sfc.bandwidth,
                cfg.reward.omega_bw,
                cfg.reward.lambda_penalty,
            )
            if dest_path is None:
                embedding_failed = True
            else:
                env.allocate(selected_sfc.dest, dest_path, 0.0, selected_sfc.bandwidth)
                partial_allocations.append(
                    (selected_sfc.dest, dest_path, 0.0, selected_sfc.bandwidth)
                )
                dest_cost, dest_delay = compute_path_cost_delay(env.G, dest_path)

        sfc_total_count += 1

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
                ll_fail_vec = failure_vec / max(1, len(ll_records))
                for rec in ll_records:
                    ll_agent.store_transition(rec["state"], None, ll_fail_vec, 1.0)
        else:
            deploy_cost = total_deploy_cost(env.G, partial_allocations, cfg)
            env.commit_sfc(selected_sfc, partial_allocations, deploy_cost=deploy_cost)
            env.remove_sfc_from_queue(selected_sfc)
            graph_dirty, sfc_vgae_count = True, sfc_vgae_count + 1

            load_std = env.get_load_std()
            n_steps = selected_sfc.F_k + 1
            for rec in ll_records:
                rec["r_quality"] = -(
                    cfg.reward.alpha * rec["path_cost"]
                    + cfg.reward.beta * rec["path_delay"]
                    + cfg.reward.gamma_load * load_std
                )
            r_dest = -(
                cfg.reward.alpha * dest_cost
                + cfg.reward.beta * dest_delay
                + cfg.reward.gamma_load * load_std
            )
            r_bar_quality = (sum(r["r_quality"] for r in ll_records) + r_dest) / n_steps

            revenue = env.norm_revenue(selected_sfc)
            r_acc_c = (
                revenue
                + cfg.reward.theta * selected_sfc.urgency(env.t)
                + cfg.reward.lambda_L * r_bar_quality
            )
            r_cost_c = -cfg.reward.mu_deploy_cost * deploy_cost
            hl_reward_vec = _build_reward_vec_4obj(
                r_cost_c, r_bar_quality, -load_std, 1.0
            )

            r_H = scalarizer.scalarize(r_acc_c, r_cost_c, w_accept, w_cost)
            scalarizer.update_utopia(r_acc_c, r_cost_c)
            ep_r_acc += r_acc_c
            ep_r_cost += r_cost_c
            ep_ll_r += r_bar_quality

            if tracker:
                tracker.record_sfc_success(selected_sfc.F_k + 1, revenue)

            if train:
                cost_per_step = deploy_cost / n_steps
                for k, cur_rec in enumerate(ll_records):
                    is_term = k == len(ll_records) - 1
                    r_vec = _build_reward_vec_4obj(
                        -cfg.reward.mu_deploy_cost * cost_per_step,
                        cur_rec["r_quality"],
                        -load_std,
                        1.0 if is_term else 0.0,
                    )
                    next_s = (
                        None
                        if is_term
                        else (
                            z_global.cpu(),
                            z_nodes_cpu[cur_rec["target_node"]],
                            z_nodes_cpu,
                            build_vnf_feature(
                                selected_sfc.vnf_sequence[k + 1], cfg, device
                            ).cpu(),
                            sfc_feat.cpu(),
                            ll_masks[k + 1],
                            pareto_w.cpu(),
                        )
                    )
                    ll_agent.store_transition(
                        cur_rec["state"], next_s, r_vec, float(is_term)
                    )

        ep_hl_r += r_H
        if env._dataset_mode:
            env._flush_arrivals()

        if train:
            next_sfcs = [q for q in env.queue if not q.is_expired(env.t)]
            next_feats = [
                torch.tensor(
                    q.to_feature_vector(env.t), dtype=torch.float32, device=device
                )
                for q in next_sfcs
            ]
            hl_done = not env.queue and (
                not hasattr(env, "_req_cursor")
                or env._req_cursor >= len(env._all_requests)
            )
            hl_agent.store_transition(
                z_global.cpu(),
                sfc_feat.cpu(),
                pareto_w.cpu(),
                sfc_idx,
                hl_reward_vec,
                z_global.cpu(),
                [f.cpu() for f in next_feats],
                pareto_w.cpu(),
                float(hl_done),
                valid_action_mask=np.ones(len(next_feats), dtype=bool),
            )

        if train and (sfc_total_count % cfg.train.train_interval_sfc == 0):
            loss_ll = ll_agent.train_step()
            if loss_ll is not None:
                ll_loss_acc += loss_ll
                ll_updates += 1
            loss_hl = hl_agent.train_step()
            if loss_hl is not None:
                hl_loss_acc += loss_hl
                hl_updates += 1

        sfc_step_count += 1
        if env.queue and sfc_step_count < cfg.sfc_quota_per_timestep:
            continue

        sfc_step_count = 0
        done = env.step_time()
        if not done and graph_dirty:
            z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device)
            z_nodes_cpu, graph_dirty = z_nodes.cpu(), False

    if pareto_archive is not None and env.accepted > 0:
        pareto_archive.add(
            np.array(
                [ep_r_acc / max(1, env.accepted), ep_r_cost / max(1, env.accepted)],
                dtype=np.float64,
            ),
            episode=getattr(env, "_episode_id", 0),
            w_accept=w_accept,
            w_cost=w_cost,
            acceptance_ratio=env.acceptance_ratio(),
            total_deploy_cost=env.total_deploy_cost,
        )

    if train and not fixed_weight:
        scalarizer.adapt_weights(
            env.acceptance_ratio(),
            min(1.0, (env.total_deploy_cost / max(1.0, env.accepted)) / 1000.0),
        )
        scalarizer.w_accept = w_accept * 0.9 + scalarizer.w_accept * 0.1
        scalarizer.w_cost = 1.0 - scalarizer.w_accept

    return {
        "hl_reward": ep_hl_r,
        "ll_reward": ep_ll_r,
        "acceptance_ratio": env.acceptance_ratio(),
        "accepted": env.accepted,
        "rejected": env.rejected,
        "vgae_loss": vgae_loss_acc / max(1, vgae_updates),
        "hl_loss": hl_loss_acc / max(1, hl_updates),
        "ll_loss": ll_loss_acc / max(1, ll_updates),
        "hl_updates": hl_updates,
        "ll_updates": ll_updates,
        "epsilon_hl": hl_agent.epsilon,
        "epsilon_ll": ll_agent.epsilon,
        "total_deploy_cost": env.total_deploy_cost,
        "w_accept": w_accept,
        "w_cost": w_cost,
        "pareto_archive_size": len(pareto_archive) if pareto_archive else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="data/train")
    parser.add_argument("--test-dir", type=str, default="data/test1")
    parser.add_argument("--checkpoint", type=str, default="vgae_hrl_ql_checkpoint.pt")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--csv", type=str, default="training_log.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--passes-per-file", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--archive-size", type=int, default=200)
    parser.add_argument("--train-file", type=str, default="")
    parser.add_argument("--pretrained-vgae", type=str, default="")
    args = parser.parse_args()

    cfg = Config()
    set_seeds(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths = (
        [args.train_file] if args.train_file else discover_episodes(args.train_dir)
    )
    if not train_paths:
        raise FileNotFoundError("No training files found.")

    env = NFVEnvironment(cfg)
    vgae = MNVGAE(cfg).to(device)
    hl_agent = HLAgent(cfg, device)
    ll_agent = LLAgent(cfg, device)
    vgae_optimizer = (
        load_pretrained_vgae(vgae, args.pretrained_vgae, device, cfg)
        if args.pretrained_vgae
        else optim.Adam(vgae.parameters(), lr=cfg.vgae.lr)
    )
    scalarizer = ParetoScalarizer(cfg)

    tracker = MetricsTracker()
    pareto_archive = ParetoArchive(max_size=args.archive_size)

    global_ep = 0
    total_train_episodes = args.epochs * len(train_paths) * args.passes_per_file
    warmup_episodes = args.warmup_epochs * len(train_paths) * args.passes_per_file
    decay_episodes = int((total_train_episodes - warmup_episodes) * 0.75)
    eps_decay_per_ep = (cfg.qnet.eps_end / cfg.qnet.eps_start) ** (
        1.0 / max(1, decay_episodes)
    )

    with TrainingLogger(log_dir=args.log_dir, csv_path=args.csv) as logger:
        for epoch in range(1, args.epochs + 1):
            epoch_paths = train_paths.copy()
            random.shuffle(epoch_paths)
            is_warmup = epoch <= args.warmup_epochs

            for path in epoch_paths:
                G, reqs, _, topo_id = parse_episode(path)
                for _ in range(args.passes_per_file):
                    global_ep += 1
                    if is_warmup:
                        hl_agent.epsilon, ll_agent.epsilon = 1.0, 1.0

                    env.reset(G, reqs, topology_id=topo_id)
                    env._episode_id = global_ep
                    ep_start = time.perf_counter()

                    stats = run_episode(
                        env,
                        vgae,
                        hl_agent,
                        ll_agent,
                        vgae_optimizer,
                        scalarizer,
                        cfg,
                        device,
                        pareto_archive=pareto_archive,
                        tracker=tracker,
                        train=True,
                        verbose=args.verbose and global_ep <= 2,
                    )

                    if not is_warmup:
                        hl_agent.epsilon = max(
                            cfg.qnet.eps_end, hl_agent.epsilon * eps_decay_per_ep
                        )
                        ll_agent.epsilon = max(
                            cfg.qnet.eps_end, ll_agent.epsilon * eps_decay_per_ep
                        )

                    stats["epsilon_hl"], stats["epsilon_ll"] = (
                        hl_agent.epsilon,
                        ll_agent.epsilon,
                    )
                    m = tracker.flush_episode(global_ep, env, stats)
                    logger.log(m)

                    phase = "WARMUP" if is_warmup else "TRAIN"
                    print(
                        f"[{phase}] Ep {global_ep:5d} | Epoch {epoch}/{args.epochs} | "
                        f"AccRatio={m.acceptance_ratio:.3f} | acc={m.accepted} rej={m.rejected} | "
                        f"HL_R={m.hl_reward:.2f} LL_R={m.ll_reward:.2f} | "
                        f"HL_L={m.hl_loss:.4f} LL_L={m.ll_loss:.4f} VGAE_L={m.vgae_loss:.4f} | "
                        f"eps={m.epsilon:.3f} | DeployCost={m.total_deploy_cost:.2f} | "
                        f"Archive={stats['pareto_archive_size']} | time={time.perf_counter() - ep_start:.1f}s",
                        flush=True,
                    )

    torch.save(
        {
            "vgae": vgae.state_dict(),
            "hl_q": hl_agent.q_net.state_dict(),
            "ll_q": ll_agent.q_net.state_dict(),
            "w_accept": scalarizer.w_accept,
            "w_cost": scalarizer.w_cost,
        },
        args.checkpoint,
    )
    print(f"\nSaved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
