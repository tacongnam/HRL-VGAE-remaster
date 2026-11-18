import argparse
import os
import random
import sys
import time
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.jit\.script.*",
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from agents import HLAgent, LLAgent
from config import Config
from data import discover_episodes, parse_episode
from env import NFVEnvironment
from models import MNVGAE
from utils import (
    MetricsTracker,
    ParetoArchive,
    ParetoScalarizer,
    TrainingLogger,
    compute_path_cost_delay,
    custom_dijkstra,
    total_deploy_cost,
)


def set_seeds(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _atomic_torch_save(payload: dict, path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _runtime_env_meta(device: torch.device, seed: int, deterministic: bool) -> dict:
    return {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] --device={device_arg} requested but CUDA unavailable; falling back to cpu", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def encode_graph(vgae, env, device, cfg=None, update_temporal=True, t=None):
    x, ei = env.get_node_features_tensor(device, t=t)
    if cfg is not None and cfg.train.ablation_mode == "HRL-noVGAE":
        n = env.G.number_of_nodes()
        d_z = cfg.vgae.d_latent
        z_nodes = x.detach()
        z_global = torch.cat([z_nodes.mean(dim=0), z_nodes.max(dim=0).values], dim=-1)
        mu = torch.zeros(n, d_z, device=device)
        logvar = torch.zeros(n, d_z, device=device)
        return z_nodes, z_global, x, ei, mu, logvar
    vgae.eval()
    with torch.inference_mode():
        z_nodes, z_global, mu, logvar = vgae(x, ei, graph_id=env.topology_id, update_temporal=update_temporal)
    return z_nodes.detach(), z_global.detach(), x, ei, mu, logvar


def build_vnf_feature(vnf, cfg, device, env=None):
    cpu_norm = env.get_vnf_cpu_norm(vnf) if env is not None else float(np.clip(vnf.cpu_req / max(1e-6, cfg.network.cpu_capacity_mips), 0.0, 1.0))
    return torch.tensor([cpu_norm], dtype=torch.float32, device=device)


def build_vnf_feature_cpu(vnf, cfg, env=None) -> torch.Tensor:
    cpu_norm = env.get_vnf_cpu_norm(vnf) if env is not None else float(np.clip(vnf.cpu_req / max(1e-6, cfg.network.cpu_capacity_mips), 0.0, 1.0))
    return torch.tensor([cpu_norm], dtype=torch.float32)


def build_sfc_feature(sfc, t: float, z_nodes: torch.Tensor, env: NFVEnvironment, device: torch.device) -> torch.Tensor:
    scalars = env.get_sfc_scalar_features(sfc, t)
    scalar_t = torch.from_numpy(scalars).to(z_nodes.device)
    return torch.cat([z_nodes[sfc.source], z_nodes[sfc.dest], scalar_t], dim=-1).to(device)


def build_sfc_feature_cpu(sfc, t: float, z_nodes_cpu: torch.Tensor, env: NFVEnvironment) -> torch.Tensor:
    scalars = env.get_sfc_scalar_features(sfc, t)
    return torch.cat([z_nodes_cpu[sfc.source], z_nodes_cpu[sfc.dest], torch.from_numpy(scalars)], dim=-1)


def _compute_recon_loss_sampled(z_nodes, edge_index, num_nodes, neg_ratio, device):
    num_pos = edge_index.shape[1]
    if num_pos == 0:
        return torch.tensor(0.0, device=device)
    pos_score = (z_nodes[edge_index[0]] * z_nodes[edge_index[1]]).sum(dim=-1)
    num_neg = max(1, int(num_pos * neg_ratio))
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_score = (z_nodes[neg_src] * z_nodes[neg_dst]).sum(dim=-1)
    return F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + \
           F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))


def _build_reward_vec_2obj(r_perf: float, r_cost: float) -> np.ndarray:
    return np.array([r_perf, r_cost], dtype=np.float32)


def _validate_ckpt_arch(ckpt: dict, cfg: Config, *, check_rl: bool = True):
    arch = ckpt.get("arch_cfg") or ckpt.get("pretrain_cfg")
    if arch:
        checks = [("d_in", cfg.vgae.d_in), ("d_latent", cfg.vgae.d_latent)]
        if check_rl:
            checks.extend([("d_sfc", cfg.qnet.d_sfc), ("d_vnf", cfg.qnet.d_vnf),
                           ("d_hl_input", cfg.d_hl_input), ("d_ll_input", cfg.d_ll_input)])
        mismatches = [f"{k}: ckpt={arch[k]} != cfg={v}" for k, v in checks if k in arch and arch[k] != v]
        if mismatches:
            raise ValueError(f"Incompatible checkpoint architecture ({', '.join(mismatches)})")
        ckpt_abl = ckpt.get("ablation_mode")
        if check_rl and ckpt_abl and ckpt_abl != cfg.train.ablation_mode:
            print(f"[WARN] checkpoint ablation_mode={ckpt_abl} differs from current={cfg.train.ablation_mode}")


def load_pretrained_vgae(vgae: MNVGAE, checkpoint_path: str, device: torch.device, cfg: Config = None) -> optim.Optimizer | None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if cfg is not None:
        _validate_ckpt_arch(ckpt, cfg, check_rl=False)
    state = ckpt.get("vgae", ckpt)
    if cfg is not None and not cfg.vgae.temporal:
        state = {k: v for k, v in state.items() if not k.startswith("temporal_cell.")}
    vgae.load_state_dict(state)
    _STATIC_PREFIXES = ("gcn0.", "skip.", "gcn_mu.", "gcn_logvar.")
    temporal_params = []
    for name, param in vgae.named_parameters():
        if any(name.startswith(p) for p in _STATIC_PREFIXES):
            param.requires_grad_(False)
        elif name.startswith("temporal_cell"):
            temporal_params.append(param)
    print(f"[pretrain] Loaded VGAE from {checkpoint_path} | Trainable temporal params: {len(temporal_params)}")
    return optim.Adam(temporal_params, lr=1e-3) if temporal_params else None


def run_episode(
    env: NFVEnvironment,
    vgae: MNVGAE,
    hl_agent: HLAgent,
    ll_agent: LLAgent,
    vgae_optimizer: optim.Optimizer | None,
    scalarizer: ParetoScalarizer,
    cfg: Config,
    device: torch.device,
    pareto_archive: ParetoArchive | None = None,
    tracker: MetricsTracker | None = None,
    train: bool = True,
    verbose: bool = False,
    fixed_weight: tuple | None = None,
    ll_frozen: bool = False,
) -> dict:
    ep_hl_r, ep_ll_r, ep_r_acc, ep_r_cost = 0.0, 0.0, 0.0, 0.0
    vgae_loss_acc, hl_loss_acc, ll_loss_acc = 0.0, 0.0, 0.0
    vgae_updates, hl_updates, ll_updates = 0, 0, 0
    sfc_vgae_count, sfc_total_count, sfc_step_count = 0, 0, 0
    done = False
    failure_vec = cfg.pareto.failure_penalty_vector()

    if vgae is not None:
        vgae.reset_temporal_state(env.topology_id)

    w_accept, w_cost = (
        fixed_weight
        if fixed_weight
        else (scalarizer.sample_weight() if train else (scalarizer.w_accept, scalarizer.w_cost))
    )
    pareto_w = torch.tensor([w_accept, w_cost], dtype=torch.float32, device=device)

    z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device, cfg)
    graph_dirty = False
    z_nodes_cpu = z_nodes.cpu() if train else None

    while not done:
        if not env.queue:
            done = env.step_time()
            if not done:
                z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device, cfg)
                z_nodes_cpu = z_nodes.cpu() if train else None
                graph_dirty = False
            continue

        if (train and vgae_optimizer is not None
                and sfc_vgae_count >= cfg.vgae.train_every_steps
                and cfg.train.ablation_mode != "HRL-noVGAE"):
            vgae_optimizer.zero_grad(set_to_none=True)
            vgae.train()
            _is_pretrained = not any(
                p.requires_grad
                for n, p in vgae.named_parameters()
                if any(n.startswith(s) for s in ("gcn0.", "skip.", "gcn_mu.", "gcn_logvar."))
            )
            z_nodes_tr, z_global_tr, mu_tr, logvar_tr = vgae(
                x, ei, graph_id=env.topology_id, update_temporal=_is_pretrained
            )
            v_loss = _compute_recon_loss_sampled(
                z_nodes_tr, ei, env.num_nodes, cfg.vgae.neg_sample_ratio, device
            ) + vgae.kl_loss(mu_tr, logvar_tr)
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
            z_nodes_cpu = z_nodes.cpu() if train else None
            sfc_vgae_count = 0

        hl_res = hl_agent.select_sfc(z_global, env.queue, env.t, pareto_w, z_nodes=z_nodes, env=env)
        if hl_res is None:
            done = env.step_time()
            if not done:
                z_nodes, z_global, x, ei, mu, logvar = encode_graph(vgae, env, device, cfg)
                z_nodes_cpu = z_nodes.cpu() if train else None
                graph_dirty = False
            continue

        pre_sfc_state = (z_nodes, z_global, z_nodes_cpu, x, ei)
        prev_hl_z_global = z_global.detach().cpu() if train else None
        sfc_idx, selected_sfc = hl_res
        sfc_feat = build_sfc_feature(selected_sfc, env.t, z_nodes, env, device)
        sfc_feat_cpu = sfc_feat.cpu() if train else None
        pareto_w_cpu = pareto_w.cpu() if train else None
        ll_records, partial_allocations, embedding_failed = [], [], False
        curr_src = selected_sfc.source
        cur_t = float(env.t)

        for i in range(selected_sfc.F_k):
            vnf = selected_sfc.vnf_sequence[i]
            ll_mask_i = env.build_ll_mask(vnf.cpu_req, vnf.ram_req, vnf.hdd_req, t=cur_t)
            z_prev = z_nodes[curr_src].detach()
            vnf_feat = build_vnf_feature(vnf, cfg, device, env=env)

            chosen_node = ll_agent.select_node(
                z_global, z_prev, z_nodes, vnf_feat, sfc_feat, ll_mask_i, pareto_w,
                current_source=curr_src, G=env.G, k_hop=cfg.train.k_hop_explore, t=cur_t,
            )
            if chosen_node is None:
                embedding_failed = True
                break

            state_record = (
                z_global.cpu(), z_prev.cpu(), z_nodes_cpu[chosen_node],
                vnf_feat.cpu(), sfc_feat_cpu, pareto_w_cpu,
            ) if train else None

            path = custom_dijkstra(
                env.G, curr_src, chosen_node, selected_sfc.bandwidth,
                cfg.reward.omega_bw,
                0.0 if cfg.train.ablation_mode == "DefaultBaseline" else cfg.reward.lambda_penalty,
                t=cur_t,
            )
            if path is None or env.get_node_cpu_free(chosen_node, cur_t) < vnf.cpu_req:
                embedding_failed = True
                ll_records.append({
                    "state": state_record, "path_cost": 0.0, "path_delay": 0.0,
                    "target_node": chosen_node, "serve_end": cur_t, "load_std": 0.0,
                })
                break

            link_delay = sum(env.G[path[k]][path[k + 1]]["delay"] for k in range(len(path) - 1)) if len(path) > 1 else 0.0
            serve_start = cur_t + link_delay
            proc_delay = env.G.nodes[chosen_node].get("proc_delay", 0.0) + vnf.delay_at(chosen_node)
            serve_end = serve_start + proc_delay

            env.allocate(
                chosen_node, path, vnf.cpu_req, selected_sfc.bandwidth,
                ram_req=vnf.ram_req, hdd_req=vnf.hdd_req, t_start=cur_t, t_end=serve_end,
            )
            partial_allocations.append((
                chosen_node, path, vnf.cpu_req, selected_sfc.bandwidth,
                vnf.ram_req, vnf.hdd_req, cur_t, serve_end,
            ))
            p_cost, p_delay = compute_path_cost_delay(env.G, path, cur_t)
            load_std_step = env.get_load_std(t=cur_t)
            ll_records.append({
                "state": state_record, "path_cost": p_cost, "path_delay": p_delay,
                "target_node": chosen_node, "serve_end": serve_end, "load_std": load_std_step,
            })
            curr_src = chosen_node
            cur_t = serve_end

            if i < selected_sfc.F_k - 1:
                z_nodes, z_global, x, ei, _, _ = encode_graph(
                    vgae, env, device, cfg, update_temporal=False, t=cur_t
                )
                if train:
                    z_nodes_cpu = z_nodes.cpu()
                    ll_records[-1]["next_z_global"] = z_global.cpu()
                    ll_records[-1]["next_z_nodes"] = z_nodes_cpu

        dest_cost, dest_delay, load_std_dest = 0.0, 0.0, 0.0
        if not embedding_failed:
            dest_path = custom_dijkstra(
                env.G, curr_src, selected_sfc.dest, selected_sfc.bandwidth,
                cfg.reward.omega_bw,
                0.0 if cfg.train.ablation_mode == "DefaultBaseline" else cfg.reward.lambda_penalty,
                t=cur_t,
            )
            if dest_path is None:
                embedding_failed = True
            else:
                dest_delay_val = sum(env.G[dest_path[k]][dest_path[k + 1]]["delay"] for k in range(len(dest_path) - 1)) if len(dest_path) > 1 else 0.0
                dest_end = cur_t + dest_delay_val
                env.allocate(selected_sfc.dest, dest_path, 0.0, selected_sfc.bandwidth, t_start=cur_t, t_end=dest_end)
                partial_allocations.append(
                    (selected_sfc.dest, dest_path, 0.0, selected_sfc.bandwidth, 0.0, 0.0, cur_t, dest_end)
                )
                dest_cost, dest_delay = compute_path_cost_delay(env.G, dest_path, cur_t)
                load_std_dest = env.get_load_std(t=cur_t)
            if not embedding_failed:
                total_delay = env.compute_embedding_delay(partial_allocations, selected_sfc)
                if float(env.t) + total_delay > selected_sfc.deadline:
                    embedding_failed = True

        sfc_total_count += 1

        if embedding_failed:
            env.remove_sfc_from_queue(selected_sfc)
            env.reject_sfc(selected_sfc, partial_allocations)

            if partial_allocations:
                z_nodes, z_global, z_nodes_cpu, x, ei = pre_sfc_state
            graph_dirty = False

            hl_reward_vec = failure_vec.copy()
            r_H = scalarizer.scalarize(float(failure_vec[0]), float(failure_vec[1]), w_accept, w_cost)

            if train and ll_records:
                ll_fail_vec = failure_vec / max(1, len(ll_records))
                for rec in ll_records:
                    ll_agent.store_transition(rec["state"], None, ll_fail_vec, 1.0)

        else:
            deploy_cost = total_deploy_cost(env.G, partial_allocations, cfg)
            env.commit_sfc(selected_sfc, partial_allocations, deploy_cost=deploy_cost)
            env.remove_sfc_from_queue(selected_sfc)
            graph_dirty, sfc_vgae_count = True, sfc_vgae_count + 1

            z_nodes, z_global, x, ei, _, _ = encode_graph(
                vgae, env, device, cfg, update_temporal=False
            )
            z_nodes_cpu = z_nodes.cpu() if train else None

            n_steps = selected_sfc.F_k + 1
            dl_budget = max(1.0, float(selected_sfc.deadline - selected_sfc.arrival_time))
            hop_ref = max(1.0, env._mean_hop_dist)
            w_q_sum = max(1e-6, cfg.reward.alpha + cfg.reward.beta + cfg.reward.gamma_load)
            for rec in ll_records:
                q_pen = (
                    cfg.reward.alpha * min(1.0, rec["path_cost"] / hop_ref)
                    + cfg.reward.beta * min(1.0, rec["path_delay"] / dl_budget)
                    + cfg.reward.gamma_load * min(1.0, 2.0 * rec["load_std"])
                ) / w_q_sum
                rec["r_quality"] = float(np.clip(1.0 - q_pen, 0.0, 1.0))
            q_pen_dest = (
                cfg.reward.alpha * min(1.0, dest_cost / hop_ref)
                + cfg.reward.beta * min(1.0, dest_delay / dl_budget)
                + cfg.reward.gamma_load * min(1.0, 2.0 * load_std_dest)
            ) / w_q_sum
            r_dest = float(np.clip(1.0 - q_pen_dest, 0.0, 1.0))
            r_bar_quality = (sum(r["r_quality"] for r in ll_records) + r_dest) / n_steps

            revenue = env.norm_revenue(selected_sfc)
            norm_rev = float(np.clip(revenue / max(1e-6, env._max_norm_revenue), 0.0, 1.0))
            norm_urg = float(np.clip(selected_sfc.urgency(env.t), 0.0, 1.0))
            r_acc_c = float(
                np.clip(
                    (norm_rev + cfg.reward.theta * norm_urg + cfg.reward.lambda_L * r_bar_quality)
                    / max(1e-6, 1.0 + cfg.reward.theta + cfg.reward.lambda_L),
                    0.0,
                    1.0,
                )
            )
            max_sfc_cost = env.estimate_sfc_max_cost(selected_sfc)
            norm_cost = float(np.clip(deploy_cost / max_sfc_cost, 0.0, 1.0))
            r_cost_c = float(1.0 - norm_cost)
            hl_reward_vec = _build_reward_vec_2obj(r_acc_c, r_cost_c)
            r_H = scalarizer.scalarize(r_acc_c, r_cost_c, w_accept, w_cost)
            scalarizer.update_utopia(r_acc_c, r_cost_c)
            ep_r_acc += r_acc_c
            ep_r_cost += r_cost_c
            ep_ll_r += r_bar_quality

            if tracker:
                tracker.record_sfc_success(selected_sfc.F_k + 1, revenue)

            if train:
                for k, cur_rec in enumerate(ll_records):
                    is_term = k == len(ll_records) - 1
                    r_vec = _build_reward_vec_2obj(
                        cur_rec["r_quality"], r_cost_c
                    )
                    next_s = (
                        None
                        if is_term
                        else (
                            cur_rec["next_z_global"],
                            cur_rec["next_z_nodes"][cur_rec["target_node"]],
                            cur_rec["next_z_nodes"],
                            build_vnf_feature_cpu(selected_sfc.vnf_sequence[k + 1], cfg, env=env),
                            sfc_feat_cpu,
                            env.build_ll_mask(
                                selected_sfc.vnf_sequence[k + 1].cpu_req,
                                selected_sfc.vnf_sequence[k + 1].ram_req,
                                selected_sfc.vnf_sequence[k + 1].hdd_req,
                                t=cur_rec["serve_end"],
                            ),
                            pareto_w_cpu,
                        )
                    )
                    ll_agent.store_transition(cur_rec["state"], next_s, r_vec, float(is_term))

        ep_hl_r += r_H
        if env._dataset_mode:
            env._flush_arrivals()

        if train:
            next_sfcs = [q for q in env.queue if not q.is_expired(env.t)]
            next_feats = [
                build_sfc_feature_cpu(q, env.t, z_nodes_cpu, env)
                for q in next_sfcs
            ]
            hl_done = not env.queue and (
                not hasattr(env, "_req_cursor") or env._req_cursor >= len(env._all_requests)
            )
            hl_agent.store_transition(
                prev_hl_z_global, sfc_feat_cpu, pareto_w_cpu, sfc_idx, hl_reward_vec,
                z_global.cpu(), next_feats, pareto_w_cpu,
                float(hl_done), valid_action_mask=np.ones(len(next_feats), dtype=bool),
            )

        if train and (sfc_total_count % cfg.train.train_interval_sfc == 0):
            if cfg.train.ablation_mode in ["HRL-VGAE", "HRL-noVGAE", "Random-Fit", "GreedyFIFO"] and not ll_frozen:
                loss_ll = ll_agent.train_step()
                if loss_ll is not None:
                    ll_loss_acc += loss_ll
                    ll_updates += 1
            if cfg.train.ablation_mode in ["HRL-VGAE", "HRL-noVGAE", "GreedyLowLevel"]:
                loss_hl = hl_agent.train_step()
                if loss_hl is not None:
                    hl_loss_acc += loss_hl
                    hl_updates += 1

        sfc_step_count += 1
        if not env.queue:
            done = env.step_time()
            if not done:
                z_nodes, z_global, x, ei, _mu, _logvar = encode_graph(vgae, env, device, cfg)
                z_nodes_cpu = z_nodes.cpu() if train else None
                graph_dirty = False

    if pareto_archive is not None and env.accepted > 0:
        pareto_archive.add(
            np.array(
                [ep_r_acc / max(1, env.accepted), ep_r_cost / max(1, env.accepted)],
                dtype=np.float64,
            ),
            episode=getattr(env, "_episode_id", 0),
            w_accept=w_accept, w_cost=w_cost,
            acceptance_ratio=env.acceptance_ratio(),
            total_deploy_cost=env.total_deploy_cost,
        )

    if train and not fixed_weight:
        scalarizer.adapt_weights(
            env.acceptance_ratio(),
            min(1.0, (env.total_deploy_cost / max(1.0, env.accepted)) / 1000.0),
        )
        scalarizer.w_accept = scalarizer.w_accept * 0.9 + w_accept * 0.1
        scalarizer.w_cost = 1.0 - scalarizer.w_accept

    return {
        "hl_reward": ep_hl_r, "ll_reward": ep_ll_r,
        "acceptance_ratio": env.acceptance_ratio(),
        "accepted": env.accepted, "rejected": env.rejected,
        "vgae_loss": vgae_loss_acc / max(1, vgae_updates),
        "hl_loss": hl_loss_acc / max(1, hl_updates),
        "ll_loss": ll_loss_acc / max(1, ll_updates),
        "hl_updates": hl_updates, "ll_updates": ll_updates,
        "epsilon_hl": hl_agent.epsilon, "epsilon_ll": ll_agent.epsilon,
        "total_deploy_cost": env.total_deploy_cost,
        "avg_cost_per_accepted": (env.total_deploy_cost / env.accepted) if env.accepted > 0 else 0.0,
        "total_node_placement_cost": env.total_node_placement_cost,
        "w_accept": w_accept, "w_cost": w_cost,
        "pareto_archive_size": len(pareto_archive) if pareto_archive else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="data/train")
    parser.add_argument("--checkpoint", type=str, default="vgae_hrl_ql_checkpoint.pt")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--csv", type=str, default="training_log.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--passes-per-file", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--archive-size", type=int, default=200)
    parser.add_argument("--train-file", type=str, default="")
    parser.add_argument("--pretrained-vgae", type=str, default="")
    parser.add_argument("--ablation", type=str, default="HRL-VGAE",
                        choices=["DefaultBaseline", "Random-Fit", "GreedyFIFO",
                                 "GreedyLowLevel", "HRL-VGAE", "HRL-noVGAE"])
    parser.add_argument("--ll-checkpoint", type=str, default="")
    parser.add_argument("--resume-train", action="store_true")
    parser.add_argument("--save-replay-buffer", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--val-file", type=str, default="")
    parser.add_argument("--val-dir", type=str, default="")
    parser.add_argument("--best-checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = Config()
    cfg.train.ablation_mode = args.ablation
    if args.seed is not None:
        cfg.train.seed = int(args.seed)
    set_seeds(cfg.train.seed, deterministic=args.deterministic)
    device = _resolve_device(args.device)
    env_meta = _runtime_env_meta(device, cfg.train.seed, args.deterministic)

    train_paths = [args.train_file] if args.train_file else discover_episodes(args.train_dir)
    if not train_paths:
        raise FileNotFoundError("No training files found.")
    val_paths = [args.val_file] if args.val_file else (discover_episodes(args.val_dir) if args.val_dir else [])
    best_ckpt_path = args.best_checkpoint
    if val_paths and not best_ckpt_path:
        root, ext = os.path.splitext(args.checkpoint)
        best_ckpt_path = f"{root}_best{ext or '.pt'}"

    env = NFVEnvironment(cfg)
    vgae = MNVGAE(cfg).to(device)
    hl_agent = HLAgent(cfg, device)
    ll_agent = LLAgent(cfg, device)
    vgae_optimizer = (
        load_pretrained_vgae(vgae, args.pretrained_vgae, device, cfg)
        if args.pretrained_vgae else optim.Adam(vgae.parameters(), lr=cfg.vgae.lr)
    )
    scalarizer = ParetoScalarizer(cfg)

    ll_frozen = False
    if args.ll_checkpoint:
        _ckpt = torch.load(args.ll_checkpoint, map_location=device)
        _validate_ckpt_arch(_ckpt, cfg, check_rl=True)
        ll_agent.q_net.load_state_dict(_ckpt["ll_q"])
        ll_agent.target_net.load_state_dict(_ckpt["ll_q"])
        ll_agent.epsilon = cfg.qnet.eps_end
        ll_frozen = True
        print(f"[ablation] LL policy loaded from {args.ll_checkpoint}, training frozen.")

    tracker = MetricsTracker()
    pareto_archive = ParetoArchive(max_size=args.archive_size)

    best_val_ar = -1.0
    best_val_cost = float("inf")
    start_epoch = 1
    global_ep = 0
    total_train_episodes = args.epochs * len(train_paths) * args.passes_per_file
    warmup_episodes = args.warmup_epochs * len(train_paths) * args.passes_per_file
    decay_episodes = int((total_train_episodes - warmup_episodes) * 0.75)
    eps_decay_per_ep = (cfg.qnet.eps_end / cfg.qnet.eps_start) ** (1.0 / max(1, decay_episodes))

    if args.resume_train and os.path.isfile(args.checkpoint):
        _rckpt = torch.load(args.checkpoint, map_location=device)
        _validate_ckpt_arch(_rckpt, cfg, check_rl=True)
        vgae_state = _rckpt["vgae"]
        if not cfg.vgae.temporal:
            vgae_state = {k: v for k, v in vgae_state.items() if not k.startswith("temporal_cell.")}
        vgae.load_state_dict(vgae_state)
        hl_agent.q_net.load_state_dict(_rckpt["hl_q"])
        hl_agent.target_net.load_state_dict(_rckpt.get("hl_target_q", _rckpt["hl_q"]))
        ll_agent.q_net.load_state_dict(_rckpt["ll_q"])
        ll_agent.target_net.load_state_dict(_rckpt.get("ll_target_q", _rckpt["ll_q"]))
        if "hl_opt" in _rckpt:
            hl_agent.optimizer.load_state_dict(_rckpt["hl_opt"])
        if "ll_opt" in _rckpt:
            ll_agent.optimizer.load_state_dict(_rckpt["ll_opt"])
        if vgae_optimizer is not None and _rckpt.get("vgae_opt") is not None:
            vgae_optimizer.load_state_dict(_rckpt["vgae_opt"])
        hl_agent.epsilon = float(_rckpt.get("epsilon_hl", hl_agent.epsilon))
        if not ll_frozen:
            ll_agent.epsilon = float(_rckpt.get("epsilon_ll", ll_agent.epsilon))
        hl_agent.train_steps = int(_rckpt.get("hl_train_steps", 0))
        ll_agent.train_steps = int(_rckpt.get("ll_train_steps", 0))
        scalarizer.w_accept = float(_rckpt.get("w_accept", scalarizer.w_accept))
        scalarizer.w_cost = float(_rckpt.get("w_cost", scalarizer.w_cost))
        start_epoch = int(_rckpt.get("epoch", 0)) + 1
        global_ep = int(_rckpt.get("global_ep", 0))
        best_val_ar = float(_rckpt.get("best_val_ar", -1.0))
        best_val_cost = float(_rckpt.get("best_val_cost", float("inf")))
        if "hl_buffer" in _rckpt:
            hl_agent.buffer.buf.extend(_rckpt["hl_buffer"])
        if "ll_buffer" in _rckpt:
            ll_agent.buffer.buf.extend(_rckpt["ll_buffer"])
        rng_s = _rckpt.get("rng_states")
        if rng_s:
            random.setstate(rng_s["python"])
            np.random.set_state(rng_s["numpy"])
            torch.set_rng_state(rng_s["torch"].cpu())
            if torch.cuda.is_available() and rng_s.get("cuda") is not None:
                torch.cuda.set_rng_state_all([t.cpu() for t in rng_s["cuda"]])
        print(f"[resume] Resumed from {args.checkpoint} at epoch {start_epoch}, global_ep={global_ep}")

    parsed_cache = {}
    if len(train_paths) <= 50:
        for p in train_paths:
            parsed_cache[p] = parse_episode(p)
    val_cache = {p: parse_episode(p) for p in val_paths} if val_paths else {}

    def _build_ckpt_payload(epoch_num: int) -> dict:
        payload = {
            "vgae": vgae.state_dict(),
            "hl_q": hl_agent.q_net.state_dict(),
            "ll_q": ll_agent.q_net.state_dict(),
            "ablation_mode": cfg.train.ablation_mode,
            "arch_cfg": {
                "d_in": cfg.vgae.d_in,
                "d_hidden": cfg.vgae.d_hidden,
                "d_latent": cfg.vgae.d_latent,
                "temporal": cfg.vgae.temporal,
                "d_sfc": cfg.qnet.d_sfc,
                "d_vnf": cfg.qnet.d_vnf,
                "d_hl_input": cfg.d_hl_input,
                "d_ll_input": cfg.d_ll_input,
            },
            "env_meta": env_meta,
            "hl_target_q": hl_agent.target_net.state_dict(),
            "ll_target_q": ll_agent.target_net.state_dict(),
            "hl_opt": hl_agent.optimizer.state_dict(),
            "ll_opt": ll_agent.optimizer.state_dict(),
            "vgae_opt": vgae_optimizer.state_dict() if vgae_optimizer is not None else None,
            "w_accept": scalarizer.w_accept,
            "w_cost": scalarizer.w_cost,
            "epsilon_hl": hl_agent.epsilon,
            "epsilon_ll": ll_agent.epsilon,
            "hl_train_steps": hl_agent.train_steps,
            "ll_train_steps": ll_agent.train_steps,
            "epoch": epoch_num,
            "global_ep": global_ep,
            "best_val_ar": best_val_ar,
            "best_val_cost": best_val_cost,
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if args.save_replay_buffer:
            payload["hl_buffer"] = list(hl_agent.buffer.buf)
            payload["ll_buffer"] = list(ll_agent.buffer.buf)
        return payload

    with TrainingLogger(log_dir=args.log_dir, csv_path=args.csv) as logger:
        try:
            for epoch in range(start_epoch, args.epochs + 1):
                epoch_paths = train_paths.copy()
                random.shuffle(epoch_paths)
                is_warmup = epoch <= args.warmup_epochs

                for path in epoch_paths:
                    G, reqs, _, topo_id = parsed_cache[path] if path in parsed_cache else parse_episode(path)
                    for _ in range(args.passes_per_file):
                        global_ep += 1
                        if is_warmup:
                            hl_agent.epsilon, ll_agent.epsilon = 1.0, 1.0

                        env.reset(G, reqs, topology_id=topo_id)
                        env._episode_id = global_ep
                        ep_start = time.perf_counter()

                        stats = run_episode(
                            env, vgae, hl_agent, ll_agent, vgae_optimizer, scalarizer, cfg, device,
                            pareto_archive=pareto_archive, tracker=tracker, train=True,
                            verbose=args.verbose and global_ep <= 2,
                            ll_frozen=ll_frozen,
                        )

                        if not is_warmup:
                            hl_agent.epsilon = max(cfg.qnet.eps_end, hl_agent.epsilon * eps_decay_per_ep)
                            if not ll_frozen:
                                ll_agent.epsilon = max(cfg.qnet.eps_end, ll_agent.epsilon * eps_decay_per_ep)

                        stats["epsilon_hl"], stats["epsilon_ll"] = hl_agent.epsilon, ll_agent.epsilon
                        m = tracker.flush_episode(global_ep, env, stats)
                        logger.log(m)

                        phase = "WARMUP" if is_warmup else "TRAIN"
                        avg_c = (m.total_deploy_cost / m.accepted) if m.accepted > 0 else 0.0
                        print(
                            f"[{phase}] Ep {global_ep:5d} | Epoch {epoch}/{args.epochs} | "
                            f"AccRatio={m.acceptance_ratio:.3f} | acc={m.accepted} rej={m.rejected} | "
                            f"HL_R={m.hl_reward:.2f} LL_R={m.ll_reward:.2f} | "
                            f"HL_L={m.hl_loss:.4f} LL_L={m.ll_loss:.4f} VGAE_L={m.vgae_loss:.4f} | "
                            f"eps={m.epsilon:.3f} | DeployCost={m.total_deploy_cost:.2f} AvgCost={avg_c:.2f} "
                            f"NodeCost={stats.get('total_node_placement_cost', 0.0):.2f} | "
                            f"Archive={stats['pareto_archive_size']} | time={time.perf_counter() - ep_start:.1f}s",
                            flush=True,
                        )

                if val_paths and (not is_warmup or epoch == args.epochs):
                    prev_hl_eps, prev_ll_eps = hl_agent.epsilon, ll_agent.epsilon
                    hl_agent.epsilon, ll_agent.epsilon = 0.0, 0.0
                    val_ars, val_costs = [], []
                    try:
                        with torch.no_grad():
                            for vp in val_paths:
                                vG, vreqs, _, vtopo = val_cache[vp]
                                env.reset(vG, vreqs, topology_id=vtopo)
                                vstats = run_episode(env, vgae, hl_agent, ll_agent, None, scalarizer, cfg, device, train=False)
                                val_ars.append(float(vstats["acceptance_ratio"]))
                                val_costs.append(float(vstats["total_deploy_cost"]))
                    finally:
                        hl_agent.epsilon, ll_agent.epsilon = prev_hl_eps, prev_ll_eps
                    v_ar, v_cost = float(np.mean(val_ars)), float(np.mean(val_costs))
                    print(f"[VAL] Epoch {epoch} | val_AR={v_ar:.4f} | val_cost={v_cost:.2f}", flush=True)
                    if v_ar > best_val_ar or (abs(v_ar - best_val_ar) < 1e-12 and v_cost < best_val_cost):
                        best_val_ar, best_val_cost = v_ar, v_cost
                        _atomic_torch_save(_build_ckpt_payload(epoch), best_ckpt_path)
                        print(f"[VAL] Saved best checkpoint -> {best_ckpt_path}", flush=True)
                _atomic_torch_save(_build_ckpt_payload(epoch), args.checkpoint)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[OOM] CUDA out of memory at global_ep={global_ep}; latest atomic checkpoint preserved at {args.checkpoint}", flush=True)
            raise

    if start_epoch > args.epochs:
        print(f"Checkpoint already at epoch >= {args.epochs}: {args.checkpoint}")
    print(f"\nSaved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()