import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import argparse
import time
import json
import numpy as np
import torch
from config import Config
from env.nfv_env import NFVEnvironment
from models.vgae import MNVGAE
from agents.hl_agent import HLAgent
from agents.ll_agent import LLAgent
from utils.pareto import ParetoScalarizer, pareto_front
from train import run_episode, set_seeds
from data.loader import discover_episodes, parse_episode

def eval_episode(env, vgae, hl_agent, ll_agent, scalarizer, cfg, device, G=None, reqs=None, topo_id=None, fixed_weight=None):
    if G is not None:
        env.reset(G, reqs, topology_id=topo_id)
    else:
        env.reset()
    return run_episode(env, vgae, hl_agent, ll_agent, None, scalarizer, cfg, device, tracker=None, train=False, fixed_weight=fixed_weight)

def scan_pareto_front(env, vgae, hl_agent, ll_agent, scalarizer, cfg, device, eval_paths, num_points=11):
    results = []
    for k in range(num_points):
        w_accept = k / max(1, num_points - 1)
        w_cost = 1.0 - w_accept
        ratios, costs = [], []
        for path in eval_paths:
            G, reqs, _, topo_id = parse_episode(path)
            r = eval_episode(env, vgae, hl_agent, ll_agent, scalarizer, cfg, device, G=G, reqs=reqs, topo_id=topo_id, fixed_weight=(w_accept, w_cost))
            ratios.append(r['acceptance_ratio'])
            costs.append(r['total_deploy_cost'])
        results.append({'w_accept': w_accept, 'w_cost': w_cost, 'acc_ratio': float(np.mean(ratios)), 'deploy_cost': float(np.mean(costs))})
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='vgae_hrl_ql_checkpoint.pt')
    parser.add_argument('--data-dir', type=str, default='data/dataset_01/test')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--pareto-scan', action='store_true')
    parser.add_argument('--pareto-points', type=int, default=11)
    parser.add_argument('--pareto-out', type=str, default='pareto_front.json')
    args = parser.parse_args()
    cfg = Config()
    set_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = NFVEnvironment(cfg)
    vgae = MNVGAE(cfg).to(device)
    hl_agent = HLAgent(cfg, device)
    ll_agent = LLAgent(cfg, device)
    scalarizer = ParetoScalarizer(cfg)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint không tồn tại: {args.checkpoint}")
    print(f"Đang load checkpoint từ: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    vgae.load_state_dict(ckpt['vgae'])
    hl_agent.q_net.load_state_dict(ckpt['hl_q'])
    ll_agent.q_net.load_state_dict(ckpt['ll_q'])
    scalarizer.w_accept = ckpt.get('w_accept', 0.5)
    scalarizer.w_cost = ckpt.get('w_cost', 0.5)
    vgae.eval()
    hl_agent.epsilon = 0.0
    ll_agent.epsilon = 0.0
    eval_paths = discover_episodes(args.data_dir) if args.data_dir and os.path.exists(args.data_dir) else []
    if args.pareto_scan and eval_paths:
        print(f"Đang quét Pareto front với {args.pareto_points} điểm trọng số trên {len(eval_paths)} episodes")
        results = scan_pareto_front(env, vgae, hl_agent, ll_agent, scalarizer, cfg, device, eval_paths, args.pareto_points)
        with open(args.pareto_out, 'w') as f:
            json.dump(results, f, indent=2)
        front_pts = pareto_front([(r['acc_ratio'], -r['deploy_cost']) for r in results])
        print(f"Đã lưu Pareto scan vào: {args.pareto_out}")
        for r in results:
            print(f"  w_accept={r['w_accept']:.2f} w_cost={r['w_cost']:.2f} | AccRatio={r['acc_ratio']:.4f} | DeployCost={r['deploy_cost']:.4f}")
        print(f"  Số điểm không bị dominate: {len(front_pts)}/{len(results)}")
        return
    all_ratios, all_revenues, all_rewards, all_costs = [], [], [], []
    start_time = time.time()
    if eval_paths:
        total_eps = len(eval_paths)
        print(f"Bắt đầu evaluate trên {total_eps} episodes từ thư mục: {args.data_dir}")
        for ep, path in enumerate(eval_paths, 1):
            ep_start = time.time()
            G, reqs, _, topo_id = parse_episode(path)
            r = eval_episode(env, vgae, hl_agent, ll_agent, scalarizer, cfg, device, G=G, reqs=reqs, topo_id=topo_id)
            all_ratios.append(r['acceptance_ratio'])
            all_revenues.append(r.get('total_deploy_cost', 0.0))
            all_rewards.append(r['hl_reward'])
            all_costs.append(r.get('total_deploy_cost', 0.0))
            if ep % args.log_interval == 0 or ep == total_eps:
                elapsed = time.time() - ep_start
                total_elapsed = time.time() - start_time
                print(f"  [Eval] Ep {ep:3d}/{total_eps} [{os.path.basename(path)}] | AccRatio={r['acceptance_ratio']:.3f} | "
                    f"HLReward={r['hl_reward']:.2f} | DeployCost={r['total_deploy_cost']:.2f} | acc={r['accepted']} rej={r['rejected']} | "
                    f"Time: {elapsed:.2f}s (Tổng: {total_elapsed/60:.1f}m)")
    else:
        total_eps = args.episodes
        print(f"Bắt đầu evaluate trên {total_eps} episodes ngẫu nhiên")
        for ep in range(1, total_eps + 1):
            ep_start = time.time()
            r = eval_episode(env, vgae, hl_agent, ll_agent, scalarizer, cfg, device)
            all_ratios.append(r['acceptance_ratio'])
            all_rewards.append(r['hl_reward'])
            all_costs.append(r.get('total_deploy_cost', 0.0))
            if ep % args.log_interval == 0 or ep == total_eps:
                elapsed = time.time() - ep_start
                total_elapsed = time.time() - start_time
                print(f"  [Eval] Ep {ep:3d}/{total_eps} | AccRatio={r['acceptance_ratio']:.3f} | HLReward={r['hl_reward']:.2f} | "
                    f"DeployCost={r['total_deploy_cost']:.2f} | acc={r['accepted']} rej={r['rejected']} | Time: {elapsed:.2f}s (Tổng: {total_elapsed/60:.1f}m)")
    print(f"\n=== Evaluation Summary ===")
    print(f"  Tổng thời gian : {(time.time() - start_time)/60:.2f} phút")
    print(f"  AccRatio       : {np.mean(all_ratios):.4f} ± {np.std(all_ratios):.4f}")
    print(f"  HL Reward      : {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    print(f"  DeployCost     : {np.mean(all_costs):.4f} ± {np.std(all_costs):.4f}")

if __name__ == '__main__':
    main()
