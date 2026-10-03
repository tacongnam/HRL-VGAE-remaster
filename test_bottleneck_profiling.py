#!/usr/bin/env python3
"""
Xác minh số lần LL/VGAE forward được gọi trong một episode nhỏ.

Chạy: python test_bottleneck_profiling.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import time
import numpy as np
from config import Config
from env.nfv_env import NFVEnvironment
from data.loader import parse_episode
from models.vgae import MNVGAE
from agents.hl_agent import HLAgent
from agents.ll_agent import LLAgent
from utils.pareto import ParetoScalarizer

# Wrapper để đếm forward passes
class CountingModule:
    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.count = 0
        self.time_total = 0.0
        
    def __call__(self, *args, **kwargs):
        self.count += 1
        start = time.perf_counter()
        result = self.module(*args, **kwargs)
        self.time_total += time.perf_counter() - start
        return result
    
    def report(self):
        avg_ms = (self.time_total / max(1, self.count)) * 1000
        print(f"  {self.name}: {self.count:6d} calls | {self.time_total:8.3f}s | {avg_ms:6.2f}ms/call")

def main():
    cfg = Config()
    device = torch.device("cpu")  # CPU để dễ profiling
    
    # Load một episode nhỏ để test
    try:
        episodes = sorted(__import__('glob').glob(
            os.path.join(cfg.train.seed, "data", "dataset_01", "test", "episode_*.json")
        ))
        if not episodes:
            # Fallback: tìm trong current dir
            episodes = sorted(__import__('glob').glob("data/dataset_01/test/episode_*.json"))
        
        if not episodes:
            print("❌ Không tìm thấy episode. Chạy với test data mặc định.")
            return
        
        test_path = episodes[0]
        print(f"📁 Load episode: {test_path}")
        
    except Exception as e:
        print(f"⚠️  {e}")
        return
    
    G, reqs, _, topo_id = parse_episode(test_path)
    
    # Build agents
    vgae = MNVGAE(cfg).to(device)
    hl_agent = HLAgent(cfg, device)
    ll_agent = LLAgent(cfg, device)
    vgae.eval()
    hl_agent.q_net.eval()
    ll_agent.q_net.eval()
    
    # Wrap forward
    vgae_forward_orig = vgae.forward
    hl_forward_orig = hl_agent.q_net.forward
    ll_forward_orig = ll_agent.q_net.forward
    
    counter_vgae = CountingModule("VGAE.forward", lambda x, ei, **kw: vgae_forward_orig(x, ei, **kw))
    counter_hl = CountingModule("HL_Q.forward", hl_forward_orig)
    counter_ll = CountingModule("LL_Q.forward", ll_forward_orig)
    
    vgae.forward = counter_vgae
    hl_agent.q_net.forward = counter_hl
    ll_agent.q_net.forward = counter_ll
    
    # Run mini episode (sample only 10 requests)
    env = NFVEnvironment(cfg)
    reqs_subset = reqs[:10]
    env.reset(G, reqs_subset, topology_id=topo_id)
    
    scalarizer = ParetoScalarizer(cfg)
    w_accept, w_cost = 0.5, 0.5
    pareto_w = torch.tensor([w_accept, w_cost], dtype=torch.float32, device=device)
    
    print("\n" + "="*70)
    print(f"🧪 Test Episode: {len(reqs_subset)} requests (sample từ {len(reqs)})")
    print("="*70)
    
    from train import run_episode
    
    t_start = time.perf_counter()
    stats = run_episode(
        env, vgae, hl_agent, ll_agent,
        vgae_optimizer=None,
        scalarizer=scalarizer,
        cfg=cfg, device=device,
        train=False, fixed_weight=(w_accept, w_cost),
    )
    t_elapsed = time.perf_counter() - t_start
    
    print("\n" + "="*70)
    print("📊 Profiling Results:")
    print("="*70)
    counter_vgae.report()
    counter_hl.report()
    counter_ll.report()
    
    print("\n" + "="*70)
    print(f"⏱️  Episode Time: {t_elapsed:.2f}s | Requests: {len(reqs_subset)}")
    print(f"   Avg/request: {t_elapsed / max(1, len(reqs_subset)) * 1000:.1f}ms")
    print("="*70)
    
    print("\n📈 Metrics:")
    print(f"   Accepted: {stats['accepted']} / {len(reqs_subset)}")
    print(f"   Acc Ratio: {stats['acceptance_ratio']:.3f}")
    print(f"   Deploy Cost: {stats['total_deploy_cost']:.2f}")
    
    # Estimasi full Pareto scan
    n_full_reqs = len(reqs)
    n_weights = 11
    scaling_factor = n_full_reqs / len(reqs_subset)
    
    print("\n" + "="*70)
    print("📊 Extrapolation để Full Pareto Scan:")
    print("="*70)
    
    ll_calls_per_req = counter_ll.count / max(1, len(reqs_subset))
    vgae_calls_per_req = counter_vgae.count / max(1, len(reqs_subset))
    
    print(f"   LL forward/request: {ll_calls_per_req:.1f}")
    print(f"   VGAE forward/request: {vgae_calls_per_req:.1f}")
    
    total_ll_calls = ll_calls_per_req * n_full_reqs * n_weights * 10  # assume 10 episodes
    total_time_est = counter_ll.time_total * total_ll_calls / max(1, counter_ll.count) / 3600
    
    print(f"\n   Estimated LL forward calls (full): {total_ll_calls:.0f}")
    print(f"   Estimated LL inference time (11 weights × 10 episodes): {total_time_est:.1f} hours")
    print("="*70)

if __name__ == "__main__":
    main()
