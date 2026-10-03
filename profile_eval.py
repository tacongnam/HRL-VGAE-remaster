#!/usr/bin/env python3
"""
Profiling script to measure time breakdown in Pareto evaluation.
Run: python profile_eval.py --checkpoint <ckpt> --data-dir <small_test_dir> --pareto-points 1
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import torch
from config import Config
from env.nfv_env import NFVEnvironment
from data.loader import parse_episode, discover_episodes
from train import run_episode
from evaluate import _load_checkpoint


def profile_single_weight(checkpoint, data_dir, cfg, device):
    """Run 1 episode with timing breakdown."""
    vgae, hl_agent, ll_agent, scalarizer = _load_checkpoint(checkpoint, cfg, device)
    
    # Load first episode only
    episodes = discover_episodes(data_dir)
    if not episodes:
        raise FileNotFoundError(f"No episodes in {data_dir}")
    
    path = episodes[0]
    print(f"\n=== Profiling single episode: {path} ===")
    
    # Parse
    t_parse_start = time.perf_counter()
    G, reqs, _, topo_id = parse_episode(path)
    t_parse = time.perf_counter() - t_parse_start
    
    # Environment setup
    t_env_start = time.perf_counter()
    env = NFVEnvironment(cfg)
    env.reset(G, reqs, topology_id=topo_id)
    t_env = time.perf_counter() - t_env_start
    
    # Episode run with profiling hooks
    t_ep_start = time.perf_counter()
    
    # Monkey-patch run_episode to insert timing
    import train as train_module
    original_run = train_module.run_episode
    
    timing_data = {}
    
    def patched_encode_graph(vgae, env, device, update_temporal=True):
        t0 = time.perf_counter()
        from train import encode_graph as orig_encode
        result = orig_encode(vgae, env, device, update_temporal)
        timing_data.setdefault("encode_graph", 0.0)
        timing_data["encode_graph"] += time.perf_counter() - t0
        return result
    
    def patched_dijkstra(*args, **kwargs):
        t0 = time.perf_counter()
        from utils.dijkstra import custom_dijkstra
        result = custom_dijkstra(*args, **kwargs)
        timing_data.setdefault("dijkstra", 0.0)
        timing_data["dijkstra"] += time.perf_counter() - t0
        return result
    
    def patched_select_sfc(*args, **kwargs):
        t0 = time.perf_counter()
        result = hl_agent._original_select_sfc(*args, **kwargs)
        timing_data.setdefault("hl_select", 0.0)
        timing_data["hl_select"] += time.perf_counter() - t0
        return result
    
    def patched_select_node(*args, **kwargs):
        t0 = time.perf_counter()
        result = ll_agent._original_select_node(*args, **kwargs)
        timing_data.setdefault("ll_select", 0.0)
        timing_data["ll_select"] += time.perf_counter() - t0
        return result
    
    # Patch
    train_module.encode_graph = patched_encode_graph
    train_module.custom_dijkstra = patched_dijkstra
    hl_agent._original_select_sfc = hl_agent.select_sfc
    ll_agent._original_select_node = ll_agent.select_node
    hl_agent.select_sfc = patched_select_sfc
    ll_agent.select_node = patched_select_node
    
    # Run episode
    with torch.no_grad():
        stats = run_episode(
            env,
            vgae,
            hl_agent,
            ll_agent,
            vgae_optimizer=None,
            scalarizer=scalarizer,
            cfg=cfg,
            device=device,
            train=False,
            fixed_weight=(0.5, 0.5),
        )
    
    t_episode = time.perf_counter() - t_ep_start
    
    # Restore
    train_module.run_episode = original_run
    
    # Report
    print(f"\nTiming breakdown:")
    print(f"  parse_episode:  {t_parse*1000:.1f}ms")
    print(f"  env.reset:      {t_env*1000:.1f}ms")
    print(f"  run_episode:    {t_episode*1000:.1f}ms ({t_episode:.2f}s)")
    
    total_internal = sum(timing_data.values())
    print(f"\n  Inside run_episode:")
    for key, val in sorted(timing_data.items(), key=lambda x: -x[1]):
        pct = 100.0 * val / t_episode if t_episode > 0 else 0
        print(f"    {key:20s}: {val*1000:7.1f}ms ({pct:5.1f}%)")
    
    unaccounted = t_episode - total_internal
    unaccounted_pct = 100.0 * unaccounted / t_episode if t_episode > 0 else 0
    print(f"    {'[other/overhead]':20s}: {unaccounted*1000:7.1f}ms ({unaccounted_pct:5.1f}%)")
    
    print(f"\nAcceptance ratio: {stats['acceptance_ratio']:.3f}")
    print(f"Deploy cost:      {stats['total_deploy_cost']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="vgae_hrl_ql_checkpoint.pt")
    parser.add_argument("--data-dir", type=str, default="data/dataset_01/test")
    args = parser.parse_args()
    
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    profile_single_weight(args.checkpoint, args.data_dir, cfg, device)


if __name__ == "__main__":
    main()
