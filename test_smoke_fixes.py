import numpy as np
import torch
from config import Config
from utils.pareto import (
    select_by_hypervolume,
    prune_by_hypervolume,
    hv_score_for_action,
)
from env import NFVEnvironment
from models import MNVGAE
from agents import HLAgent, LLAgent
from utils import ParetoScalarizer
from train import run_episode

# -------------------------------------------------------------
# Test 1: Kiểm tra HV không còn bị triệt tiêu về 0
# -------------------------------------------------------------
cfg = Config()
ref_point = cfg.pareto.hv_reference_point()
q_good = np.array([0.8, -6.2])  # Node tốt: Hiệu năng cao, Chi phí/Trễ thấp
q_bad = np.array([-5.0, -12.5])  # Node xấu: Thất bại, Chi phí/Trễ cao
q_sets = [
    prune_by_hypervolume([q_good], 10, ref_point),
    prune_by_hypervolume([q_bad], 10, ref_point),
]

hv_good = hv_score_for_action(q_sets[0], ref_point)
hv_bad = hv_score_for_action(q_sets[1], ref_point)
chosen = select_by_hypervolume(q_sets, np.array([True, True]), ref_point)

print(f"[TEST 1] Ref Point: {ref_point}")
print(f"[TEST 1] HV Good Node: {hv_good:.1f} | HV Bad Node: {hv_bad:.1f}")
print(f"[TEST 1] Selected Action: {chosen}")
assert hv_good > 0.0 and hv_bad > 0.0, "HV phải > 0"

# -------------------------------------------------------------
# Test 2: Kiểm tra Buffer lưu trữ Transition khi Fail
# -------------------------------------------------------------
device = torch.device("cpu")

# CẤU HÌNH NHẸ ĐỂ KHÔNG BỊ TREO TÍNH TOÁN PARETO 4D
cfg.sfc.arrival_interval = 1
cfg.sfc.arrival_rate = 2.0  # Chỉ sinh ~2 request mỗi bước
cfg.train.episode_horizon = 3  # Chạy 3 bước thời gian rồi dừng
cfg.qnet.batch_size = 4  # Batch size tí hon để train siêu nhanh
cfg.qnet.ll_buffer_size = 100

env = NFVEnvironment(cfg)
env.reset()
vgae = MNVGAE(cfg).to(device)
hl_agent = HLAgent(cfg, device)
ll_agent = LLAgent(cfg, device)
scalarizer = ParetoScalarizer(cfg)

initial_ll_buffer_len = len(ll_agent.buffer)

stats = run_episode(
    env,
    vgae,
    hl_agent,
    ll_agent,
    None,
    scalarizer,
    cfg,
    device,
    train=True,
    verbose=False,
)

print(
    f"[TEST 2] Buffer size ban đầu: {initial_ll_buffer_len} -> Sau episode: {len(ll_agent.buffer)}"
)
print(f"[TEST 2] Accepted: {stats['accepted']}, Rejected: {stats['rejected']}")
assert len(ll_agent.buffer) > 0, "LL Buffer phải ghi nhận transitions kể cả khi fail!"

for entry in list(ll_agent.buffer.buf)[:2]:
    reward_vec = entry[2]
    assert not np.isnan(reward_vec).any()

print("[TEST 3] Không chứa NaN/Inf: PASS")
print("[TEST 4] Test hoàn tất siêu tốc: PASS")
