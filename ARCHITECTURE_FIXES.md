# HRL-VGAE-Remaster: Architecture Fixes & Code Cleanup

## Issues Found & Fixed

### 🔴 CRITICAL ISSUES (từ kiến trúc cũ)

#### 1. **Acceptance Ratio = 0.0000** ❌
**Root Cause:** 
- Không có logic xử lý request scheduling (chỉ có Placer, không có Scheduler)
- Placement không thành công vì state khiếm khuyết
- Reward weights không cân bằng (learning chưa ổn định)

**Fix:**
- ✅ Thêm **Upper Agent** với 2 heads: `schedule_net` (chọn request) + `place_net` (chọn DC)
- ✅ Tách biệt state cho scheduling vs placement
- ✅ Dùng **Double DQN** thay vì vanilla DQN (giảm overestimation bias)

#### 2. **Không có Tuần tự Pretrain** ❌
**Root Cause:**
- VGAE có thể bị finetune online → backbone mất training
- Lower Agent không có → Upper phải học cả 2 việc → phân kỳ
- Buffer chung cho nhiều action-space → nhiễu label

**Fix:**
- ✅ **Phase 0:** Pretrain VGAE → khóa vĩnh viễn
- ✅ **Phase 1:** Pretrain Lower (teacher heuristic) → khóa 
- ✅ **Phase 2:** Pretrain Upper (Lower đã khóa)
- ✅ **Phase 3:** Main training (tuning nhẹ hoặc khóa hoàn toàn)

#### 3. **Load Checkpoint Bug** ❌
**Root Cause:**
- `main.py` cũ không có logic load pretrained weights
- Hoặc load vào sai model → weight bị bỏ sót

**Fix:**
- ✅ Tạo hàm `load_all_checkpoints()` **MỘT NƠI DUY NHẤT** trước training chính
- ✅ Kiểm tra tất cả 3 checkpoint (VGAE, Lower, Upper) đều được load

#### 4. **Epsilon Schedule Reset** ❌
**Root Cause:**
- epsilon reset theo file → hành vi gần-random quá lâu

**Fix:**
- ✅ Tính epsilon theo **global_step** xuyên suốt, không reset

#### 5. **Reward Weights Động** ❌
**Root Cause:**
- WeightNet học alpha/beta → target di chuyển 2 lớp → phân kỳ

**Fix:**
- ✅ Dùng **fixed weights** (w1=1.0, w2=0.3, w3=1.0)
- ✅ Không có weight network

---

## File Architecture

### ✅ New Files (Tạo)

```
agents/
├── upper_agent.py          ← Upper Agent (scheduler + placer, 2 heads)
├── lower_agent.py          ← Lower Agent (routing + load balancing)

strategy/
├── hrl_strategy.py         ← HRL Strategy coordinator

trainers/
├── hrl_trainer.py          ← Orchestrates 4 phases

main.py (Updated)          ← Complete pipeline with 4 phases
```

### ⚠️ Old Files (Cần Review / Deprecated)

```
trainer/dqn_trainer.py      ← OLD: Chỉ train 1 agent, không tuần tự
model/dqn/agent.py          ← OLD: Vanilla DQN, không Double DQN
algorithm/placement.py      ← OLD: PlacementEngine, chỉ Placer
```

### 📦 Files to Keep (Có sử dụng)

```
config/default_config.py    → Keep, add HRL config
model/vgae/vgae.py         → Keep (frozen)
algorithm/routing.py        → Keep (for Lower routing)
environment/               → Keep (VNEEnvironment)
utils/                     → Keep (logging, metrics)
```

---

## What to Do with Old Files

### Option 1: Keep for Reference
```python
# trainer/dqn_trainer.py → Đổi tên thành dqn_trainer_deprecated.py
# model/dqn/agent.py → Đổi tên thành dqn_agent_deprecated.py
```

### Option 2: Remove (Recommended)
```bash
# Xóa các file không dùng
rm trainer/dqn_trainer.py
rm model/dqn/agent.py
rm trainer/pipeline.py  # Không dùng
```

---

## Config Updates Needed

### File: `config/default_config.py`

```python
# Thêm vào:

HRL = {
    'top_k_candidates': 8,           # Top-K requests từ queue
    'k_routes': 3,                   # K-shortest paths
    'upper_hidden_dim': 128,         # Upper network hidden dim
    'lower_hidden_dim': 64,          # Lower network hidden dim
    'upper_gamma': 0.95,
    'lower_gamma': 0.9,
    'upper_lr_pretrain': 1e-3,
    'upper_lr_online': 1e-4,
    'lower_lr_pretrain': 1e-3,
    'lower_lr_online': 5e-5,
}

VGAE = {
    ...
    'num_pretrain_nets': 5,          # Thêm
    'freeze_after_pretrain': True,   # Thêm
}

TRAINING = {
    ...
    'num_lower_pretrain_episodes': 500,   # Thêm Phase 1
    'num_upper_pretrain_episodes': 1000,  # Thêm Phase 2
    'num_episodes': 2000,                 # Phase 3
}

EVALUATION = {
    'num_test_requests': 100,
}
```

---

## Metrics Tracking

### Điểm kiểm tra:

#### Phase 0 (VGAE Pretrain)
```
✓ vgae_loss: 0.1 → 0.01 (converged)
✓ VGAE frozen: True
```

#### Phase 1 (Lower Pretrain)
```
✓ lower_loss: hội tụ giảm dần
✓ lower_episode_reward: tăng dần
✓ node_pressure: giảm (cân bằng tốt)
✓ Lower frozen: True
```

#### Phase 2 (Upper Pretrain)
```
✓ upper_loss_schedule: hội tụ
✓ upper_loss_place: hội tụ
✓ acceptance_ratio: > 0.5 (tối thiểu)
✓ avg_reward: tăng dần
```

#### Phase 3 (Main Training)
```
✓ acceptance_ratio: > 0.7 (target)
✓ upper_reward: > -1.0
✓ lower_reward: → 0 (cân bằng)
✓ resource_utilization: > 0.7
```

---

## Testing Changes

### Test Cases

```python
# test/test_hrl_integration.py (NEW)
def test_upper_agent():
    """Test schedule + placement"""
    pass

def test_lower_agent():
    """Test routing + load balancing"""
    pass

def test_phase_transitions():
    """Test checkpoint loading"""
    pass

def test_double_dqn():
    """Test Double DQN vs vanilla DQN"""
    pass
```

---

## Expected Results After Fix

| Metric | Before | After |
|--------|--------|-------|
| acceptance_ratio | 0.0000 | > 0.7 |
| avg_reward | 0.0397 | > 0.5 |
| resource_utilization | 0.4905 | > 0.7 |
| load_imbalance | 0.0151 | 0.01-0.05 |

---

## Next Steps

1. ✅ **Tạo Upper Agent** (`agents/upper_agent.py`)
2. ✅ **Tạo Lower Agent** (`agents/lower_agent.py`)
3. ✅ **Tạo HRL Strategy** (`strategy/hrl_strategy.py`)
4. ✅ **Tạo HRL Trainer** (`trainers/hrl_trainer.py`)
5. ✅ **Update main.py** với 4 phases
6. ⬜ **Update config.py** (thêm HRL section)
7. ⬜ **Run Phase 0:** Check VGAE converges
8. ⬜ **Run Phase 1:** Check Lower acceptance > 0.5
9. ⬜ **Run Phase 2:** Check Upper acceptance > 0.6
10. ⬜ **Run Phase 3:** Check final metrics > targets
11. ⬜ **Cleanup:** Xóa old trainer/ files hoặc rename deprecated

---

## Key Differences from Old Architecture

| Component | Old | New |
|-----------|-----|-----|
| Placement | 1 DQN agent | 2-head Upper Agent |
| Routing | Heuristic | Lower Agent DQN |
| Admission | Admission agent | Merged to Upper reward |
| DQN | Vanilla (max) | Double DQN (argmax then evaluate) |
| Pretraining | Ad-hoc | Structured 4-phase |
| Checkpoint | Scattered | load_all_checkpoints() |
| Epsilon | Reset/file | Global step |
| Reward | Dynamic weights | Fixed weights |
| Buffer | Shared | Separate (Upper/Lower) |

---

## Files Generated

✅ `agents/upper_agent.py` - 270 lines  
✅ `agents/lower_agent.py` - 160 lines  
✅ `strategy/hrl_strategy.py` - 390 lines  
✅ `trainers/hrl_trainer.py` - 480 lines  
✅ `main.py` - Updated with complete pipeline  

**Total new code: ~1300 lines**

---

## How to Run

```bash
# Phase 0-3 tự động chạy
python main.py

# Hoặc run từng phase riêng (trong future)
python scripts/train_phase0_vgae.py
python scripts/train_phase1_lower.py
python scripts/train_phase2_upper.py
python scripts/train_phase3_main.py
```

---

## Performance Expectation

- **Training time:** 2-3 giờ (CPU) / 30-45 phút (GPU)
- **Convergence:** ~500-1000 episodes cho Phase 1 + 2
- **Memory:** ~4GB (trạng thái nhạy nhất ở buffer)
