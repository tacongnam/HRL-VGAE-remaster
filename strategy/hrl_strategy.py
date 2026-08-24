"""
Strategy: Hierarchical Reinforcement Learning Strategy
Điều phối hai tầng Upper và Lower với các phase pretrain tuần tự
"""
import numpy as np
import torch
from agents.upper_agent import UpperAgent
from agents.lower_agent import LowerAgent


class HRLStrategy:
    """
    Hierarchical RL Strategy với 2 tầng:
    - Upper: Request Scheduler + Coarse DC Placement
    - Lower: Load Balancer + Routing (đã khóa trong Pha 2)
    
    Các pha:
    1. Pretrain VGAE (khóa vĩnh viễn sau đó)
    2. Pretrain Lower (khóa vĩnh viễn, dùng làm black-box trong Pha 3)
    3. Pretrain Upper (Lower đã khóa)
    4. Finetune xen kẽ (tùy chọn, sau khi Pha 3 ổn định)
    """
    
    def __init__(self, 
                 p_net, 
                 vgae_model,
                 latent_dim=32,
                 max_dcs=10,
                 top_k_candidates=8,
                 k_routes=3,
                 upper_hidden_dim=128,
                 lower_hidden_dim=64,
                 device='cpu'):
        self.p_net = p_net
        self.vgae_model = vgae_model
        self.latent_dim = latent_dim
        self.max_dcs = max_dcs
        self.top_k = top_k_candidates
        self.k_routes = k_routes
        self.device = device
        
        # Khởi tạo Upper Agent
        self.upper_agent = UpperAgent(
            latent_dim=latent_dim,
            max_dcs=max_dcs,
            top_k_candidates=top_k_candidates,
            hidden_dim=upper_hidden_dim,
            gamma=0.95,
            learning_rate=1e-3,
            device=device
        )
        
        # Khởi tạo Lower Agent
        # State dim = node_pressure (1) + per_resource_load (3) + link_pressure (1) + vnf_demand (varies) ≈ 20
        lower_state_dim = 20
        self.lower_agent = LowerAgent(
            state_dim=lower_state_dim,
            k_candidates=k_routes,
            hidden_dim=lower_hidden_dim,
            gamma=0.9,
            learning_rate=1e-3,
            device=device
        )
        
        # Global step counter (xuyên suốt toàn bộ training)
        self.global_step = 0
        
        # Reward weights (cố định, không học)
        self.w1_r2c = 1.0  # revenue/cost weight
        self.w2_sla_penalty = 0.3  # SLA violation penalty
        self.w3_reject_penalty = 1.0  # Rejection penalty
        
        # Epsilon schedule
        self.epsilon_max = 0.5
        self.epsilon_min = 0.05
    
    def epsilon_schedule(self, current_step, total_steps=None):
        """
        Tính epsilon theo global step (không reset theo file)
        Linear decay từ epsilon_max đến epsilon_min
        """
        if total_steps is None:
            total_steps = 1000000  # Default nếu không biết tổng
        
        progress = min(current_step / total_steps, 1.0)
        epsilon = self.epsilon_max - progress * (self.epsilon_max - self.epsilon_min)
        return max(epsilon, self.epsilon_min)
    
    def build_global_state(self, pending_queue, node_pressures):
        """
        Xây dựng global network state
        Returns:
            globalz: VGAE embedding trung bình
            queue_pressure: số request đang chờ / max queue size
            avg_urgency: trung bình urgency của toàn queue
        """
        # Tính VGAE embedding
        with torch.no_grad():
            p_net_features = torch.FloatTensor(self.p_net.get_node_features()).to(self.device)
            p_net_edges = self.p_net.get_edge_index()
            z, _, _ = self.vgae_model.encode(p_net_features, p_net_edges)
            globalz = z.mean(dim=0).cpu().numpy()  # Shape: (latent_dim,)
        
        # Queue pressure
        MAX_QUEUE_SIZE = 100  # Config này
        queue_pressure = min(len(pending_queue) / MAX_QUEUE_SIZE, 1.0)
        
        # Average urgency
        avg_urgency = 0.0
        if len(pending_queue) > 0:
            urgencies = []
            for req in pending_queue:
                # urgency = (delay_max - waiting_time) / delay_max
                # Giả sử request có waiting_time attribute
                if hasattr(req, 'delay_max') and hasattr(req, 'waiting_time'):
                    urg = (req.delay_max - req.waiting_time) / req.delay_max
                else:
                    urg = 0.5  # Default
                urgencies.append(urg)
            avg_urgency = np.mean(urgencies)
        
        return globalz, queue_pressure, avg_urgency
    
    def build_request_features(self, request):
        """
        Xây dựng đặc trưng của 1 request
        Returns: ndarray shape (5,)
            - total resource demand (normalized)
            - bandwidth
            - urgency
            - revenue estimate
            - number of VNFs
        """
        total_resource = getattr(request, 'cpu_demand', 0) + getattr(request, 'ram_demand', 0)
        total_resource = min(total_resource / 1000.0, 1.0)  # Normalize
        
        bw = getattr(request, 'bw_demand', 0) / 100.0  # Normalize
        
        if hasattr(request, 'delay_max') and hasattr(request, 'waiting_time'):
            urgency = (request.delay_max - request.waiting_time) / request.delay_max
        else:
            urgency = 0.5
        
        revenue = getattr(request, 'revenue', 1.0) / 100.0  # Normalize
        
        num_vnfs = len(getattr(request, 'vnfs', []))
        num_vnfs_norm = min(num_vnfs / 10.0, 1.0)  # Normalize
        
        return np.array([total_resource, bw, urgency, revenue, num_vnfs_norm])
    
    def build_schedule_state(self, pending_queue, node_pressures):
        """
        Xây dựng state cho action scheduling (chọn request từ queue)
        Returns: ndarray shape (schedule_input_dim,)
        """
        globalz, queue_pressure, avg_urgency = self.build_global_state(pending_queue, node_pressures)
        
        # Lấy top-K candidates từ queue
        candidates = pending_queue[:self.top_k]
        
        # Pad zeros nếu queue có ít hơn K candidates
        req_features_list = []
        for i in range(self.top_k):
            if i < len(candidates):
                feat = self.build_request_features(candidates[i])
            else:
                feat = np.zeros(5)
            req_features_list.append(feat)
        
        req_features_flat = np.concatenate(req_features_list)
        
        # Ghép toàn bộ
        state = np.concatenate([
            globalz,
            [queue_pressure, avg_urgency],
            req_features_flat
        ])
        
        return state
    
    def build_place_state(self, vnf, locz, node_pressure):
        """
        Xây dựng state cho action placement (chọn DC cho VNF)
        Returns: ndarray shape (place_input_dim,)
        """
        globalz, _, _ = self.build_global_state([], {})
        
        # VNF features
        vnf_feat = np.array([
            getattr(vnf, 'cpu', 0) / 100.0,
            getattr(vnf, 'ram', 0) / 100.0,
            getattr(vnf, 'storage', 0) / 100.0,
            getattr(vnf, 'bw', 0) / 100.0,
            0.5  # Default feature
        ])
        
        # locz = position của VNF trước đó trong SFC (0-1 normalized)
        if locz is None:
            locz = 0.0
        
        # node_pressure = áp lực tài nguyên hiện tại
        if node_pressure is None:
            node_pressure = 0.5
        
        state = np.concatenate([
            globalz,
            vnf_feat,
            [locz, node_pressure]
        ])
        
        return state
    
    def build_lower_state(self, dc_idx, vnf, node_pressures, link_pressures):
        """
        Xây dựng state cho action lower (chọn route hoặc slot)
        Returns: ndarray shape (state_dim,)
        """
        # Node pressure của DC này
        node_pressure = node_pressures.get(dc_idx, 0.5)
        
        # Per-resource load
        per_resource_load = np.array([
            0.5,  # CPU load (placeholder)
            0.5,  # RAM load
            0.5   # Storage load
        ])
        
        # Link pressure trung bình của các link nối tới DC này
        link_pressure_avg = link_pressures.get(dc_idx, 0.5)
        
        # VNF demand
        vnf_demand = np.array([
            getattr(vnf, 'cpu', 0) / 100.0,
            getattr(vnf, 'ram', 0) / 100.0,
            getattr(vnf, 'bw', 0) / 100.0
        ])
        
        # Pad to state_dim = 20
        state = np.concatenate([
            [node_pressure],
            per_resource_load,
            [link_pressure_avg],
            vnf_demand,
            np.zeros(20 - 1 - 3 - 1 - 3)  # Pad remaining
        ])
        
        return state[:20]  # Ensure exactly 20
    
    def compute_upper_reward(self, request, placed_vnfs, accepted):
        """
        Tính reward cho Upper Agent
        R_upper = R_BASE + w1*r2c - w2*SLAviolation - w3*reject_penalty
        """
        R_BASE = 0.1
        
        # r2c = revenue / cost
        if accepted and placed_vnfs:
            revenue = getattr(request, 'revenue', 1.0)
            cost = sum(getattr(vnf, 'placement_cost', 0.1) for vnf in placed_vnfs)
            r2c = revenue / max(cost, 1e-6)
        else:
            r2c = 0.0
        
        # SLA violation penalty
        if hasattr(request, 'delay_max') and hasattr(request, 'actual_delay'):
            if request.actual_delay > request.delay_max:
                sla_penalty = 1.0
            else:
                sla_penalty = 0.0
        else:
            sla_penalty = 0.0
        
        # Reject penalty
        reject_penalty = 0.0 if accepted else 1.0
        
        reward = R_BASE + self.w1_r2c * r2c - self.w2_sla_penalty * sla_penalty - self.w3_reject_penalty * reject_penalty
        
        return reward
    
    def compute_lower_reward(self, node_pressure_after, link_pressure):
        """
        Tính reward cho Lower Agent (thuần túy cân bằng tải)
        R_lower = -node_pressure - avg_link_pressure
        """
        reward = -node_pressure_after - link_pressure
        return reward
    
    def load_all_checkpoints(self, checkpoint_dir):
        """
        MỘT nơi duy nhất load toàn bộ weights - ĐIỂM SỬA LỖI QUAN TRỌNG
        Không có đường tắt nào bỏ sót weight
        """
        import os
        
        print(f"Loading all checkpoints from {checkpoint_dir}")
        
        # Load VGAE (khóa vĩnh viễn)
        vgae_path = os.path.join(checkpoint_dir, 'vgae.pth')
        if os.path.exists(vgae_path):
            self.vgae_model.load_state_dict(torch.load(vgae_path, map_location=self.device))
            self.vgae_model.freeze()
            print("✓ VGAE loaded and frozen")
        else:
            print("⚠ VGAE checkpoint not found")
        
        # Load Lower Agent (có thể finetune nhẹ)
        lower_path = os.path.join(checkpoint_dir, 'lower_agent.pth')
        if os.path.exists(lower_path):
            self.lower_agent.load_weights(lower_path)
            print("✓ Lower Agent loaded")
        else:
            print("⚠ Lower Agent checkpoint not found")
        
        # Load Upper Agent
        upper_schedule_path = os.path.join(checkpoint_dir, 'upper_schedule.pth')
        upper_place_path = os.path.join(checkpoint_dir, 'upper_place.pth')
        if os.path.exists(upper_schedule_path) and os.path.exists(upper_place_path):
            self.upper_agent.load_weights(upper_schedule_path, upper_place_path)
            print("✓ Upper Agent loaded")
        else:
            print("⚠ Upper Agent checkpoint not found")
    
    def save_all_checkpoints(self, checkpoint_dir):
        """Lưu toàn bộ weights"""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save VGAE
        torch.save(self.vgae_model.state_dict(), 
                   os.path.join(checkpoint_dir, 'vgae.pth'))
        
        # Save Lower Agent
        self.lower_agent.save_weights(
            os.path.join(checkpoint_dir, 'lower_agent.pth'))
        
        # Save Upper Agent
        self.upper_agent.save_weights(
            os.path.join(checkpoint_dir, 'upper_schedule.pth'),
            os.path.join(checkpoint_dir, 'upper_place.pth'))
        
        print(f"✓ All checkpoints saved to {checkpoint_dir}")
    
    def train_step_upper(self, batch_size=32):
        """Train Upper Agent một bước"""
        loss_schedule = self.upper_agent.train_schedule(batch_size)
        loss_place = self.upper_agent.train_place(batch_size)
        return loss_schedule, loss_place
    
    def train_step_lower(self, batch_size=32):
        """Train Lower Agent một bước"""
        loss = self.lower_agent.train(batch_size)
        return loss
