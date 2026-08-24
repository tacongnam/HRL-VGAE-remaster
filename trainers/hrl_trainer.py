"""
HRL Trainer: Orchestrates pretraining phases and main training loop
Phases:
1. Pretrain VGAE
2. Pretrain Lower
3. Pretrain Upper
4. Main training with both agents
"""
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict


class HRLTrainer:
    """Trainer cho Hierarchical RL Strategy"""
    
    def __init__(self, strategy, config, device='cpu'):
        self.strategy = strategy
        self.config = config
        self.device = device
        
        self.logs = defaultdict(list)
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def pretrain_vgae(self, p_nets, num_epochs=100):
        """
        Phase 0: Pretrain VGAE
        Input: danh sách physical network samples
        Output: vgae_weights.pth (khóa vĩnh viễn sau đó)
        """
        print("\n" + "="*60)
        print("PHASE 0: VGAE Pretraining")
        print("="*60)
        
        vgae_model = self.strategy.vgae_model
        vgae_model.train()
        
        optimizer = torch.optim.Adam(vgae_model.parameters(), lr=1e-3)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for p_net in p_nets:
                # Chuẩn bị data
                node_features = torch.FloatTensor(p_net.get_node_features()).to(self.device)
                edge_index = p_net.get_edge_index()
                
                # Forward pass
                z, mu, logvar = vgae_model.encode(node_features, edge_index)
                recon_loss = vgae_model.decode_loss(z, edge_index)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                loss = recon_loss + kl_loss / node_features.size(0)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(p_nets)
            self.logs['vgae_loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Đóng băng VGAE vĩnh viễn
        vgae_model.eval()
        for param in vgae_model.parameters():
            param.requires_grad = False
        
        # Lưu checkpoint
        save_path = self.checkpoint_dir / 'vgae.pth'
        torch.save(vgae_model.state_dict(), save_path)
        print(f"✓ VGAE pretrained and frozen, saved to {save_path}\n")
        
        return vgae_model
    
    def pretrain_lower(self, environment, num_episodes=500):
        """
        Phase 1: Pretrain Lower Agent
        Teacher: Heuristic cân bằng tải (LeastLoadedRoute)
        Input: Environment + VGAE đã khóa
        Output: lower_agent.pth (khóa vĩnh viễn trong Pha 2)
        """
        print("\n" + "="*60)
        print("PHASE 1: Lower Agent Pretraining (with teacher heuristic)")
        print("="*60)
        
        lower_agent = self.strategy.lower_agent
        lower_agent.unfreeze()
        
        batch_size = self.config.get('lower_batch_size', 32)
        train_frequency = self.config.get('lower_train_frequency', 10)
        
        for episode in range(num_episodes):
            # Reset environment
            obs = environment.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                # Teacher policy: chọn route có link pressure thấp nhất
                # Hoặc dùng epsilon-greedy từ đầu
                epsilon = 0.5 * (1 - episode / num_episodes)
                
                # Xây dựng state cho Lower
                # (Giả sử environment cung cấp thông tin này)
                dc_idx = obs.get('current_dc', 0)
                vnf = obs.get('current_vnf', None)
                node_pressures = obs.get('node_pressures', {})
                link_pressures = obs.get('link_pressures', {})
                
                state = self.strategy.build_lower_state(dc_idx, vnf, node_pressures, link_pressures)
                
                # Action từ Lower Agent
                action = lower_agent.act(state, epsilon=epsilon)
                
                # Execute action trong environment
                next_obs, reward, done, info = environment.step({
                    'agent': 'lower',
                    'action': action
                })
                
                # Store transition
                next_state = self.strategy.build_lower_state(
                    next_obs.get('current_dc', 0),
                    next_obs.get('current_vnf', None),
                    next_obs.get('node_pressures', {}),
                    next_obs.get('link_pressures', {})
                )
                
                lower_agent.remember(state, action, reward, next_state, float(done))
                
                episode_reward += reward
                obs = next_obs
                step += 1
                
                # Train Lower Agent
                if step % train_frequency == 0:
                    loss = lower_agent.train(batch_size)
                    if loss is not None:
                        self.logs['lower_loss'].append(loss)
            
            self.logs['lower_episode_reward'].append(episode_reward)
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.logs['lower_episode_reward'][-50:])
                print(f"  Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.4f}")
        
        # Đóng băng Lower Agent
        lower_agent.freeze()
        
        # Lưu checkpoint
        save_path = self.checkpoint_dir / 'lower_agent.pth'
        lower_agent.save_weights(str(save_path))
        print(f"✓ Lower Agent pretrained and frozen, saved to {save_path}\n")
        
        return lower_agent
    
    def pretrain_upper(self, environment, sfc_generator, num_episodes=1000):
        """
        Phase 2: Pretrain Upper Agent
        Teacher: RandomFit / EDF heuristic (warm-start)
        Input: Environment + VGAE đã khóa + Lower đã khóa
        Output: upper_schedule.pth, upper_place.pth
        """
        print("\n" + "="*60)
        print("PHASE 2: Upper Agent Pretraining (with teacher heuristic)")
        print("="*60)
        
        upper_agent = self.strategy.upper_agent
        upper_agent.unfreeze()
        
        batch_size = self.config.get('upper_batch_size', 32)
        train_frequency = self.config.get('upper_train_frequency', 10)
        
        pending_queue = []
        global_step = 0
        total_steps = num_episodes * 100  # Estimate
        
        for episode in range(num_episodes):
            # Reset environment
            environment.reset()
            pending_queue = []
            episode_reward = 0
            episode_accepted = 0
            episode_rejected = 0
            
            # Sinh request cho episode này
            num_requests_per_episode = np.random.randint(5, 15)
            new_requests = sfc_generator.generate(num_requests_per_episode)
            pending_queue.extend(new_requests)
            
            # Xử lý từng request trong queue
            while len(pending_queue) > 0:
                # Build state cho scheduling
                node_pressures = environment.get_node_pressures()
                schedule_state = self.strategy.build_schedule_state(pending_queue, node_pressures)
                
                # Action 1: Chọn request
                epsilon = self.strategy.epsilon_schedule(global_step, total_steps)
                req_idx = upper_agent.act_schedule(schedule_state, epsilon=epsilon)
                req_idx = min(req_idx, len(pending_queue) - 1)
                
                request = pending_queue.pop(req_idx)
                placed_vnfs = []
                accepted = True
                
                # Xử lý từng VNF trong request
                locz = 0.0  # Vị trí VNF trước đó
                for vnf_idx, vnf in enumerate(request.vnfs):
                    # Build state cho placement
                    node_pressure = node_pressures.get(0, 0.5)
                    place_state = self.strategy.build_place_state(vnf, locz, node_pressure)
                    
                    # Action 2: Chọn DC cho VNF
                    dc_idx = upper_agent.act_place(place_state, epsilon=epsilon)
                    dc_idx = min(dc_idx, self.strategy.max_dcs - 1)
                    
                    # Try placement
                    success, placement_cost = environment.place_vnf(vnf, dc_idx)
                    
                    if success:
                        placed_vnfs.append(vnf)
                        # Update locz cho VNF tiếp theo
                        locz = (vnf_idx + 1) / max(len(request.vnfs), 1)
                        
                        # Lower Agent: chọn route cụ thể (đã khóa)
                        link_pressures = environment.get_link_pressures()
                        lower_state = self.strategy.build_lower_state(dc_idx, vnf, node_pressures, link_pressures)
                        
                        # Lower agent act (không train, chỉ inference)
                        with torch.no_grad():
                            route_idx = self.strategy.lower_agent.act(lower_state, epsilon=0.0)
                        
                        # Execute routing
                        route_reward = environment.execute_routing(dc_idx, vnf, route_idx)
                    else:
                        accepted = False
                        break
                
                # Compute reward cho Upper
                if accepted and placed_vnfs:
                    upper_reward = self.strategy.compute_upper_reward(request, placed_vnfs, True)
                    episode_accepted += 1
                else:
                    upper_reward = self.strategy.compute_upper_reward(request, [], False)
                    episode_rejected += 1
                
                # Store transitions (simplified: lưu lần cuối của mỗi action type)
                # Trong thực tế nên lưu từng VNF transition
                upper_agent.remember_schedule(schedule_state, req_idx, upper_reward, 
                                             schedule_state, float(len(pending_queue) == 0))
                
                if placed_vnfs:
                    upper_agent.remember_place(place_state, dc_idx, upper_reward,
                                              place_state, True)
                
                episode_reward += upper_reward
                global_step += 1
                
                # Train Upper Agent
                if global_step % train_frequency == 0:
                    loss_sch, loss_pl = self.strategy.train_step_upper(batch_size)
                    if loss_sch is not None:
                        self.logs['upper_loss_schedule'].append(loss_sch)
                    if loss_pl is not None:
                        self.logs['upper_loss_place'].append(loss_pl)
            
            self.logs['upper_episode_reward'].append(episode_reward)
            self.logs['acceptance_ratio'].append(episode_accepted / max(episode_accepted + episode_rejected, 1))
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.logs['upper_episode_reward'][-100:])
                avg_accept = np.mean(self.logs['acceptance_ratio'][-100:])
                print(f"  Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.4f}, Accept Ratio: {avg_accept:.4f}")
        
        # Lưu checkpoint
        save_sch = self.checkpoint_dir / 'upper_schedule.pth'
        save_pl = self.checkpoint_dir / 'upper_place.pth'
        upper_agent.save_weights(str(save_sch), str(save_pl))
        print(f"✓ Upper Agent pretrained, saved to {save_sch} and {save_pl}\n")
        
        return upper_agent
    
    def train_main(self, environment, sfc_generator, num_episodes=2000):
        """
        Phase 3: Main Training Loop
        Cả 2 agent đều bật:
        - Upper: train bình thường
        - Lower: finetune nhẹ với learning rate rất nhỏ (hoặc khóa hoàn toàn)
        
        Điểm mấu chốt: load_all_checkpoints() MỘT LẦN DUY NHẤT trước vòng lặp
        """
        print("\n" + "="*60)
        print("PHASE 3: Main Training (Online)")
        print("="*60)
        
        # Load all checkpoints - MỘT NƠI DUY NHẤT
        self.strategy.load_all_checkpoints(str(self.checkpoint_dir))
        
        upper_agent = self.strategy.upper_agent
        lower_agent = self.strategy.lower_agent
        
        # Có thể mở khóa Lower để finetune nhẹ (nhưng learning rate rất nhỏ)
        # lower_agent.unfreeze()  # Bỏ comment nếu muốn finetune Lower
        lower_agent.freeze()  # Giữ Lower cố định trong training chính
        
        batch_size_upper = self.config.get('upper_batch_size', 32)
        batch_size_lower = self.config.get('lower_batch_size', 32)
        train_frequency = self.config.get('train_frequency', 10)
        
        pending_queue = []
        global_step = self.strategy.global_step
        total_steps = num_episodes * 100  # Estimate
        
        for episode in range(num_episodes):
            environment.reset()
            pending_queue = []
            episode_reward_upper = 0
            episode_reward_lower = 0
            episode_accepted = 0
            episode_rejected = 0
            
            # Generate requests
            num_requests = np.random.randint(5, 15)
            new_requests = sfc_generator.generate(num_requests)
            pending_queue.extend(new_requests)
            
            # Process queue
            while len(pending_queue) > 0:
                # Get node pressures
                node_pressures = environment.get_node_pressures()
                link_pressures = environment.get_link_pressures()
                
                # Upper: Schedule
                schedule_state = self.strategy.build_schedule_state(pending_queue, node_pressures)
                epsilon = self.strategy.epsilon_schedule(global_step, total_steps)
                
                req_idx = upper_agent.act_schedule(schedule_state, epsilon=epsilon)
                req_idx = min(req_idx, len(pending_queue) - 1)
                request = pending_queue.pop(req_idx)
                
                placed_vnfs = []
                accepted = True
                
                # Process VNFs
                locz = 0.0
                for vnf_idx, vnf in enumerate(request.vnfs):
                    # Upper: Placement
                    node_pressure = node_pressures.get(0, 0.5)
                    place_state = self.strategy.build_place_state(vnf, locz, node_pressure)
                    
                    dc_idx = upper_agent.act_place(place_state, epsilon=epsilon)
                    dc_idx = min(dc_idx, self.strategy.max_dcs - 1)
                    
                    # Try placement
                    success, placement_cost = environment.place_vnf(vnf, dc_idx)
                    
                    if success:
                        placed_vnfs.append(vnf)
                        locz = (vnf_idx + 1) / max(len(request.vnfs), 1)
                        
                        # Lower: Routing (có thể train hoặc không)
                        lower_state = self.strategy.build_lower_state(dc_idx, vnf, node_pressures, link_pressures)
                        route_idx = lower_agent.act(lower_state, epsilon=epsilon if not lower_agent.policy_net.training else epsilon)
                        
                        # Execute routing
                        route_reward = environment.execute_routing(dc_idx, vnf, route_idx)
                        
                        # Store for Lower
                        next_lower_state = self.strategy.build_lower_state(dc_idx, vnf, node_pressures, link_pressures)
                        lower_agent.remember(lower_state, route_idx, route_reward, next_lower_state, False)
                        
                        episode_reward_lower += route_reward
                    else:
                        accepted = False
                        break
                
                # Compute Upper reward
                upper_reward = self.strategy.compute_upper_reward(request, placed_vnfs, accepted)
                
                # Store for Upper
                upper_agent.remember_schedule(schedule_state, req_idx, upper_reward, schedule_state, False)
                if placed_vnfs:
                    upper_agent.remember_place(place_state, dc_idx, upper_reward, place_state, True)
                
                episode_reward_upper += upper_reward
                
                if accepted:
                    episode_accepted += 1
                else:
                    episode_rejected += 1
                
                global_step += 1
                
                # Train
                if global_step % train_frequency == 0:
                    loss_sch, loss_pl = self.strategy.train_step_upper(batch_size_upper)
                    if loss_sch is not None:
                        self.logs['upper_loss_schedule'].append(loss_sch)
                    if loss_pl is not None:
                        self.logs['upper_loss_place'].append(loss_pl)
                    
                    if lower_agent.policy_net.training:
                        loss_lower = self.strategy.train_step_lower(batch_size_lower)
                        if loss_lower is not None:
                            self.logs['lower_loss'].append(loss_lower)
            
            self.logs['upper_episode_reward'].append(episode_reward_upper)
            self.logs['lower_episode_reward'].append(episode_reward_lower)
            self.logs['acceptance_ratio'].append(episode_accepted / max(episode_accepted + episode_rejected, 1))
            
            if (episode + 1) % 100 == 0:
                avg_reward_upper = np.mean(self.logs['upper_episode_reward'][-100:])
                avg_reward_lower = np.mean(self.logs['lower_episode_reward'][-100:])
                avg_accept = np.mean(self.logs['acceptance_ratio'][-100:])
                print(f"  Episode {episode+1}/{num_episodes}")
                print(f"    Upper Reward: {avg_reward_upper:.4f}")
                print(f"    Lower Reward: {avg_reward_lower:.4f}")
                print(f"    Accept Ratio: {avg_accept:.4f}")
        
        self.strategy.global_step = global_step
        
        # Save final checkpoints
        self.strategy.save_all_checkpoints(str(self.checkpoint_dir))
        print(f"✓ Main training completed\n")
    
    def get_logs(self):
        """Return all logged metrics"""
        return dict(self.logs)
