import numpy as np
from tqdm import tqdm


class DQNTrainer:

    def __init__(self, placement_engine, dqn_agent, config):
        self.placement_engine = placement_engine
        self.dqn_agent = dqn_agent
        self.config = config
        self.metrics = {
            'acceptance_ratio': [],
            'total_reward': [],
            'avg_reward': [],
            'epsilon': [],
        }

    def train_episode(self, sfc_requests):
        episode_reward = 0.0
        success_count = 0
        total_transitions = 0

        for sfc_request in sfc_requests:
            result = self.placement_engine.place_sfc(sfc_request, training=True, 
                                                     use_epsilon_greedy=True)

            for transition in result.get('transitions', []):
                self.dqn_agent.store_transition(
                    transition['state'],
                    transition['action'],
                    transition['reward'],
                    transition['next_state'],
                    transition['done']
                )

                loss = self.dqn_agent.train_step()
                total_transitions += 1

            episode_reward += result['total_reward']
            if result['success']:
                success_count += 1

        acceptance_ratio = success_count / len(sfc_requests) if len(sfc_requests) > 0 else 0
        avg_reward = episode_reward / len(sfc_requests) if len(sfc_requests) > 0 else 0

        return {
            'acceptance_ratio': acceptance_ratio,
            'total_reward': episode_reward,
            'avg_reward': avg_reward,
            'epsilon': self.dqn_agent.epsilon,
            'num_transitions': total_transitions,
        }

    def train(self, num_episodes, sfc_generator):
        for episode in range(num_episodes):
            sfc_requests = sfc_generator.generate(
                self.config.get('num_requests_per_episode', 50))

            episode_metrics = self.train_episode(sfc_requests)

            self.metrics['acceptance_ratio'].append(episode_metrics['acceptance_ratio'])
            self.metrics['total_reward'].append(episode_metrics['total_reward'])
            self.metrics['avg_reward'].append(episode_metrics['avg_reward'])
            self.metrics['epsilon'].append(episode_metrics['epsilon'])

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}: "
                      f"SAR={episode_metrics['acceptance_ratio']:.4f}, "
                      f"AvgReward={episode_metrics['avg_reward']:.4f}, "
                      f"Epsilon={episode_metrics['epsilon']:.4f}")

    def get_metrics(self):
        return self.metrics

    def save_model(self, path):
        self.dqn_agent.save(path)

    def load_model(self, path):
        self.dqn_agent.load(path)