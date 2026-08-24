import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path


class Visualizer:

    @staticmethod
    def plot_training_curves(metrics: dict, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        if 'acceptance_ratio' in metrics:
            axes[0, 0].plot(metrics['acceptance_ratio'], label='Acceptance Ratio')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Acceptance Ratio')
            axes[0, 0].set_title('Service Acceptance Ratio')
            axes[0, 0].grid(True)
            axes[0, 0].legend()

        if 'avg_reward' in metrics:
            axes[0, 1].plot(metrics['avg_reward'], label='Average Reward')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].set_title('Average Reward per Episode')
            axes[0, 1].grid(True)
            axes[0, 1].legend()

        if 'total_reward' in metrics:
            axes[1, 0].plot(metrics['total_reward'], label='Total Reward')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Total Reward')
            axes[1, 0].set_title('Total Reward per Episode')
            axes[1, 0].grid(True)
            axes[1, 0].legend()

        if 'epsilon' in metrics:
            axes[1, 1].plot(metrics['epsilon'], label='Epsilon')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].set_title('Exploration Rate (Epsilon)')
            axes[1, 1].grid(True)
            axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_network_graph(p_net, placement_result=None, save_path=None):
        fig, ax = plt.subplots(figsize=(12, 10))

        pos = nx.spring_layout(p_net.graph, k=0.5, iterations=50)

        nx.draw_networkx_nodes(p_net.graph, pos, node_color='lightblue',
                              node_size=300, ax=ax)
        nx.draw_networkx_edges(p_net.graph, pos, ax=ax, alpha=0.5)
        nx.draw_networkx_labels(p_net.graph, pos, ax=ax, font_size=8)

        if placement_result:
            placement_nodes = placement_result.get('placement', [])
            if placement_nodes:
                nx.draw_networkx_nodes(p_net.graph, pos, nodelist=placement_nodes,
                                      node_color='red', node_size=400, ax=ax)

            paths = placement_result.get('paths', [])
            for path in paths:
                if path and len(path) > 1:
                    edges_in_path = [(path[i], path[i+1]) for i in range(len(path)-1)]
                    nx.draw_networkx_edges(p_net.graph, pos, edgelist=edges_in_path,
                                          edge_color='red', width=2, ax=ax)

        ax.set_title('Physical Network Topology')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network graph saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_resource_utilization(p_net, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        cpu_usage = [p_net.nodes[i].cpu_allocated / p_net.nodes[i].cpu_capacity
                     for i in range(p_net.num_nodes)]
        ram_usage = [p_net.nodes[i].ram_allocated / p_net.nodes[i].ram_capacity
                     for i in range(p_net.num_nodes)]
        storage_usage = [p_net.nodes[i].storage_allocated / p_net.nodes[i].storage_capacity
                        for i in range(p_net.num_nodes)]

        axes[0].bar(range(len(cpu_usage)), cpu_usage)
        axes[0].set_xlabel('Node ID')
        axes[0].set_ylabel('CPU Usage (%)')
        axes[0].set_title('CPU Utilization')
        axes[0].set_ylim([0, 1.1])

        axes[1].bar(range(len(ram_usage)), ram_usage)
        axes[1].set_xlabel('Node ID')
        axes[1].set_ylabel('RAM Usage (%)')
        axes[1].set_title('RAM Utilization')
        axes[1].set_ylim([0, 1.1])

        axes[2].bar(range(len(storage_usage)), storage_usage)
        axes[2].set_xlabel('Node ID')
        axes[2].set_ylabel('Storage Usage (%)')
        axes[2].set_title('Storage Utilization')
        axes[2].set_ylim([0, 1.1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Resource utilization plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_comparison_results(results_dict: dict, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        methods = list(results_dict.keys())
        acceptance_ratios = [results_dict[m]['acceptance_ratio'] for m in methods]
        costs = [results_dict[m]['total_cost'] for m in methods]
        path_lengths = [results_dict[m]['avg_path_length'] for m in methods]
        load_imbalances = [results_dict[m]['load_imbalance'] for m in methods]

        x_pos = np.arange(len(methods))

        axes[0, 0].bar(x_pos, acceptance_ratios)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(methods, rotation=45)
        axes[0, 0].set_ylabel('Acceptance Ratio')
        axes[0, 0].set_title('Service Acceptance Ratio Comparison')
        axes[0, 0].set_ylim([0, 1.0])

        axes[0, 1].bar(x_pos, costs)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(methods, rotation=45)
        axes[0, 1].set_ylabel('Total Cost')
        axes[0, 1].set_title('Deployment Cost Comparison')

        axes[1, 0].bar(x_pos, path_lengths)
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(methods, rotation=45)
        axes[1, 0].set_ylabel('Average Path Length')
        axes[1, 0].set_title('Average Path Length Comparison')

        axes[1, 1].bar(x_pos, load_imbalances)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(methods, rotation=45)
        axes[1, 1].set_ylabel('Load Imbalance')
        axes[1, 1].set_title('Load Imbalance Comparison')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison results saved to {save_path}")

        plt.show()