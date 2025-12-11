import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def create_visualizations(metrics_path, output_dir):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    episodes = metrics['episodes']
    avg_rewards = metrics['avg_rewards']
    win_rates = metrics['win_rates']
    epsilon_values = metrics['epsilon_values']
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, avg_rewards, linewidth=2, color='#2E86AB')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward (Last 100 Episodes)', fontsize=12)
    plt.title('DQN Training Progress on Blackjack', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, win_rates, linewidth=2, color='#06A77D')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Win Rate (Last 100 Episodes)', fontsize=12)
    plt.title('Win Rate Progress During Training', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Win Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Epsilon Decay
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, epsilon_values, linewidth=2, color='#D741A7')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Epsilon (Exploration Rate)', fontsize=12)
    plt.title('Epsilon Decay During Training', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epsilon_decay.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(episodes, avg_rewards, linewidth=2, color='#2E86AB')
    axes[0].set_ylabel('Avg Reward', fontsize=11)
    axes[0].set_title('DQN Training Metrics on Blackjack', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(episodes, win_rates, linewidth=2, color='#06A77D')
    axes[1].set_ylabel('Win Rate', fontsize=11)
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(episodes, epsilon_values, linewidth=2, color='#D741A7')
    axes[2].set_xlabel('Episode', fontsize=12)
    axes[2].set_ylabel('Epsilon', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")
    print(f"Final Average Reward: {avg_rewards[-1]:.3f}")
    print(f"Final Win Rate: {win_rates[-1]:.3f}")
    print(f"Final Epsilon: {epsilon_values[-1]:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations')
    args = parser.parse_args()
    
    create_visualizations(args.metrics_path, args.output_dir)
