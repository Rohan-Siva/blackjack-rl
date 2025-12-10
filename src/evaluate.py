import argparse
import rlcard
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from src.agent import DQNAgent
from src.mc_agent import MCAgent
from rlcard.agents import RandomAgent

def evaluate(args):
    env = rlcard.make('blackjack')
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load agent
    if isinstance(env.state_shape, list):
        state_shape = env.state_shape[0]
    else:
        state_shape = env.state_shape
    
    num_actions = env.num_actions
    agent = DQNAgent(num_actions, state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    agent.load(args.model_path)
    agent.epsilon = 0.0 # Greedy policy for evaluation

    # baseline is random moves
    random_agent = RandomAgent(num_actions=num_actions)

    print("Evaluating DQN Agent...")
    dqn_rewards = np.array(run_eval(env, agent, args.num_episodes))
    dqn_mean = np.mean(dqn_rewards)
    dqn_std = np.std(dqn_rewards)
    dqn_sem = dqn_std / np.sqrt(len(dqn_rewards))
    dqn_ci = 1.96 * dqn_sem
    
    print(f"DQN Agent Average Reward: {dqn_mean:.4f} ± {dqn_ci:.4f} (95% CI)")
    print(f"DQN Agent Win Rate: {np.sum(dqn_rewards > 0) / args.num_episodes:.4f}")
    print(f"  Std Dev: {dqn_std:.4f}")
    print(f"  SEM: {dqn_sem:.4f}")

    print("\nEvaluating Random Agent...")
    random_rewards = np.array(run_eval(env, random_agent, args.num_episodes))
    rand_mean = np.mean(random_rewards)
    rand_std = np.std(random_rewards)
    rand_sem = rand_std / np.sqrt(len(random_rewards))
    rand_ci = 1.96 * rand_sem
    
    print(f"Random Agent Average Reward: {rand_mean:.4f} ± {rand_ci:.4f} (95% CI)")
    print(f"Random Agent Win Rate: {np.sum(random_rewards > 0) / args.num_episodes:.4f}")
    print(f"  Std Dev: {rand_std:.4f}")
    print(f"  SEM: {rand_sem:.4f}")

    print("\nEvaluating MC Agent...")
    mc_agent = MCAgent(action_num=num_actions)
    if os.path.exists('models/mc_agent.pkl'):
        mc_agent.load('models/mc_agent.pkl')
        mc_rewards = np.array(run_eval(env, mc_agent, args.num_episodes))
        mc_mean = np.mean(mc_rewards)
        mc_std = np.std(mc_rewards)
        mc_sem = mc_std / np.sqrt(len(mc_rewards))
        mc_ci = 1.96 * mc_sem
        
        print(f"MC Agent Average Reward: {mc_mean:.4f} ± {mc_ci:.4f} (95% CI)")
        print(f"MC Agent Win Rate: {np.sum(mc_rewards > 0) / args.num_episodes:.4f}")
        print(f"  Std Dev: {mc_std:.4f}")
        print(f"  SEM: {mc_sem:.4f}")
    else:
        print("MC Agent model not found at models/mc_agent.pkl. Skipping.")
        mc_mean, mc_ci = 0, 0

    os.makedirs('visualizations', exist_ok=True)
    
    means = [dqn_mean, rand_mean, mc_mean]
    cis = [dqn_ci, rand_ci, mc_ci]
    labels = ['DQN Agent', 'Random Agent', 'MC Agent']
    colors = ['#2E86AB', '#D9534F', '#F0A202']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, means, yerr=cis, capsize=10, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, mean, ci in zip(bars, means, cis):
        height = bar.get_height()
        y_pos = height/2
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{mean:.3f}\n±{ci:.3f}',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Agent Performance Comparison (95% CI)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = 'visualizations/evaluation_comparison_mc.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nComparison plot saved to {save_path}")

def run_eval(env, agent, num_episodes):
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        while not env.is_over():
            if isinstance(agent, DQNAgent):
                action = agent.eval_step(state['obs'])
            elif isinstance(agent, MCAgent):
                action = agent.eval_step(state)
            else:
                action = agent.step(state)
            
            next_state, _ = env.step(action)
            state = next_state
        
        rewards.append(env.get_payoffs()[0])
    return rewards

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[64, 64])
    args = parser.parse_args()
    
    evaluate(args)
