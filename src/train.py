import os
import argparse
import rlcard
import torch
import numpy as np
import random
import json
from src.agent import DQNAgent
from src.utils import plot_curve

def train(args):
    # rl card environment
    env = rlcard.make('blackjack')
    eval_env = rlcard.make('blackjack')

    env.seed(args.seed)
    eval_env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # list of ints for blackjack env
    if isinstance(env.state_shape, list):
        state_shape = env.state_shape[0]
    else:
        state_shape = env.state_shape
        
    num_actions = env.num_actions
    agent = DQNAgent(num_actions, state_shape, hidden_sizes=args.hidden_sizes, lr=args.lr, device=args.device)

    rewards = []
    avg_rewards = []
    win_rates = []
    epsilon_values = []
    episodes_logged = []

    for episode in range(args.num_episodes):
        state, _ = env.reset()
        # env observation vector
        state = state['obs']
        
        total_reward = 0
        
        while not env.is_over():
            action = agent.step(state)
            next_state, _ = env.step(action)
            
            if env.is_over():
                payoffs = env.get_payoffs()
                reward = payoffs[0] # Player 0
                done = True
            else:
                reward = 0
                done = False
            
            agent.memory.push(state, action, reward, next_state['obs'], done)
            state = next_state['obs']
            total_reward += reward
            
            agent.train()
        
        # update target network
        if episode % args.target_update == 0:
            agent.update_target_network()
            
        rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            win_rate = sum(1 for r in rewards[-100:] if r > 0) / min(len(rewards), 100)
            avg_rewards.append(avg_reward)
            win_rates.append(win_rate)
            epsilon_values.append(agent.epsilon)
            episodes_logged.append(episode)
            #logging
            print(f"Episode {episode}, Reward: {total_reward}, Avg Reward: {avg_reward:.2f}, Win Rate: {win_rate:.2f}, Epsilon: {agent.epsilon:.2f}")

    # save after training
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    agent.save(os.path.join(args.save_dir, 'dqn_agent.pth'))
    
    # save metrics
    metrics = {
        'episodes': [int(e) for e in episodes_logged],
        'avg_rewards': [float(r) for r in avg_rewards],
        'win_rates': [float(w) for w in win_rates],
        'epsilon_values': [float(e) for e in epsilon_values],
        'all_rewards': [float(r) for r in rewards]
    }
    with open(os.path.join(args.save_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    plot_curve(avg_rewards, os.path.join(args.save_dir, 'training_curve.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--target_update', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    train(args)
