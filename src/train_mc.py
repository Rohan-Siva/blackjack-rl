import os
import argparse
import rlcard
import numpy as np
import random
import json
from src.mc_agent import MCAgent
from src.utils import plot_curve

def train(args):
    env = rlcard.make('blackjack')
    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    agent = MCAgent(action_num=env.num_actions, epsilon=0.1)
    
    rewards = []
    avg_rewards = []
    win_rates = []
    
    print(f"Starting MC training for {args.num_episodes} episodes...")
    
    for episode in range(args.num_episodes):
        state, _ = env.reset()
        episode_transitions = []
        
        while not env.is_over():
            action = agent.step(state)
            next_state, _ = env.step(action)
            
           
            reward = 0
            if env.is_over():
                reward = env.get_payoffs()[0]
            
            episode_transitions.append((state, action, reward))
            state = next_state
        
        agent.update(episode_transitions)
        rewards.append(episode_transitions[-1][2])
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            win_rate = sum(1 for r in rewards[-100:] if r > 0) / min(len(rewards), 100)
            avg_rewards.append(avg_reward)
            win_rates.append(win_rate)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Win Rate: {win_rate:.2f}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    agent.save(os.path.join(args.save_dir, 'mc_agent.pkl'))
    
    metrics = {
        'avg_rewards': [float(r) for r in avg_rewards],
        'win_rates': [float(w) for w in win_rates]
    }
    with open(os.path.join(args.save_dir, 'mc_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=50000) # MC needs more episodes usually
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()
    
    train(args)
