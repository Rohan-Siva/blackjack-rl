import argparse
import rlcard
import torch
import numpy as np
from src.agent import DQNAgent

def play(args):
    env = rlcard.make('blackjack')
    
    agent = None
    if args.model_path:
        if isinstance(env.state_shape, list):
            state_shape = env.state_shape[0]
        else:
            state_shape = env.state_shape
        num_actions = env.num_actions
        agent = DQNAgent(num_actions, state_shape, device=args.device)
        try:
            agent.load(args.model_path)
            agent.epsilon = 0.0
            print(f"Loaded agent from {args.model_path}")
        except Exception as e:
            print(f"Could not load agent: {e}")
            agent = None

    print("Welcome to Blackjack!")
    print("Actions: 0 = Hit, 1 = Stand")
    
    while True:
        print("\n--- New Game ---")
        state, _ = env.reset()
        
        while not env.is_over():
            if 'raw_obs' in state:
                print(f"State: {state['raw_obs']}")
            else:
                print(f"State (obs): {state['obs']}")
            
            if agent:
                action_idx = agent.eval_step(state['obs'])
                action_str = "Hit" if action_idx == 0 else "Stand"
                print(f"Agent suggests: {action_idx} ({action_str})")
            
            action_input = input("Choose action (0: Hit, 1: Stand): ")
            try:
                action = int(action_input)
                if action not in [0, 1]:
                    print("Invalid action.")
                    continue
            except:
                print("Invalid input.")
                continue
            
            next_state, _ = env.step(action)
            state = next_state
        
        payoffs = env.get_payoffs()
        reward = payoffs[0]
        print(f"Game Over. Reward: {reward}")
        if reward > 0:
            print("You Won!")
        elif reward < 0:
            print("You Lost!")
        else:
            print("Push!")
        
        if input("Play again? (y/n): ").lower() != 'y':
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    play(args)
