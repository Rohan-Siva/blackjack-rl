import argparse
import subprocess
import os
import json
import itertools
import numpy as np

def run_experiment(hidden_sizes, lr, seed, num_episodes, device):
    hidden_sizes_str = ' '.join(map(str, hidden_sizes))
    hidden_sizes_name = '_'.join(map(str, hidden_sizes))
    model_dir = f"models/tune_h{hidden_sizes_name}_lr{lr}"
    
    print(f"Running experiment: Hidden Sizes={hidden_sizes}, LR={lr}")
    
    train_cmd = [
        "python", "-m", "src.train",
        "--save_dir", model_dir,
        "--hidden_sizes", *map(str, hidden_sizes),
        "--lr", str(lr),
        "--num_episodes", str(num_episodes),
        "--seed", str(seed),
        "--device", device
    ]
    subprocess.run(train_cmd, check=True)
    
    with open(os.path.join(model_dir, "training_metrics.json"), "r") as f:
        metrics = json.load(f)
        
    final_avg_reward = metrics['avg_rewards'][-1]
    final_win_rate = metrics['win_rates'][-1]
    
    return {
        "hidden_sizes": hidden_sizes,
        "lr": lr,
        "final_avg_reward": final_avg_reward,
        "final_win_rate": final_win_rate,
        "model_dir": model_dir
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    hidden_sizes_grid = [
        [64, 64],
        [128, 128],
        [64, 64, 64]
    ]
    lr_grid = [0.01, 0.001, 0.0001]
    
    results = []
    
    for hidden_sizes, lr in itertools.product(hidden_sizes_grid, lr_grid):
        try:
            res = run_experiment(hidden_sizes, lr, args.seed, args.num_episodes, args.device)
            results.append(res)
        except Exception as e:
            print(f"Failed for {hidden_sizes}, {lr}: {e}")
            
    results.sort(key=lambda x: x['final_avg_reward'], reverse=True)
    
    print("\n")
    print("HYPERPARAMETER TUNING RESULTS")
    print("\n")
    print(f"{'Hidden Sizes':<20} | {'LR':<10} | {'Avg Reward':<12} | {'Win Rate':<10}")
    print("\n")
    
    for r in results:
        print(f"{str(r['hidden_sizes']):<20} | {r['lr']:<10} | {r['final_avg_reward']:.4f}       | {r['final_win_rate']:.4f}")
        
    with open("tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nBest configuration found:")
    print(results[0])

if __name__ == "__main__":
    main()
