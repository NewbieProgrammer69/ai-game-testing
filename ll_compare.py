"""Compare PPO and BC evaluation results on LunarLander-v3 and save a summary CSV."""

import os
import pandas as pd

import config


def compute_stats(df: pd.DataFrame) -> dict:
    return {
        "mean_reward": df["reward"].mean(),
        "std_reward": df["reward"].std(),
        "mean_length": df["length"].mean(),
        "success_rate": df["success"].mean() * 100.0,
    }


def main():
    # Load per-episode evaluation logs for LunarLander.
    ppo_df = pd.read_csv(os.path.join(config.LOG_DIR, "ll_ppo_eval_results.csv"))
    bc_df = pd.read_csv(os.path.join(config.LOG_DIR, "ll_bc_eval_results.csv"))

    ppo = compute_stats(ppo_df)
    bc = compute_stats(bc_df)

    # Pretty-printed side-by-side table.
    line = "=========================================="
    print(line)
    print(f"|  {'Metric':<19} |   PPO   |   BC   |")
    print(line)
    print(f"|  {'Mean Reward':<19} | {ppo['mean_reward']:>7.1f} | {bc['mean_reward']:>6.1f} |")
    print(f"|  {'Std Reward':<19} | {ppo['std_reward']:>7.1f} | {bc['std_reward']:>6.1f} |")
    print(f"|  {'Mean Episode Length':<19} | {ppo['mean_length']:>7.1f} | {bc['mean_length']:>6.1f} |")
    print(f"|  {'Success Rate (%)':<19} | {ppo['success_rate']:>7.1f} | {bc['success_rate']:>6.1f} |")
    print(line)

    # Save a tidy comparison CSV.
    comparison_df = pd.DataFrame({
        "metric": ["mean_reward", "std_reward", "mean_length", "success_rate"],
        "PPO": [ppo["mean_reward"], ppo["std_reward"], ppo["mean_length"], ppo["success_rate"]],
        "BC":  [bc["mean_reward"],  bc["std_reward"],  bc["mean_length"],  bc["success_rate"]],
    })
    out_path = os.path.join(config.LOG_DIR, "ll_comparison.csv")
    comparison_df.to_csv(out_path, index=False)
    print(f"\nSaved comparison to {out_path}")

    # Short verbal summary.
    print("\n=== LunarLander Summary ===")
    if ppo["mean_reward"] > bc["mean_reward"]:
        diff = ppo["mean_reward"] - bc["mean_reward"]
        print(f"PPO outperformed BC, scoring {diff:.1f} more reward on average across {len(ppo_df)} episodes.")
        print(f"PPO achieved a {ppo['success_rate']:.1f}% success rate versus {bc['success_rate']:.1f}% for BC.")
        print("On LunarLander, reinforcement learning produced a stronger policy than supervised imitation.")
    elif bc["mean_reward"] > ppo["mean_reward"]:
        diff = bc["mean_reward"] - ppo["mean_reward"]
        print(f"BC outperformed PPO, scoring {diff:.1f} more reward on average across {len(bc_df)} episodes.")
        print(f"BC achieved a {bc['success_rate']:.1f}% success rate versus {ppo['success_rate']:.1f}% for PPO.")
        print("Behavior Cloning successfully imitated (and surpassed) the expert on this task.")
    else:
        print(f"PPO and BC tied at {ppo['mean_reward']:.1f} mean reward across {len(ppo_df)} episodes.")
        print(f"Both achieved a {ppo['success_rate']:.1f}% success rate.")
        print("BC fully replicated the expert PPO policy's behavior on LunarLander.")


if __name__ == "__main__":
    main()
