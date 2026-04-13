"""Evaluate the trained PPO agent on CartPole-v1 and save per-episode metrics to CSV."""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import config
from utils import make_env


def main():
    # Path to the model saved by train_ppo.py.
    model_path = os.path.join(config.MODEL_DIR, "ppo_cartpole.zip")

    # Load the trained PPO agent.
    model = PPO.load(model_path)

    # Create a fresh CartPole-v1 environment for evaluation.
    env = make_env()

    # Per-episode tracking lists.
    records = []

    for ep in range(1, config.N_EVAL_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        # Run one full episode to termination or truncation.
        while not done:
            # deterministic=True -> use the greedy policy, not sampling.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated

        # Mark the episode as successful if reward meets the threshold.
        success = total_reward >= config.SUCCESS_THRESHOLD
        records.append({
            "episode": ep,
            "reward": total_reward,
            "length": steps,
            "success": bool(success),
        })

    env.close()

    # Build a DataFrame and print summary statistics.
    df = pd.DataFrame(records)
    mean_reward = df["reward"].mean()
    std_reward = df["reward"].std()
    mean_length = df["length"].mean()
    success_rate = df["success"].mean() * 100.0

    print("\n=== PPO Evaluation Summary ===")
    print(f"Episodes      : {len(df)}")
    print(f"Mean reward   : {mean_reward:.2f}")
    print(f"Std  reward   : {std_reward:.2f}")
    print(f"Mean length   : {mean_length:.2f}")
    print(f"Success rate  : {success_rate:.2f}%  (threshold = {config.SUCCESS_THRESHOLD})")

    # Save per-episode results to CSV.
    os.makedirs(config.LOG_DIR, exist_ok=True)
    csv_path = os.path.join(config.LOG_DIR, "ppo_eval_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-episode results to {csv_path}")


if __name__ == "__main__":
    main()
