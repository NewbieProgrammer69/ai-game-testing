"""Evaluate the trained Behavior Cloning agent on LunarLander-v3 and save per-episode metrics."""

import os
import numpy as np
import pandas as pd
import torch

import config
from ll_bc_model import LLBCNetwork
from utils import make_ll_env


def main():
    # Load the trained BC network.
    model_path = os.path.join(config.MODEL_DIR, "bc_lunarlander.pth")
    model = LLBCNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    env = make_ll_env()
    records = []

    for ep in range(1, config.LL_N_EVAL_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        # Run one episode using the BC network as the policy.
        while not done:
            action = model.predict(np.asarray(obs, dtype=np.float32))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        success = total_reward >= config.LL_SUCCESS_THRESHOLD
        records.append({
            "episode": ep,
            "reward": total_reward,
            "length": steps,
            "success": bool(success),
        })

    env.close()

    # Summary statistics.
    df = pd.DataFrame(records)
    mean_reward = df["reward"].mean()
    std_reward = df["reward"].std()
    mean_length = df["length"].mean()
    success_rate = df["success"].mean() * 100.0

    print("\n=== BC LunarLander Evaluation Summary ===")
    print(f"Episodes      : {len(df)}")
    print(f"Mean reward   : {mean_reward:.2f}")
    print(f"Std  reward   : {std_reward:.2f}")
    print(f"Mean length   : {mean_length:.2f}")
    print(f"Success rate  : {success_rate:.2f}%  (threshold = {config.LL_SUCCESS_THRESHOLD})")

    # Save per-episode results.
    os.makedirs(config.LOG_DIR, exist_ok=True)
    csv_path = os.path.join(config.LOG_DIR, "ll_bc_eval_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-episode results to {csv_path}")


if __name__ == "__main__":
    main()
