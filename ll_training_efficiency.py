"""Training efficiency experiment: measure PPO learning curve and BC training cost.

Answers: "How much computational effort does each method need to produce a working agent?"

We retrain PPO from scratch and pause every 25,000 timesteps to evaluate on 20 episodes,
recording both environment steps consumed and wall-clock time elapsed. We then time a full
BC training run on the existing demo data. The two cost profiles let us plot learning
curves and the efficiency/performance tradeoff.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from stable_baselines3 import PPO

import config
from ll_bc_model import LLBCNetwork
from utils import make_ll_env, ensure_dirs


# Evaluate PPO every CHECKPOINT_STEPS environment steps.
CHECKPOINT_STEPS = 25_000

# Number of episodes for each mid-training checkpoint evaluation.
CHECKPOINT_EVAL_EPISODES = 20


def evaluate_policy_simple(predict_fn, n_episodes: int):
    """Run `n_episodes` full episodes using the given predict function and return metrics.

    `predict_fn(obs)` must return an integer action. Works for both BC and PPO.
    """
    env = make_ll_env()
    rewards, successes = [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        successes.append(total_reward >= config.LL_SUCCESS_THRESHOLD)

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes) * 100.0),
    }


def ppo_predict_factory(model: PPO):
    """Return a predict function that always picks the deterministic action from PPO."""
    def predict(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return predict


def bc_predict_factory(model: LLBCNetwork):
    """Return a predict function that wraps LLBCNetwork.predict."""
    def predict(obs):
        return model.predict(np.asarray(obs, dtype=np.float32))
    return predict


def train_ppo_with_checkpoints():
    """Train PPO in chunks of CHECKPOINT_STEPS and evaluate after each chunk."""
    env = make_ll_env()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.LL_PPO_LEARNING_RATE,
        n_steps=config.LL_PPO_N_STEPS,
        batch_size=config.LL_PPO_BATCH_SIZE,
        n_epochs=config.LL_PPO_N_EPOCHS,
        gamma=config.LL_PPO_GAMMA,
        clip_range=config.LL_PPO_CLIP_RANGE,
        verbose=0,
        device="auto",
    )

    total_target = config.LL_PPO_TOTAL_TIMESTEPS
    checkpoints = []

    start_time = time.time()
    steps_done = 0

    # Train in fixed-size chunks until we hit the target.
    while steps_done < total_target:
        chunk = min(CHECKPOINT_STEPS, total_target - steps_done)
        # reset_num_timesteps=False so SB3 keeps counting from the previous chunk.
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk
        wall_time = time.time() - start_time

        # Evaluate the current policy for CHECKPOINT_EVAL_EPISODES episodes.
        metrics = evaluate_policy_simple(ppo_predict_factory(model), CHECKPOINT_EVAL_EPISODES)

        print(
            f"PPO @ {steps_done} steps | Time: {wall_time:6.1f}s | "
            f"Mean Reward: {metrics['mean_reward']:7.2f} | "
            f"Success: {metrics['success_rate']:5.1f}%"
        )

        checkpoints.append({
            "timesteps": steps_done,
            "wall_time_seconds": wall_time,
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
            "success_rate": metrics["success_rate"],
        })

    env.close()
    total_time = time.time() - start_time
    return checkpoints, total_time


def train_bc_timed():
    """Train a fresh BC network on the clean demo data and return timing + metrics."""
    # Load demos.
    demo_path = os.path.join(config.DEMO_DIR, "ll_demo_data.npz")
    data = np.load(demo_path)
    obs_tensor = torch.tensor(data["observations"], dtype=torch.float32)
    act_tensor = torch.tensor(data["actions"], dtype=torch.long)

    # Standard train/val split, same convention as the rest of the codebase.
    dataset = TensorDataset(obs_tensor, act_tensor)
    n_train = int(len(dataset) * config.LL_BC_TRAIN_SPLIT)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_set, batch_size=config.LL_BC_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.LL_BC_BATCH_SIZE, shuffle=False)

    model = LLBCNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LL_BC_LEARNING_RATE)

    # Time the full training loop.
    start_time = time.time()
    for epoch in range(1, config.LL_BC_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Run a validation pass too — part of honest wall-clock training time.
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                _ = model(xb)
    total_train_time = time.time() - start_time

    # Evaluate for 20 episodes.
    metrics = evaluate_policy_simple(bc_predict_factory(model), CHECKPOINT_EVAL_EPISODES)
    print(
        f"BC trained in {total_train_time:.1f}s | "
        f"Mean Reward: {metrics['mean_reward']:7.2f} | "
        f"Success: {metrics['success_rate']:5.1f}%"
    )

    return total_train_time, metrics


def main():
    ensure_dirs()

    print("=== Training PPO with checkpoint evaluations ===")
    ppo_checkpoints, ppo_total_time = train_ppo_with_checkpoints()

    print("\n=== Training BC (timed) ===")
    bc_total_time, bc_metrics = train_bc_timed()

    # --- Save PPO learning curve CSV ---
    ppo_df = pd.DataFrame(ppo_checkpoints)
    ppo_out = os.path.join(config.LOG_DIR, "ll_ppo_learning_curve.csv")
    ppo_df.to_csv(ppo_out, index=False)

    # --- Save BC timing CSV ---
    bc_df = pd.DataFrame([{
        "total_train_time_seconds": bc_total_time,
        "mean_reward": bc_metrics["mean_reward"],
        "std_reward": bc_metrics["std_reward"],
        "success_rate": bc_metrics["success_rate"],
    }])
    bc_out = os.path.join(config.LOG_DIR, "ll_bc_timing.csv")
    bc_df.to_csv(bc_out, index=False)

    # --- Final comparison summary ---
    final_ppo = ppo_checkpoints[-1]
    print("\n=== Training Efficiency Summary ===")
    print(f"PPO total training time : {ppo_total_time:8.1f} s")
    print(f"BC  total training time : {bc_total_time:8.1f} s")
    if bc_total_time > 0:
        print(f"PPO / BC time ratio     : {ppo_total_time / bc_total_time:8.1f}x")
    print()
    print(f"PPO final mean reward   : {final_ppo['mean_reward']:7.2f}  "
          f"(success {final_ppo['success_rate']:.1f}%)")
    print(f"BC  final mean reward   : {bc_metrics['mean_reward']:7.2f}  "
          f"(success {bc_metrics['success_rate']:.1f}%)")

    print(f"\nSaved PPO learning curve to {ppo_out}")
    print(f"Saved BC timing to           {bc_out}")


if __name__ == "__main__":
    main()
