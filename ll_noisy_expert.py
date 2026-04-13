"""Noisy expert experiment: train BC on demo data with varying levels of action corruption.

Answers: "How does expert demonstration quality affect Behavior Cloning performance?"
We take the clean demo data and randomly replace a fraction of the actions with random ones,
simulating experts of decreasing quality. BC has no way to tell good actions from bad ones,
so its performance should degrade as the noise level increases.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import config
from ll_bc_model import LLBCNetwork
from utils import make_ll_env, ensure_dirs


# Fractions of actions to replace with random actions.
NOISE_LEVELS = [0.0, 0.10, 0.25, 0.50, 0.75]

# LunarLander has 4 discrete actions (0..3).
NUM_ACTIONS = 4


def corrupt_actions(clean_actions: np.ndarray, noise_level: float) -> np.ndarray:
    """Return a noisy copy of `clean_actions`.

    For each action, with probability `noise_level`, replace it with a random
    action in [0, NUM_ACTIONS). Uses a fixed seed for reproducibility so the
    same noise level always produces the same corrupted dataset.
    """
    # Fresh seed per noise level so runs are reproducible.
    np.random.seed(42)

    noisy = clean_actions.copy()
    # Boolean mask of which transitions to corrupt.
    mask = np.random.rand(len(noisy)) < noise_level
    # Random replacement actions for the corrupted positions.
    random_actions = np.random.randint(0, NUM_ACTIONS, size=mask.sum())
    noisy[mask] = random_actions
    return noisy


def train_bc(observations: torch.Tensor, actions: torch.Tensor):
    """Train a fresh BC network on the given tensors. Returns (model, train_loss, val_loss)."""
    # Train/val split with fixed seed.
    dataset = TensorDataset(observations, actions)
    n_train = int(len(dataset) * config.LL_BC_TRAIN_SPLIT)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_set, batch_size=config.LL_BC_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.LL_BC_BATCH_SIZE, shuffle=False)

    # Fresh model each call.
    model = LLBCNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LL_BC_LEARNING_RATE)

    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(1, config.LL_BC_EPOCHS + 1):
        # --- Training pass ---
        model.train()
        train_loss_sum, train_count = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_count += xb.size(0)
        final_train_loss = train_loss_sum / train_count

        # --- Validation pass ---
        model.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_count += xb.size(0)
        final_val_loss = val_loss_sum / val_count

    return model, final_train_loss, final_val_loss


def evaluate_model(model: LLBCNetwork):
    """Run LL_N_EVAL_EPISODES episodes on LunarLander and return aggregate metrics."""
    env = make_ll_env()
    rewards, lengths, successes = [], [], []

    for _ in range(config.LL_N_EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = model.predict(np.asarray(obs, dtype=np.float32))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(total_reward >= config.LL_SUCCESS_THRESHOLD)

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes) * 100.0),
    }


def main():
    ensure_dirs()

    # --- Load clean demo data once ---
    demo_path = os.path.join(config.DEMO_DIR, "ll_demo_data.npz")
    data = np.load(demo_path)
    clean_obs_np = data["observations"]
    clean_act_np = data["actions"]
    print(f"Loaded {len(clean_obs_np)} clean demo transitions from {demo_path}\n")

    # Observations never change — only actions get corrupted.
    obs_tensor = torch.tensor(clean_obs_np, dtype=torch.float32)

    results = []
    for noise_level in NOISE_LEVELS:
        # Corrupt a copy of the actions according to the noise level.
        noisy_actions_np = corrupt_actions(clean_act_np, noise_level)
        actions_tensor = torch.tensor(noisy_actions_np, dtype=torch.long)

        # Train fresh BC on the corrupted data.
        model, train_loss, val_loss = train_bc(obs_tensor, actions_tensor)

        # Save this noisy-expert model so other scripts (e.g. gameplay recording)
        # can reload it without retraining.
        model_path = os.path.join(config.MODEL_DIR, f"bc_ll_noise_{noise_level}.pth")
        torch.save(model.state_dict(), model_path)

        # Evaluate on the real environment.
        metrics = evaluate_model(model)

        print(
            f"Noise {noise_level:.2f} | "
            f"Mean Reward: {metrics['mean_reward']:7.2f} | "
            f"Success: {metrics['success_rate']:5.1f}% | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        results.append({
            "noise_level": noise_level,
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
            "mean_length": metrics["mean_length"],
            "success_rate": metrics["success_rate"],
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
        })

    # --- Save results CSV ---
    df = pd.DataFrame(results)
    out_path = os.path.join(config.LOG_DIR, "ll_noisy_expert.csv")
    df.to_csv(out_path, index=False)

    print("\n=== Noisy Expert Results ===")
    print(df.to_string(index=False))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
