"""Data efficiency experiment: train BC on varying fractions of demo data and evaluate each model.

Answers: "How does the amount of demonstration data affect Behavior Cloning performance?"
This directly probes the covariate shift problem — less data means fewer states seen in
training, so more mistakes on unseen states at evaluation time.
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


# Fractions of the full demo dataset to test.
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


def train_bc_on_subset(observations: torch.Tensor, actions: torch.Tensor):
    """Train a fresh BC network on the given (observations, actions) tensors.

    Returns the trained model plus the final train and validation losses.
    """
    # Train/validation split (deterministic seed for reproducibility).
    dataset = TensorDataset(observations, actions)
    n_train = int(len(dataset) * config.LL_BC_TRAIN_SPLIT)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_set, batch_size=config.LL_BC_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.LL_BC_BATCH_SIZE, shuffle=False)

    # Fresh model for each fraction.
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
    """Run LL_N_EVAL_EPISODES episodes and return aggregate metrics."""
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

    # --- Load the full demo data once ---
    demo_path = os.path.join(config.DEMO_DIR, "ll_demo_data.npz")
    data = np.load(demo_path)
    all_obs = torch.tensor(data["observations"], dtype=torch.float32)
    all_actions = torch.tensor(data["actions"], dtype=torch.long)
    total_n = len(all_obs)
    print(f"Loaded {total_n} total demo transitions from {demo_path}\n")

    results = []
    for fraction in FRACTIONS:
        # Take the first N% (simple slicing for reproducibility).
        n = int(total_n * fraction)
        obs_sub = all_obs[:n]
        act_sub = all_actions[:n]

        # Train a fresh BC network on this subset.
        model, train_loss, val_loss = train_bc_on_subset(obs_sub, act_sub)

        # Save this model.
        model_path = os.path.join(config.MODEL_DIR, f"bc_ll_{fraction}.pth")
        torch.save(model.state_dict(), model_path)

        # Evaluate.
        metrics = evaluate_model(model)

        print(
            f"Fraction {fraction:.2f} | Samples: {n:5d} | "
            f"Mean Reward: {metrics['mean_reward']:7.2f} | "
            f"Success: {metrics['success_rate']:5.1f}% | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        results.append({
            "fraction": fraction,
            "num_samples": n,
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
            "mean_length": metrics["mean_length"],
            "success_rate": metrics["success_rate"],
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
        })

    # --- Save results CSV ---
    df = pd.DataFrame(results)
    out_path = os.path.join(config.LOG_DIR, "ll_data_efficiency.csv")
    df.to_csv(out_path, index=False)

    print("\n=== Data Efficiency Results ===")
    print(df.to_string(index=False))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
