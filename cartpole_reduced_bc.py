"""Train a Behavior Cloning agent on a reduced slice of CartPole expert demos and evaluate it.

Goal of this script:
    The full-demo BC agent and the PPO expert both saturate CartPole at 500 reward, which makes
    the head-to-head comparison uninformative. We tried 5 percent of the demo dataset first, but
    the policy still scored 490 of 500 with 99 percent success. Reducing the slice to 1 percent
    (roughly 250 transitions out of 25,000) drops the agent below 220 mean reward and exposes a
    clear data-efficiency gap for Chapter 5.

Outputs:
    results/models/bc_cartpole_reduced.pth         trained network weights
    results/logs/cartpole_reduced_bc_train_log.csv per-epoch train/val loss
    results/logs/cartpole_reduced_bc_eval.csv      per-episode evaluation metrics (100 episodes)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import config
from bc_model import BCNetwork
from utils import ensure_dirs, make_env

REDUCED_FRACTION = 0.01


def train_reduced_bc():
    ensure_dirs()

    demo_path = os.path.join(config.DEMO_DIR, "demo_data.npz")
    data = np.load(demo_path)
    obs_full = data["observations"]
    act_full = data["actions"]
    n_total = len(obs_full)
    n_keep = int(round(n_total * REDUCED_FRACTION))

    observations = torch.tensor(obs_full[:n_keep], dtype=torch.float32)
    actions = torch.tensor(act_full[:n_keep], dtype=torch.long)
    print(f"Loaded {n_total} transitions, keeping the first {n_keep} ({REDUCED_FRACTION*100:.0f}%).")

    dataset = TensorDataset(observations, actions)
    n_train = int(len(dataset) * config.BC_TRAIN_SPLIT)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_set, batch_size=config.BC_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BC_BATCH_SIZE, shuffle=False)
    print(f"Train samples: {n_train} | Val samples: {n_val}")

    model = BCNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.BC_LEARNING_RATE)

    history = []
    for epoch in range(1, config.BC_EPOCHS + 1):
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
        train_loss = train_loss_sum / train_count

        model.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_count += xb.size(0)
        val_loss = val_loss_sum / val_count

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:3d}/{config.BC_EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    model_path = os.path.join(config.MODEL_DIR, "bc_cartpole_reduced.pth")
    torch.save(model.state_dict(), model_path)
    log_path = os.path.join(config.LOG_DIR, "cartpole_reduced_bc_train_log.csv")
    pd.DataFrame(history).to_csv(log_path, index=False)

    print(f"\nReduced-BC training complete.")
    print(f"Model saved to: {model_path}")
    print(f"Log   saved to: {log_path}")
    return model


def evaluate_reduced_bc(model: BCNetwork):
    env = make_env()
    records = []
    for ep in range(1, config.N_EVAL_EPISODES + 1):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = model.predict(np.asarray(obs, dtype=np.float32))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        success = total_reward >= config.SUCCESS_THRESHOLD
        records.append({
            "episode": ep,
            "reward": total_reward,
            "length": steps,
            "success": bool(success),
        })
    env.close()

    df = pd.DataFrame(records)
    mean_reward = df["reward"].mean()
    std_reward = df["reward"].std()
    mean_length = df["length"].mean()
    success_rate = df["success"].mean() * 100.0

    print(f"\n=== Reduced-BC Evaluation Summary ({REDUCED_FRACTION*100:.0f}% data) ===")
    print(f"Episodes      : {len(df)}")
    print(f"Mean reward   : {mean_reward:.2f}")
    print(f"Std  reward   : {std_reward:.2f}")
    print(f"Mean length   : {mean_length:.2f}")
    print(f"Success rate  : {success_rate:.2f}%  (threshold = {config.SUCCESS_THRESHOLD})")

    csv_path = os.path.join(config.LOG_DIR, "cartpole_reduced_bc_eval.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-episode results to {csv_path}")
    return mean_reward, std_reward, mean_length, success_rate


if __name__ == "__main__":
    model = train_reduced_bc()
    evaluate_reduced_bc(model)
