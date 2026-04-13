"""Train a Behavior Cloning model on expert demonstrations collected from the PPO LunarLander agent."""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import config
from ll_bc_model import LLBCNetwork
from utils import ensure_dirs


def main():
    # Make sure output folders exist before writing anything.
    ensure_dirs()

    # --- Load demonstration data ---
    demo_path = os.path.join(config.DEMO_DIR, "ll_demo_data.npz")
    data = np.load(demo_path)
    observations = torch.tensor(data["observations"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.long)
    print(f"Loaded {len(observations)} transitions from {demo_path}")

    # --- Train/validation split ---
    dataset = TensorDataset(observations, actions)
    n_train = int(len(dataset) * config.LL_BC_TRAIN_SPLIT)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_set, batch_size=config.LL_BC_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.LL_BC_BATCH_SIZE, shuffle=False)
    print(f"Train samples: {n_train} | Val samples: {n_val}")

    # --- Model, loss, optimizer ---
    model = LLBCNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LL_BC_LEARNING_RATE)

    # --- Training loop ---
    history = []
    for epoch in range(1, config.LL_BC_EPOCHS + 1):
        # Training pass.
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

        # Validation pass (no gradients).
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
        print(f"Epoch {epoch:3d}/{config.LL_BC_EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    # --- Save model ---
    model_path = os.path.join(config.MODEL_DIR, "bc_lunarlander.pth")
    torch.save(model.state_dict(), model_path)

    # --- Save training log ---
    log_path = os.path.join(config.LOG_DIR, "ll_bc_training_log.csv")
    pd.DataFrame(history).to_csv(log_path, index=False)

    print(f"\nBC LunarLander training complete.")
    print(f"Model saved to: {model_path}")
    print(f"Log   saved to: {log_path}")


if __name__ == "__main__":
    main()
