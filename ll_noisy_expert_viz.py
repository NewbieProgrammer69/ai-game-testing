"""Visualize the BC noisy expert experiment against the PPO expert baseline.

Shows how BC performance degrades as the fraction of corrupted actions in the
demonstration data increases. BC treats all labels as ground truth, so more
noise means more contradictory training signal and worse final policies.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

sns.set_theme(style="whitegrid")

# Consistent colors across all plots.
BC_COLOR = "steelblue"
PPO_COLOR = "coral"


def save_fig(name: str):
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    path = os.path.join(config.FIGURE_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def load_data():
    """Load noisy expert results and PPO baseline for reference lines."""
    noisy_df = pd.read_csv(os.path.join(config.LOG_DIR, "ll_noisy_expert.csv"))
    ppo_df = pd.read_csv(os.path.join(config.LOG_DIR, "ll_ppo_eval_results.csv"))
    ppo_mean_reward = ppo_df["reward"].mean()
    ppo_success_rate = ppo_df["success"].mean() * 100.0
    return noisy_df, ppo_mean_reward, ppo_success_rate


def _noise_percent(noisy_df: pd.DataFrame) -> np.ndarray:
    # Convert 0.0..0.75 -> 0..75 for display on the x-axis.
    return (noisy_df["noise_level"].values * 100.0)


def plot_reward(noisy_df, ppo_mean_reward):
    # Line plot of BC mean reward vs noise level, with std as shaded band.
    fig, ax = plt.subplots()
    x = _noise_percent(noisy_df)
    y = noisy_df["mean_reward"].values
    err = noisy_df["std_reward"].values

    ax.plot(x, y, marker="o", color=BC_COLOR, label="BC")
    ax.fill_between(x, y - err, y + err, alpha=0.2, color=BC_COLOR)
    ax.axhline(ppo_mean_reward, linestyle="--", color=PPO_COLOR, label="PPO (RL)")

    ax.set_xticks(x)
    ax.set_xlabel("Action Noise Level (%)")
    ax.set_ylabel("Mean Reward")
    ax.set_title("BC Performance vs. Expert Demonstration Quality")
    ax.legend()
    save_fig("ll_noisy_expert_reward.png")


def plot_success(noisy_df, ppo_success_rate):
    fig, ax = plt.subplots()
    x = _noise_percent(noisy_df)
    y = noisy_df["success_rate"].values

    ax.plot(x, y, marker="o", color=BC_COLOR, label="BC")
    ax.axhline(ppo_success_rate, linestyle="--", color=PPO_COLOR, label="PPO (RL)")

    ax.set_xticks(x)
    ax.set_xlabel("Action Noise Level (%)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("BC Success Rate vs. Expert Demonstration Quality")
    ax.legend()
    save_fig("ll_noisy_expert_success.png")


def plot_loss(noisy_df):
    fig, ax = plt.subplots()
    x = _noise_percent(noisy_df)
    ax.plot(x, noisy_df["final_train_loss"].values, marker="o", color=BC_COLOR, label="Train Loss")
    ax.plot(x, noisy_df["final_val_loss"].values, marker="s", color=PPO_COLOR, label="Val Loss")

    ax.set_xticks(x)
    ax.set_xlabel("Action Noise Level (%)")
    ax.set_ylabel("Final Loss")
    ax.set_title("BC Training Loss vs. Expert Demonstration Quality")
    ax.legend()
    save_fig("ll_noisy_expert_loss.png")


def plot_combined(noisy_df, ppo_mean_reward):
    # Two subplots side by side: reward (with PPO reference) and training loss.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = _noise_percent(noisy_df)

    # Left: Mean reward vs noise with PPO reference line.
    y = noisy_df["mean_reward"].values
    err = noisy_df["std_reward"].values
    axes[0].plot(x, y, marker="o", color=BC_COLOR, label="BC")
    axes[0].fill_between(x, y - err, y + err, alpha=0.2, color=BC_COLOR)
    axes[0].axhline(ppo_mean_reward, linestyle="--", color=PPO_COLOR, label="PPO (RL)")
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Action Noise Level (%)")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Reward vs. Expert Quality")
    axes[0].legend()

    # Right: Train/val loss vs noise.
    axes[1].plot(x, noisy_df["final_train_loss"].values, marker="o", color=BC_COLOR, label="Train Loss")
    axes[1].plot(x, noisy_df["final_val_loss"].values, marker="s", color=PPO_COLOR, label="Val Loss")
    axes[1].set_xticks(x)
    axes[1].set_xlabel("Action Noise Level (%)")
    axes[1].set_ylabel("Final Loss")
    axes[1].set_title("Training Loss vs. Expert Quality")
    axes[1].legend()

    fig.suptitle("Effect of Expert Quality on Behavior Cloning")
    save_fig("ll_noisy_expert_combined.png")


def main():
    noisy_df, ppo_mean_reward, ppo_success_rate = load_data()
    plot_reward(noisy_df, ppo_mean_reward)
    plot_success(noisy_df, ppo_success_rate)
    plot_loss(noisy_df)
    plot_combined(noisy_df, ppo_mean_reward)


if __name__ == "__main__":
    main()
