"""Generate comparison plots for PPO vs BC on LunarLander-v3 and BC training curves."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

sns.set_theme(style="whitegrid")

# Consistent colors across all plots.
PPO_COLOR = "steelblue"
BC_COLOR = "coral"


def load_data():
    ppo_df = pd.read_csv(os.path.join(config.LOG_DIR, "ll_ppo_eval_results.csv"))
    bc_df = pd.read_csv(os.path.join(config.LOG_DIR, "ll_bc_eval_results.csv"))
    bc_log = pd.read_csv(os.path.join(config.LOG_DIR, "ll_bc_training_log.csv"))
    return ppo_df, bc_df, bc_log


def save_fig(name: str):
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    path = os.path.join(config.FIGURE_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_reward_comparison(ppo_df, bc_df):
    # Mean reward bar chart with std error bars.
    labels = ["PPO", "BC"]
    means = [ppo_df["reward"].mean(), bc_df["reward"].mean()]
    stds = [ppo_df["reward"].std(), bc_df["reward"].std()]

    fig, ax = plt.subplots()
    ax.bar(labels, means, yerr=stds, capsize=8, color=[PPO_COLOR, BC_COLOR])
    ax.axhline(200, linestyle="--", color="gray", label="Success Threshold")
    ax.set_title("LunarLander Mean Reward: PPO vs BC")
    ax.set_ylabel("Mean Reward")
    ax.legend()
    save_fig("ll_reward_comparison.png")


def plot_episode_length_comparison(ppo_df, bc_df):
    # Violin plot of per-episode length for PPO and BC. A violin shows median,
    # quartiles, and the full shape of the distribution in a compact form, which
    # is more informative than two big bars and also less visually heavy.
    long = pd.DataFrame({
        "length": pd.concat([ppo_df["length"], bc_df["length"]], ignore_index=True),
        "agent": (["PPO"] * len(ppo_df)) + (["BC"] * len(bc_df)),
    })
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=long, x="agent", y="length", hue="agent",
        palette={"PPO": PPO_COLOR, "BC": BC_COLOR},
        inner="quartile", cut=0, linewidth=1.4,
        ax=ax, legend=False,
    )
    ax.set_title("LunarLander Episode Length Distribution: PPO vs BC")
    ax.set_xlabel("")
    ax.set_ylabel("Episode length (steps)")
    save_fig("ll_episode_length_comparison.png")


def plot_reward_distribution(ppo_df, bc_df):
    # Side-by-side box plots of per-episode reward.
    data = [ppo_df["reward"].values, bc_df["reward"].values]
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, tick_labels=["PPO", "BC"], patch_artist=True)
    for patch, color in zip(bp["boxes"], [PPO_COLOR, BC_COLOR]):
        patch.set_facecolor(color)
    ax.axhline(200, linestyle="--", color="gray", label="Success Threshold")
    ax.set_title("LunarLander Reward Distribution: PPO vs BC")
    ax.set_ylabel("Total Reward")
    ax.legend()
    save_fig("ll_reward_distribution.png")


def plot_bc_training_loss(bc_log):
    # Train vs validation loss curves over epochs.
    fig, ax = plt.subplots()
    ax.plot(bc_log["epoch"], bc_log["train_loss"], label="Train Loss", color=PPO_COLOR)
    ax.plot(bc_log["epoch"], bc_log["val_loss"], label="Val Loss", color=BC_COLOR)
    ax.set_title("LunarLander BC Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    save_fig("ll_bc_training_loss.png")


def plot_success_rate_comparison(ppo_df, bc_df):
    labels = ["PPO", "BC"]
    rates = [ppo_df["success"].mean() * 100.0, bc_df["success"].mean() * 100.0]

    fig, ax = plt.subplots()
    ax.bar(labels, rates, color=[PPO_COLOR, BC_COLOR])
    ax.set_title("LunarLander Success Rate: PPO vs BC")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    save_fig("ll_success_rate_comparison.png")


def main():
    ppo_df, bc_df, bc_log = load_data()
    plot_reward_comparison(ppo_df, bc_df)
    plot_episode_length_comparison(ppo_df, bc_df)
    plot_reward_distribution(ppo_df, bc_df)
    plot_bc_training_loss(bc_log)
    plot_success_rate_comparison(ppo_df, bc_df)


if __name__ == "__main__":
    main()
