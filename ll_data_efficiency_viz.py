"""Visualize the BC data efficiency experiment results against the PPO expert baseline."""

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
    eff_df = pd.read_csv(os.path.join(config.LOG_DIR, "ll_data_efficiency.csv"))
    ppo_df = pd.read_csv(os.path.join(config.LOG_DIR, "ll_ppo_eval_results.csv"))
    ppo_mean_reward = ppo_df["reward"].mean()
    ppo_std_reward = ppo_df["reward"].std()
    ppo_success_rate = ppo_df["success"].mean() * 100.0
    return eff_df, ppo_mean_reward, ppo_std_reward, ppo_success_rate


def plot_reward(eff_df, ppo_mean_reward):
    # Line plot of BC mean reward vs fraction, with std as shaded band.
    fig, ax = plt.subplots()
    x = eff_df["fraction"].values
    y = eff_df["mean_reward"].values
    err = eff_df["std_reward"].values

    ax.plot(x, y, marker="o", color=BC_COLOR, label="BC")
    ax.fill_between(x, y - err, y + err, alpha=0.2, color=BC_COLOR)
    ax.axhline(ppo_mean_reward, linestyle="--", color=PPO_COLOR, label="PPO (Expert)")

    ax.set_xlabel("Fraction of Demo Data")
    ax.set_ylabel("Mean Reward")
    ax.set_title("BC Performance vs. Amount of Demonstration Data")
    ax.legend()
    save_fig("ll_data_efficiency_reward.png")


def plot_success(eff_df, ppo_success_rate):
    fig, ax = plt.subplots()
    x = eff_df["fraction"].values
    y = eff_df["success_rate"].values

    ax.plot(x, y, marker="o", color=BC_COLOR, label="BC")
    ax.axhline(ppo_success_rate, linestyle="--", color=PPO_COLOR, label="PPO (Expert)")

    ax.set_xlabel("Fraction of Demo Data")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("BC Success Rate vs. Amount of Demonstration Data")
    ax.legend()
    save_fig("ll_data_efficiency_success.png")


def plot_loss(eff_df):
    fig, ax = plt.subplots()
    x = eff_df["fraction"].values
    ax.plot(x, eff_df["final_train_loss"].values, marker="o", color=BC_COLOR, label="Train Loss")
    ax.plot(x, eff_df["final_val_loss"].values, marker="s", color=PPO_COLOR, label="Val Loss")

    ax.set_xlabel("Fraction of Demo Data")
    ax.set_ylabel("Final Loss")
    ax.set_title("BC Final Loss vs. Amount of Demonstration Data")
    ax.legend()
    save_fig("ll_data_efficiency_loss.png")


def plot_vs_ppo_bar(eff_df, ppo_mean_reward, ppo_std_reward):
    # Grouped bar chart: each BC fraction + PPO, bars = mean reward, error bars = std.
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [f"BC {f:.2f}" for f in eff_df["fraction"].values] + ["PPO"]
    means = list(eff_df["mean_reward"].values) + [ppo_mean_reward]
    stds = list(eff_df["std_reward"].values) + [ppo_std_reward]
    colors = [BC_COLOR] * len(eff_df) + [PPO_COLOR]

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=stds, capsize=6, color=colors)
    ax.axhline(
        config.LL_SUCCESS_THRESHOLD,
        linestyle="--",
        color="gray",
        label=f"Success Threshold ({config.LL_SUCCESS_THRESHOLD:.0f})",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Reward")
    ax.set_title("PPO vs BC at Different Data Levels")
    ax.legend()
    save_fig("ll_data_vs_ppo_bar.png")


def main():
    eff_df, ppo_mean_reward, ppo_std_reward, ppo_success_rate = load_data()
    plot_reward(eff_df, ppo_mean_reward)
    plot_success(eff_df, ppo_success_rate)
    plot_loss(eff_df)
    plot_vs_ppo_bar(eff_df, ppo_mean_reward, ppo_std_reward)


if __name__ == "__main__":
    main()
