"""Three-way CartPole comparison: PPO, BC (full data), BC (1% data).

Reads the three eval CSVs, prints a side-by-side summary, and regenerates the four CartPole
figures (reward_comparison, reward_distribution, episode_length_comparison, success_rate_comparison)
with three bars / boxes instead of two so the data-efficiency story is visible.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

sns.set_theme(style="whitegrid")

PPO_COLOR = "steelblue"
BC_FULL_COLOR = "coral"
BC_REDUCED_COLOR = "indianred"

LABELS = ["PPO", "BC (full data)", "BC (1% data)"]
COLORS = [PPO_COLOR, BC_FULL_COLOR, BC_REDUCED_COLOR]


def load_data():
    ppo = pd.read_csv(os.path.join(config.LOG_DIR, "ppo_eval_results.csv"))
    bc_full = pd.read_csv(os.path.join(config.LOG_DIR, "bc_eval_results.csv"))
    bc_red = pd.read_csv(os.path.join(config.LOG_DIR, "cartpole_reduced_bc_eval.csv"))
    return ppo, bc_full, bc_red


def stats(df):
    return {
        "mean_reward": df["reward"].mean(),
        "std_reward": df["reward"].std(),
        "mean_length": df["length"].mean(),
        "success_rate": df["success"].mean() * 100.0,
    }


def print_table(ppo, bc_full, bc_red):
    line = "=" * 60
    print(line)
    print(f"|  {'Metric':<19} |   PPO   | BC full | BC 1%   |")
    print(line)
    print(f"|  {'Mean Reward':<19} | {ppo['mean_reward']:>7.1f} | {bc_full['mean_reward']:>7.1f} | {bc_red['mean_reward']:>7.1f} |")
    print(f"|  {'Std Reward':<19} | {ppo['std_reward']:>7.1f} | {bc_full['std_reward']:>7.1f} | {bc_red['std_reward']:>7.1f} |")
    print(f"|  {'Mean Episode Length':<19} | {ppo['mean_length']:>7.1f} | {bc_full['mean_length']:>7.1f} | {bc_red['mean_length']:>7.1f} |")
    print(f"|  {'Success Rate (%)':<19} | {ppo['success_rate']:>7.1f} | {bc_full['success_rate']:>7.1f} | {bc_red['success_rate']:>7.1f} |")
    print(line)


def save_fig(name):
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    path = os.path.join(config.FIGURE_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_reward(ppo_df, bc_full_df, bc_red_df):
    means = [ppo_df["reward"].mean(), bc_full_df["reward"].mean(), bc_red_df["reward"].mean()]
    stds = [ppo_df["reward"].std(), bc_full_df["reward"].std(), bc_red_df["reward"].std()]
    fig, ax = plt.subplots()
    ax.bar(LABELS, means, yerr=stds, capsize=8, color=COLORS)
    ax.axhline(195, linestyle="--", color="gray", label="Success Threshold")
    ax.set_title("Mean Reward Comparison: PPO vs BC (CartPole)")
    ax.set_ylabel("Mean Reward")
    ax.legend()
    save_fig("reward_comparison.png")


def plot_length(ppo_df, bc_full_df, bc_red_df):
    means = [ppo_df["length"].mean(), bc_full_df["length"].mean(), bc_red_df["length"].mean()]
    stds = [ppo_df["length"].std(), bc_full_df["length"].std(), bc_red_df["length"].std()]
    fig, ax = plt.subplots()
    ax.bar(LABELS, means, yerr=stds, capsize=8, color=COLORS)
    ax.set_title("Mean Episode Length: PPO vs BC (CartPole)")
    ax.set_ylabel("Steps")
    save_fig("episode_length_comparison.png")


def plot_distribution(ppo_df, bc_full_df, bc_red_df):
    data = [ppo_df["reward"].values, bc_full_df["reward"].values, bc_red_df["reward"].values]
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, labels=LABELS, patch_artist=True)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
    ax.axhline(195, linestyle="--", color="gray", label="Success Threshold")
    ax.set_title("Reward Distribution: PPO vs BC (CartPole)")
    ax.set_ylabel("Total Reward")
    ax.legend()
    save_fig("reward_distribution.png")


def plot_success(ppo_df, bc_full_df, bc_red_df):
    rates = [
        ppo_df["success"].mean() * 100.0,
        bc_full_df["success"].mean() * 100.0,
        bc_red_df["success"].mean() * 100.0,
    ]
    fig, ax = plt.subplots()
    ax.bar(LABELS, rates, color=COLORS)
    ax.set_title("Success Rate Comparison: PPO vs BC (CartPole)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    save_fig("success_rate_comparison.png")


def main():
    ppo_df, bc_full_df, bc_red_df = load_data()
    ppo_s = stats(ppo_df)
    bc_full_s = stats(bc_full_df)
    bc_red_s = stats(bc_red_df)

    print_table(ppo_s, bc_full_s, bc_red_s)

    out = pd.DataFrame({
        "metric": ["mean_reward", "std_reward", "mean_length", "success_rate"],
        "PPO": [ppo_s[k] for k in ["mean_reward", "std_reward", "mean_length", "success_rate"]],
        "BC_full": [bc_full_s[k] for k in ["mean_reward", "std_reward", "mean_length", "success_rate"]],
        "BC_1pct": [bc_red_s[k] for k in ["mean_reward", "std_reward", "mean_length", "success_rate"]],
    })
    out_path = os.path.join(config.LOG_DIR, "comparison_three_way.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved comparison to {out_path}")

    plot_reward(ppo_df, bc_full_df, bc_red_df)
    plot_length(ppo_df, bc_full_df, bc_red_df)
    plot_distribution(ppo_df, bc_full_df, bc_red_df)
    plot_success(ppo_df, bc_full_df, bc_red_df)


if __name__ == "__main__":
    main()
