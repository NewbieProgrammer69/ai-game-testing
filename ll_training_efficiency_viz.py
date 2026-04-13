"""Visualize the PPO vs BC training efficiency experiment.

Plots the PPO learning curve against the (much cheaper) BC training run and
shows how the two methods trade compute for performance. Four figures:

1. ll_ppo_learning_curve.png  - PPO reward vs timesteps, BC final as hline,
                                vertical marker where PPO first passes BC.
2. ll_time_comparison.png     - bar chart of total training time (PPO vs BC).
3. ll_efficiency_tradeoff.png - scatter of final reward vs training time.
4. ll_reward_over_time.png    - PPO reward vs wall-clock time, BC marker.
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
    """Load PPO learning curve + BC single-row timing result."""
    ppo_df = pd.read_csv(os.path.join(config.LOG_DIR, "ll_ppo_learning_curve.csv"))
    bc_df = pd.read_csv(os.path.join(config.LOG_DIR, "ll_bc_timing.csv"))
    bc_row = bc_df.iloc[0]
    return ppo_df, bc_row


def find_crossover(ppo_df: pd.DataFrame, bc_reward: float):
    """Return (timesteps, wall_time) of first PPO checkpoint >= BC final reward.

    Returns (None, None) if PPO never reaches BC within the training budget.
    """
    above = ppo_df[ppo_df["mean_reward"] >= bc_reward]
    if above.empty:
        return None, None
    row = above.iloc[0]
    return int(row["timesteps"]), float(row["wall_time_seconds"])


def plot_learning_curve(ppo_df, bc_row):
    """PPO mean reward vs timesteps with shaded std, BC reference, and crossover marker."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = ppo_df["timesteps"].values
    y = ppo_df["mean_reward"].values
    err = ppo_df["std_reward"].values

    # PPO curve with std band.
    ax.plot(x, y, color=PPO_COLOR, label="PPO (RL)")
    ax.fill_between(x, y - err, y + err, alpha=0.2, color=PPO_COLOR)

    # BC reference as horizontal line.
    bc_reward = float(bc_row["mean_reward"])
    ax.axhline(bc_reward, linestyle="--", color=BC_COLOR,
               label=f"BC final ({bc_reward:.1f})")

    # Vertical marker where PPO first crosses BC.
    cross_steps, cross_time = find_crossover(ppo_df, bc_reward)
    if cross_steps is not None:
        ax.axvline(cross_steps, linestyle=":", color="gray", alpha=0.7)
        ax.annotate(
            f"PPO > BC\n@ {cross_steps:,} steps\n({cross_time:.0f}s)",
            xy=(cross_steps, bc_reward),
            xytext=(cross_steps + 0.05 * x.max(), bc_reward - 120),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean Reward (20 eval episodes)")
    ax.set_title("PPO Learning Curve vs. BC Final Performance")
    ax.legend(loc="lower right")
    save_fig("ll_ppo_learning_curve.png")


def plot_time_comparison(ppo_df, bc_row):
    """Bar chart: total training time for PPO vs BC (seconds)."""
    fig, ax = plt.subplots(figsize=(6, 5))

    ppo_time = float(ppo_df["wall_time_seconds"].iloc[-1])
    bc_time = float(bc_row["total_train_time_seconds"])

    methods = ["BC", "PPO"]
    times = [bc_time, ppo_time]
    colors = [BC_COLOR, PPO_COLOR]

    bars = ax.bar(methods, times, color=colors)

    # Annotate each bar with its time and the PPO/BC ratio on PPO.
    ratio = ppo_time / bc_time if bc_time > 0 else float("inf")
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.1f}s",
            ha="center", va="bottom", fontsize=11,
        )
    # Ratio label on top of PPO bar.
    ax.text(
        bars[1].get_x() + bars[1].get_width() / 2,
        bars[1].get_height() * 0.5,
        f"{ratio:.1f}x BC",
        ha="center", va="center", fontsize=13, color="white", fontweight="bold",
    )

    ax.set_ylabel("Total Training Time (seconds)")
    ax.set_title("Training Time: PPO vs BC")
    # Give the annotations some headroom.
    ax.set_ylim(0, max(times) * 1.15)
    save_fig("ll_time_comparison.png")


def plot_efficiency_tradeoff(ppo_df, bc_row):
    """Scatter: final reward vs training time for both methods."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ppo_time = float(ppo_df["wall_time_seconds"].iloc[-1])
    ppo_reward = float(ppo_df["mean_reward"].iloc[-1])
    bc_time = float(bc_row["total_train_time_seconds"])
    bc_reward = float(bc_row["mean_reward"])

    ax.scatter([bc_time], [bc_reward], s=250, color=BC_COLOR, label="BC", zorder=3)
    ax.scatter([ppo_time], [ppo_reward], s=250, color=PPO_COLOR, label="PPO", zorder=3)

    # Label each point with its coordinates.
    ax.annotate(
        f"BC\n{bc_time:.0f}s, reward {bc_reward:.0f}",
        xy=(bc_time, bc_reward),
        xytext=(10, 10), textcoords="offset points", fontsize=10,
    )
    ax.annotate(
        f"PPO\n{ppo_time:.0f}s, reward {ppo_reward:.0f}",
        xy=(ppo_time, ppo_reward),
        xytext=(-80, 10), textcoords="offset points", fontsize=10,
    )

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Final Mean Reward")
    ax.set_title("Efficiency Tradeoff: Performance vs. Training Cost")
    ax.legend(loc="lower right")
    save_fig("ll_efficiency_tradeoff.png")


def plot_reward_over_time(ppo_df, bc_row):
    """PPO reward vs wall-clock time, with BC finish as a single marker."""
    fig, ax = plt.subplots(figsize=(9, 5))

    t = ppo_df["wall_time_seconds"].values
    y = ppo_df["mean_reward"].values
    err = ppo_df["std_reward"].values

    ax.plot(t, y, color=PPO_COLOR, label="PPO (RL)")
    ax.fill_between(t, y - err, y + err, alpha=0.2, color=PPO_COLOR)

    bc_time = float(bc_row["total_train_time_seconds"])
    bc_reward = float(bc_row["mean_reward"])
    # BC shows up as a single point: its full training takes bc_time seconds
    # and produces bc_reward final reward.
    ax.scatter([bc_time], [bc_reward], s=200, color=BC_COLOR, zorder=3,
               label=f"BC finished ({bc_time:.0f}s, {bc_reward:.0f})")
    ax.axhline(bc_reward, linestyle="--", color=BC_COLOR, alpha=0.5)

    ax.set_xlabel("Wall-Clock Training Time (seconds)")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward Over Wall-Clock Training Time")
    ax.legend(loc="lower right")
    save_fig("ll_reward_over_time.png")


def main():
    ppo_df, bc_row = load_data()
    plot_learning_curve(ppo_df, bc_row)
    plot_time_comparison(ppo_df, bc_row)
    plot_efficiency_tradeoff(ppo_df, bc_row)
    plot_reward_over_time(ppo_df, bc_row)


if __name__ == "__main__":
    main()
