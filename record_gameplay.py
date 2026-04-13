"""Record one episode per trained agent as a GIF for visual inspection.

Runs the PPO and BC agents on CartPole-v1 and LunarLander-v3, rendering each
episode to an RGB array stream and saving it under results/gameplay/.
"""

import os
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio

from stable_baselines3 import PPO

import config
from bc_model import BCNetwork
from ll_bc_model import LLBCNetwork


GAMEPLAY_DIR = os.path.join("results", "gameplay")
FPS = 30
N_EPISODES = 10


def record_episode(predict_fn, env, save_path: str, select: str = "best") -> None:
    """Run N_EPISODES episodes and save one as a GIF.

    select="best"    -> keep the episode with the highest reward.
    select="worst"   -> keep the episode with the lowest reward.
    select="average" -> keep the episode whose reward is closest to the mean.
    """
    assert select in ("best", "worst", "average")

    episodes = []
    for ep in range(N_EPISODES):
        frames = []
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            frames.append(env.render())
            done = terminated or truncated

        print(f"  episode {ep + 1}/{N_EPISODES}: {len(frames)} frames, {total_reward:.1f} reward")
        episodes.append((total_reward, frames))

    env.close()

    rewards = [r for r, _ in episodes]
    if select == "best":
        idx = int(np.argmax(rewards))
    elif select == "worst":
        idx = int(np.argmin(rewards))
    else:
        mean_r = float(np.mean(rewards))
        idx = int(np.argmin([abs(r - mean_r) for r in rewards]))

    keep_reward, keep_frames = episodes[idx]
    imageio.mimsave(save_path, keep_frames, fps=FPS)
    print(f"Saved {save_path} - {len(keep_frames)} frames, {keep_reward:.1f} reward ({select} of {N_EPISODES})")


def ppo_predict(model: PPO):
    """Deterministic PPO predict wrapper matching predict_fn signature."""
    def predict(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return predict


def bc_predict(model):
    """BC predict wrapper — both BCNetwork and LLBCNetwork expose .predict."""
    def predict(obs):
        return model.predict(np.asarray(obs, dtype=np.float32))
    return predict


def record_ppo(env_name: str, model_path: str, save_name: str, select: str = "best"):
    env = gym.make(env_name, render_mode="rgb_array")
    model = PPO.load(model_path)
    record_episode(ppo_predict(model), env, os.path.join(GAMEPLAY_DIR, save_name), select=select)


def record_bc(env_name: str, model_cls, weights_path: str, save_name: str, select: str = "best"):
    env = gym.make(env_name, render_mode="rgb_array")
    model = model_cls()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    record_episode(bc_predict(model), env, os.path.join(GAMEPLAY_DIR, save_name), select=select)


def main():
    os.makedirs(GAMEPLAY_DIR, exist_ok=True)

    # PPO on CartPole — best of 10.
    record_ppo(
        "CartPole-v1",
        os.path.join(config.MODEL_DIR, "ppo_cartpole.zip"),
        "ppo_cartpole.gif",
        select="best",
    )

    # BC on CartPole — best of 10.
    record_bc(
        "CartPole-v1",
        BCNetwork,
        os.path.join(config.MODEL_DIR, "bc_cartpole.pth"),
        "bc_cartpole.gif",
        select="best",
    )

    # PPO on LunarLander — keep the best of 10.
    record_ppo(
        "LunarLander-v3",
        os.path.join(config.MODEL_DIR, "ppo_lunarlander.zip"),
        "ppo_lunarlander.gif",
        select="best",
    )

    # BC on LunarLander (clean expert) — keep the best of 10.
    record_bc(
        "LunarLander-v3",
        LLBCNetwork,
        os.path.join(config.MODEL_DIR, "bc_lunarlander.pth"),
        "bc_lunarlander.gif",
        select="best",
    )

    # BC on LunarLander trained on 75%-noise demos — keep the episode closest
    # to the mean reward of 10, for a representative (not cherry-picked) sample.
    record_bc(
        "LunarLander-v3",
        LLBCNetwork,
        os.path.join(config.MODEL_DIR, "bc_ll_noise_0.75.pth"),
        "bc_lunarlander_noisy75.gif",
        select="average",
    )


if __name__ == "__main__":
    main()
