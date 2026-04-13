"""Collect expert demonstrations from the trained PPO LunarLander agent for Behavior Cloning.

Only successful episodes (reward >= LL_SUCCESS_THRESHOLD) are kept, so the BC
student learns from high-quality trajectories.
"""

import os
import numpy as np
from stable_baselines3 import PPO

import config
from utils import make_ll_env


def main():
    # Load the trained PPO LunarLander agent to act as the "expert".
    model_path = os.path.join(config.MODEL_DIR, "ppo_lunarlander.zip")
    model = PPO.load(model_path)

    env = make_ll_env()

    # Accumulators for all successful episodes.
    all_obs = []
    all_actions = []
    kept_episodes = 0

    for ep in range(1, config.LL_N_DEMO_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        # Per-episode buffers — only flushed to the global lists on success.
        ep_obs = []
        ep_actions = []

        while not done:
            # Greedy action from the expert policy.
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # Record the (observation, action) pair BEFORE stepping.
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_actions.append(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Only keep the episode if it met the LunarLander success threshold.
        if total_reward >= config.LL_SUCCESS_THRESHOLD:
            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)
            kept_episodes += 1

    env.close()

    # Stack into numpy arrays (obs shape (N,8), actions shape (N,)).
    observations = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int64)

    # Save to .npz in the demo directory.
    os.makedirs(config.DEMO_DIR, exist_ok=True)
    demo_path = os.path.join(config.DEMO_DIR, "ll_demo_data.npz")
    np.savez(demo_path, observations=observations, actions=actions)

    print("\n=== LunarLander Demo Collection Summary ===")
    print(f"Episodes run       : {config.LL_N_DEMO_EPISODES}")
    print(f"Successful episodes: {kept_episodes}")
    print(f"Total transitions  : {len(observations)}")
    print(f"Observations shape : {observations.shape}")
    print(f"Actions shape      : {actions.shape}")
    print(f"Saved to           : {demo_path}")


if __name__ == "__main__":
    main()
