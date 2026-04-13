"""Train a PPO agent on LunarLander-v3 and save the model to results/models/."""

import os
from stable_baselines3 import PPO

import config
from utils import make_ll_env, ensure_dirs


def main():
    # Make sure results/ and data/ folders exist before we write anything.
    ensure_dirs()

    # Create the LunarLander-v3 environment via the shared factory.
    env = make_ll_env()

    # Build the PPO agent with LunarLander hyperparameters from config.py.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.LL_PPO_LEARNING_RATE,
        n_steps=config.LL_PPO_N_STEPS,
        batch_size=config.LL_PPO_BATCH_SIZE,
        n_epochs=config.LL_PPO_N_EPOCHS,
        gamma=config.LL_PPO_GAMMA,
        clip_range=config.LL_PPO_CLIP_RANGE,
        verbose=1,
        tensorboard_log=config.LOG_DIR,
        device="auto",
    )

    # LunarLander is harder than CartPole, so we train for many more timesteps.
    model.learn(total_timesteps=config.LL_PPO_TOTAL_TIMESTEPS)

    # Save the trained model as a .zip file inside results/models/.
    model_path = os.path.join(config.MODEL_DIR, "ppo_lunarlander.zip")
    model.save(model_path)

    env.close()
    print(f"PPO LunarLander training complete. Model saved to {model_path}")


if __name__ == "__main__":
    main()
