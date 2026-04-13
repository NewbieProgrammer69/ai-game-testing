"""Train a PPO agent on CartPole-v1 and save the model to results/models/."""

import os
from stable_baselines3 import PPO

import config
from utils import make_env, ensure_dirs


def main():
    # Make sure results/ and data/ folders exist before we write anything.
    ensure_dirs()

    # Create the CartPole-v1 environment via the shared factory.
    env = make_env()

    # Build the PPO agent with hyperparameters from config.py.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.PPO_LEARNING_RATE,
        n_steps=config.PPO_N_STEPS,
        batch_size=config.PPO_BATCH_SIZE,
        n_epochs=config.PPO_N_EPOCHS,
        gamma=config.PPO_GAMMA,
        clip_range=config.PPO_CLIP_RANGE,
        verbose=1,
        tensorboard_log=config.LOG_DIR,
    )

    # Run the training loop for the configured number of timesteps.
    model.learn(total_timesteps=config.PPO_TOTAL_TIMESTEPS)

    # Save the trained model as a .zip file inside results/models/.
    model_path = os.path.join(config.MODEL_DIR, "ppo_cartpole.zip")
    model.save(model_path)

    env.close()
    print(f"PPO training complete. Model saved to {model_path}")


if __name__ == "__main__":
    main()
