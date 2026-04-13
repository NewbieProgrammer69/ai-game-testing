"""Shared helpers: environment factory and directory setup."""

import os
import gymnasium as gym

import config


def make_env():
    # Create the Gymnasium environment defined in config.
    return gym.make(config.ENV_NAME)


def make_ll_env():
    """Create and return the LunarLander-v3 environment."""
    return gym.make(config.LL_ENV_NAME)


def ensure_dirs():
    # Create all output directories if they don't already exist.
    for path in (config.MODEL_DIR, config.LOG_DIR, config.FIGURE_DIR, config.DEMO_DIR):
        os.makedirs(path, exist_ok=True)
