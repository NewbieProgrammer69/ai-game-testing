"""Central configuration file. All hyperparameters and paths live here as plain variables."""

# === Environment ===
ENV_NAME = "CartPole-v1"

# === PPO Hyperparameters ===
PPO_TOTAL_TIMESTEPS = 50_000
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_CLIP_RANGE = 0.2

# === Behavior Cloning Hyperparameters ===
BC_LEARNING_RATE = 1e-3
BC_BATCH_SIZE = 64
BC_EPOCHS = 50
BC_HIDDEN_SIZE = 64
BC_TRAIN_SPLIT = 0.8

# === Evaluation ===
N_EVAL_EPISODES = 100
SUCCESS_THRESHOLD = 195.0

# === Demo Collection ===
N_DEMO_EPISODES = 50

# === LunarLander Environment ===
LL_ENV_NAME = "LunarLander-v3"

# === LunarLander PPO Hyperparameters ===
LL_PPO_TOTAL_TIMESTEPS = 3_000_000
LL_PPO_LEARNING_RATE = 3e-4
LL_PPO_N_STEPS = 2048
LL_PPO_BATCH_SIZE = 64
LL_PPO_N_EPOCHS = 10
LL_PPO_GAMMA = 0.99
LL_PPO_CLIP_RANGE = 0.2

# === LunarLander BC Hyperparameters ===
LL_BC_LEARNING_RATE = 1e-3
LL_BC_BATCH_SIZE = 64
LL_BC_EPOCHS = 100
LL_BC_HIDDEN_SIZE = 128
LL_BC_TRAIN_SPLIT = 0.8

# === LunarLander Evaluation ===
LL_N_EVAL_EPISODES = 100
LL_SUCCESS_THRESHOLD = 200.0

# === LunarLander Demo Collection ===
LL_N_DEMO_EPISODES = 200

# === Paths ===
MODEL_DIR = "results/models"
LOG_DIR = "results/logs"
FIGURE_DIR = "results/figures"
DEMO_DIR = "data/demos"
