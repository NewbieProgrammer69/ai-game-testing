"""Behavior Cloning model: a simple 3-layer MLP that maps CartPole observations to actions."""

import numpy as np
import torch
import torch.nn as nn

import config


class BCNetwork(nn.Module):
    def __init__(self, input_size: int = 4, output_size: int = 2, hidden_size: int = config.BC_HIDDEN_SIZE):
        super().__init__()
        # Three-layer MLP with ReLU activations between hidden layers.
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits — softmax/argmax happens outside.
        return self.net(x)

    def predict(self, observation: np.ndarray) -> int:
        # Convert numpy obs -> tensor, forward pass, pick the argmax action.
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(x)
            action = int(torch.argmax(logits, dim=1).item())
        return action
