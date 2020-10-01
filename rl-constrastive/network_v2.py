import os
import copy
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(
        self,
        device,
        in_dim: int,
        out_dim: int,
    ):
        """Initialize."""
        super(Actor, self).__init__()
        self.device = device
        kernel = 3
        stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel, stride=stride),
            nn.ReLU()
        )
        dim = 64 * 59

        self.linear1 = nn.Linear(dim, 512)
        self.linear2 = nn.Linear(512+2, 256)  # + pos
        self.linear3 = nn.Linear(256, 128)

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=False)

        self.out = nn.Linear(128, out_dim)

    def forward(self, state: torch.Tensor,
                hn: torch.Tensor = None,
                cn: torch.Tensor = None) -> torch.Tensor:
        """Forward method implementation."""

        if hn is None:
            hn = torch.zeros((1, 1, 128), device=self.device)
            cn = torch.zeros((1, 1, 128), device=self.device)

        # split to laser, pos
        x, pos = torch.split(state, 241, dim=1)

        # expand to [batch, channel, features]
        x = x.reshape(x.shape[0], 1, -1)

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        latent = F.relu(self.linear1(x))
        x = torch.cat((latent, pos), dim=-1)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        # reshape to (sequence, batch=1, feature_dim)
        x = x.reshape(x.shape[0], 1, -1)
        x, (hn, cn) = self.lstm(x, (hn, cn))
        # reshape back (batch, feature_dim)
        x = x.reshape(x.shape[0], -1)

        x = self.out(x)

        # linear action : sigmoid
        # angular action : tanh
        linear_act = torch.sigmoid(x[:, 0])
        angular_act = torch.tanh(x[:, 1])
        action = torch.stack((linear_act, angular_act), dim=1)

        return action, hn, cn, latent


class Critic(nn.Module):
    def __init__(
        self,
        device,
        in_dim: int,
        out_dim: int,
    ):
        """Initialize."""
        super(Critic, self).__init__()
        self.device = device

        kernel = 3
        stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel, stride=stride),
            nn.ReLU()
        )
        dim = 64 * 59

        self.linear1 = nn.Linear(dim, 512)
        self.linear2 = nn.Linear(512+2+2, 256)  # + pos + action
        self.linear3 = nn.Linear(256, 128)

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=False)

        self.out = nn.Linear(128, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""

       # split to laser, pos
        x, pos = torch.split(state, 241, dim=1)

        # expand to [batch, channel, features]
        x = x.reshape(x.shape[0], 1, -1)

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        latent = F.relu(self.linear1(x))
        x = torch.cat((latent, pos), dim=-1)
        x = torch.cat((x, action), dim=-1)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        # reshape to (sequence, batch=1, feature_dim)
        x = x.reshape(x.shape[0], 1, -1)
        x, (hn, cn) = self.lstm(x)

        # reshape back (batch, feature_dim)
        x = x.reshape(x.shape[0], -1)

        value = self.out(x)

        return value, latent
