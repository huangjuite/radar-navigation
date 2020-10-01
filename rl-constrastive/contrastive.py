
import gym
import time
import datetime
import os
import re
import random
import copy
import wandb
import pickle as pkl
import numpy as np
import datetime
from tqdm import tqdm, trange
from typing import Deque, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from network_v2 import Actor, Critic
from sequence_replay_buffer import SequenceReplayBuffer

seed = 888
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

hyper_parameter = dict(
    gamma=0.99,
    lambda_l2=1e-4,       # l2 regularization weight
    epoch=20,
)
ds = datetime.datetime.now()
t = str(ds).split(".")[0].replace(":", "-").replace(" ", "-")
# wandb.init(config=hyper_parameter,
#            project="mmWave-contrastive", name="bi-linear_"+t)
wandb.init(config=hyper_parameter,
           project="mmWave-contrastive", name="mse_"+t)

config = wandb.config
print(config)


class RDPG(object):

    def __init__(
        self,
        gamma: float = 0.99,
        lambda_l2: float = 1e-4,  # l2 regularization weight
    ):
        """Initialize."""
        obs_dim = 243
        action_dim = 2

        self.gamma = gamma

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks
        self.actor = Actor(self.device, obs_dim, action_dim).to(self.device)
        self.actor_laser = Actor(self.device, obs_dim,
                                 action_dim).to(self.device)

        self.actor.load_state_dict(torch.load("actor_s1536_f1869509.pth"))
        self.actor_laser.load_state_dict(
            torch.load("actor_s1536_f1869509.pth"))
        self.actor_laser.eval()

        wandb.watch(self.actor)

        # optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=3e-4,
            weight_decay=lambda_l2,
        )
        # contrastive loss
        latent_size = 512
        self.actor_project_W = torch.rand(
            latent_size, latent_size, device=self.device)
        self.actor_project_optimizer = optim.Adam(
            [self.actor_project_W], lr=1e-3)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # memory buffer
        self.memory = SequenceReplayBuffer(
            main_path='/media/ray/intelSSD/mmdemo_train',
            size=250,
            epi_len=512,
        )

    def _get_contrastive_loss(self, W, z_q, z_k):
        proj_k = torch.matmul(W, z_k.T)
        logits = torch.matmul(z_q, proj_k)
        max_logits, _ = torch.max(logits, dim=1)
        logits = logits - max_logits
        labels = torch.arange(logits.shape[0], device=self.device)
        loss = self.cross_entropy_loss(logits, labels)

        return loss

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by gradient descent."""
        device = self.device
        i = 0

        for e in trange(config.epoch):

            indices = np.random.choice(
                len(self.memory), size=len(self.memory), replace=False)

            for indx in indices:

                samples = self.memory[indx]

                state = torch.FloatTensor(samples["obs"]).to(device)
                laser = torch.FloatTensor(samples["laser"]).to(device)

                # train actor
                actions, _, _, policy_latent = self.actor(state)
                _, _, _, laser_policy_latent = self.actor_laser(laser)

                # bilinear
                # actor_contrastive_loss = self._get_contrastive_loss(
                #     self.actor_project_W, laser_policy_latent, policy_latent)
                
                # mse
                actor_contrastive_loss = torch.mean(
                    (policy_latent-laser_policy_latent).pow(2))

                self.actor_optimizer.zero_grad()
                self.actor_project_optimizer.zero_grad()

                actor_contrastive_loss.backward()

                self.actor_optimizer.step()
                self.actor_project_optimizer.step()

                metrics = {
                    'actor_contrastive_loss': actor_contrastive_loss,
                }
                wandb.log(metrics)

            i += 1
            # save model
            torch.save(self.actor.state_dict(), os.path.join(
                wandb.run.dir, "model%d_%f.pth" % (i, actor_contrastive_loss.item())))


if __name__ == "__main__":
    agent = RDPG(
        gamma=config.gamma,
        lambda_l2=config.lambda_l2,       # l2 regularization weight
    )

    agent.update_model()
