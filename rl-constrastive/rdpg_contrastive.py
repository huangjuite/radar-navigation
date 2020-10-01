
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
wandb.init(config=hyper_parameter,
           project="rl-mmWave-contrastive", name="bi-linear_"+t)
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

        self.critic = Critic(self.device, obs_dim, action_dim).to(self.device)
        self.critic_laser = Critic(
            self.device, obs_dim, action_dim).to(self.device)

        self.critic.load_state_dict(torch.load("critic_s1536_f1869509.pth"))
        self.critic_laser.load_state_dict(
            torch.load("critic_s1536_f1869509.pth"))
        self.critic_laser.eval()

        wandb.watch(self.actor)
        wandb.watch(self.critic)

        # optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=3e-4,
            weight_decay=lambda_l2,
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=1e-3,
            weight_decay=lambda_l2,
        )

        # contrastive loss
        latent_size = 512
        self.actor_project_W = torch.rand(
            latent_size, latent_size, device=self.device)
        self.actor_project_optimizer = optim.Adam(
            [self.actor_project_W], lr=1e-3)

        self.critic_project_W = torch.rand(
            latent_size, latent_size, device=self.device)
        self.critic_project_optimizer = optim.Adam(
            [self.critic_project_W], lr=1e-3)

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

                # train critic
                critic_loss, critic_contrastive_loss = self._get_critic_loss(
                    samples, self.gamma)

                self.critic_optimizer.zero_grad()
                self.critic_project_optimizer.zero_grad()

                critic_loss_total = critic_loss + critic_contrastive_loss
                critic_loss_total.backward()

                self.critic_optimizer.step()
                self.critic_project_optimizer.step()

                # train actor
                actions, _, _, policy_latent = self.actor(state)
                _, _, _, laser_policy_latent = self.actor_laser(laser)

                # actor loss
                value, _ = self.critic(state, actions)
                actor_loss_element_wise = -value
                actor_loss = torch.mean(actor_loss_element_wise)
                actor_contrastive_loss = self._get_contrastive_loss(
                    self.actor_project_W, laser_policy_latent, policy_latent)
                # actor_contrastive_loss = torch.mean(
                #     (policy_latent-laser_policy_latent).pow(2))

                self.actor_optimizer.zero_grad()
                self.actor_project_optimizer.zero_grad()

                actor_loss_total = actor_loss + actor_contrastive_loss
                actor_loss_total.backward()

                self.actor_optimizer.step()
                self.actor_project_optimizer.step()

                metrics = {
                    'critic_loss': critic_loss,
                    'critic_contrastive_loss': critic_contrastive_loss,
                    'actor_loss': actor_loss,
                    'actor_contrastive_loss': actor_contrastive_loss,
                }
                wandb.log(metrics)

            i+=1
            # save model
            torch.save(self.actor.state_dict(), os.path.join(
                wandb.run.dir, "model%d_%f.pth"%(i,actor_loss_total.item())))

    def _get_critic_loss(
        self, samples: Dict[str, np.ndarray], gamma: float
    ) -> torch.Tensor:
        """Return element-wise critic loss."""
        device = self.device

        state = torch.FloatTensor(samples["obs"]).to(device)
        laser = torch.FloatTensor(samples["laser"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        masks = 1 - done
        next_action, _, _, _ = self.actor(next_state)

        next_value, _ = self.critic(next_state, next_action)
        curr_return = reward + gamma * next_value * masks
        curr_return = curr_return.to(device).detach()

        # train critic
        values, value_latent = self.critic(state, action)
        _, laser_latent = self.critic_laser(laser, action)

        # value loss
        critic_loss_element_wise = (values - curr_return).pow(2)
        critic_loss = torch.mean(critic_loss_element_wise)

        # contrastive loss
        # contrastive_loss_element_wise = (value_latent - laser_latent).pow(2)

        contrastive_loss = self._get_contrastive_loss(
            self.critic_project_W, laser_latent, value_latent)

        return critic_loss, contrastive_loss


if __name__ == "__main__":
    agent = RDPG(
        gamma=config.gamma,
        lambda_l2=config.lambda_l2,       # l2 regularization weight
    )

    agent.update_model()
