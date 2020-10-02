
import gym
import time
import datetime
import os
import re
import random
import copy
import pickle as pkl
import numpy as np
import datetime as dt
from typing import Deque, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ou_noise import OUNoise
from sequence_replay_buffer import SequenceReplayBuffer
from network_v2 import Actor, Critic


class RDPG(object):

    def __init__(
        self,
        env: gym.Env,
        memory_size: int = 100,
        epi_batch_size: int = 8,  # numbers of sampled episodes
        max_epi_step: int = 512,
        initial_random_epi: int = 50,
        ou_noise_theta: float = 1.0,
        ou_noise_sigma: float = 0.1,
        gamma: float = 0.99,
        tau: float = 5e-3,
        lambda_l2: float = 1e-4,  # l2 regularization weight
        is_test: bool = False
    ):
        """Initialize."""
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.env = env
        self.epi_batch_size = epi_batch_size
        self.gamma = gamma
        self.tau = tau

        # replay buffer
        self.memory = SequenceReplayBuffer(
            obs_dim,
            action_dim,
            memory_size,
            max_epi_step,
            epi_batch_size,
        )

        # noise
        self.noise = OUNoise(
            action_dim,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma,
        )

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks
        self.actor = Actor(self.device, obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(
            self.device, obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.device, obs_dim, action_dim).to(self.device)
        self.critic_target = Critic(
            self.device, obs_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # lstm hidden state
        self.hn = torch.zeros((1, 1, 128), device=self.device)
        self.cn = torch.zeros((1, 1, 128), device=self.device)

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

        # logger
        if not is_test:
            self.writer = SummaryWriter()
            print(self.writer.get_logdir())
            self.actor_path = './%s/actor/' % self.writer.get_logdir()
            self.critic_path = './%s/critic/' % self.writer.get_logdir()
            os.makedirs(self.actor_path+"target")
            os.makedirs(self.actor_path+"eval")
            os.makedirs(self.critic_path+"target")
            os.makedirs(self.critic_path+"eval")

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0
        self.total_epi = 0
        self.initial_random_epi = initial_random_epi

        # mode: train / test
        self.is_test = is_test

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted

        if (len(self.memory) <= self.epi_batch_size or self.total_epi <= self.initial_random_epi) and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            state_t = torch.FloatTensor(state).to(self.device)
            state_t = torch.unsqueeze(state_t, dim=0)
            selected_action, self.hn, self.cn = self.actor(
                state_t, self.hn, self.cn)
            selected_action = selected_action.detach().cpu().numpy()[0]

        # add noise for exploration during training
        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(
                selected_action + noise, self.env.action_space.low, self.env.action_space.high)

        self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines

        sample_batch = self.memory.sample_batch()

        for i, epi_len in enumerate(sample_batch["epi_len"]):
            samples = dict(
                obs=sample_batch["obs"][i, :epi_len],
                next_obs=sample_batch["next_obs"][i, :epi_len],
                acts=sample_batch["acts"][i, :epi_len],
                rews=sample_batch["rews"][i, :epi_len],
                done=sample_batch["done"][i, :epi_len],
            )

            state = torch.FloatTensor(samples["obs"]).to(device)

            # train critic
            critic_loss_element_wise = self._get_critic_loss(
                samples, self.gamma)
            critic_loss = torch.mean(critic_loss_element_wise)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # train actor
            actions, _, _ = self.actor(state)
            actor_loss_element_wise = -self.critic(state, actions)
            actor_loss = torch.mean(actor_loss_element_wise)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target update
            self._target_soft_update()

        return actor_loss.data, critic_loss.data

    def _get_critic_loss(
        self, samples: Dict[str, np.ndarray], gamma: float
    ) -> torch.Tensor:
        """Return element-wise critic loss."""
        device = self.device  # for shortening the following lines

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        masks = 1 - done
        next_action, _, _ = self.actor_target(next_state)

        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + gamma * next_value * masks
        curr_return = curr_return.to(device).detach()

        # train critic
        values = self.critic(state, action)
        critic_loss_element_wise = (values - curr_return).pow(2)

        return critic_loss_element_wise

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def train(self, num_frames: int):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        scores = []
        score_track = np.zeros(100)
        score = 0

        for self.total_step in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                self.total_epi += 1

                # reset hidden state
                self.hn = torch.zeros((1, 1, 128), device=self.device)
                self.cn = torch.zeros((1, 1, 128), device=self.device)

                # log
                self.writer.add_scalar('reward', score, self.total_step)
                scores.append(score)

                # save
                if score > np.min(score_track):
                    score_track[np.argmin(score_track)] = score
                    self.save_model(score, self.total_step,
                                    max_to_keep=score_track.shape[0])
                score = 0

                # if training is ready
                if (len(self.memory) >= self.epi_batch_size) and (self.total_epi > self.initial_random_epi):
                    actor_loss, critic_loss = self.update_model()

                    # log
                    self.writer.add_scalar(
                        'actor_loss', actor_loss, self.total_step)
                    self.writer.add_scalar(
                        'critic_loss', critic_loss, self.total_step)

        ds = dt.datetime.now()
        t = str(ds).split(".")[0].replace(":", "-").replace(" ", "-")
        with open('scores_%s.pkl' % t, 'wb') as f:
            pkl.dump(scores, f)
        self.env.close()
        self.writer.close()

    def save_model(self, score, frame_idx, max_to_keep=10):
        file_name = 's%04d_f%d.pth' % (int(score), frame_idx)

        model_name = self.actor_path + 'eval/' + file_name
        target_name = self.actor_path + 'target/' + file_name

        critic_model_name = self.critic_path + 'eval/' + file_name
        critic_target_name = self.critic_path + 'target/' + file_name

        # actor
        torch.save(self.actor.state_dict(), model_name)
        torch.save(self.actor_target.state_dict(), target_name)

        # critic
        torch.save(self.critic.state_dict(), critic_model_name)
        torch.save(self.critic_target.state_dict(), critic_target_name)

        print("save model: %s" % model_name)

        dirs = [
            self.actor_path + 'eval/',
            self.actor_path + 'target/',
            self.critic_path + 'eval/',
            self.critic_path + 'target/',
        ]

        for files_dir in dirs:
            files = os.listdir(files_dir)
            if len(files) > max_to_keep:
                files.sort()
                os.remove(files_dir+'/'+files[0])

    def test(self, model_path):
        """Test the agent."""
        self.is_test = True

        self.actor.load_state_dict(torch.load(model_path))

        for _ in range(50):
            state = self.env.reset()

            # reset hidden state
            self.hn = torch.zeros((1, 1, 128), device=self.device)
            self.cn = torch.zeros((1, 1, 128), device=self.device)

            done = False
            score = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            print("score: ", score)
        self.env.close()
