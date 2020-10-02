import os
import copy
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np


class SequenceReplayBuffer:
    """A numpy replay buffer with demonstrations."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        size: int,
        epi_len: int = 1536,
        batch_size: int = 32
    ):
        """Initialize."""
        self.obs_buf = np.zeros([size, epi_len, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros(
            [size, epi_len, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, epi_len, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, epi_len], dtype=np.float32)
        self.done_buf = np.zeros([size, epi_len], dtype=np.float32)
        self.epi_ptr = np.zeros([size], dtype=np.int32)
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        self.reset_epi_ptr = True

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Store the transition in buffer."""
        if self.reset_epi_ptr:
            self.epi_ptr[self.ptr] = 0
            self.reset_epi_ptr = False

        self.obs_buf[self.ptr, self.epi_ptr[self.ptr]] = obs
        self.next_obs_buf[self.ptr, self.epi_ptr[self.ptr]] = next_obs
        self.acts_buf[self.ptr, self.epi_ptr[self.ptr]] = act
        self.rews_buf[self.ptr, self.epi_ptr[self.ptr]] = rew
        self.done_buf[self.ptr, self.epi_ptr[self.ptr]] = done

        self.epi_ptr[self.ptr] += 1

        if done:
            self.reset_epi_ptr = True
            self.ptr += 1
            self.ptr = self.ptr % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, indices: List[int] = None) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size

        if indices is None:
            indices = np.random.choice(
                len(self), size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            epi_len=self.epi_ptr[indices]
        )

    def __len__(self) -> int:
        return self.size
