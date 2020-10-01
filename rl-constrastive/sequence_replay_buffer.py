import os
import copy
import random
import pandas
import pickle as pkl
import numpy as np

from tqdm import tqdm, trange
from collections import deque
from typing import Deque, Dict, List, Tuple


class SequenceReplayBuffer:
    """A numpy replay buffer with demonstrations."""

    def __init__(
        self,
        main_path: str,
        obs_dim: int = 243,
        act_dim: int = 2,
        size: int = 200,
        epi_len: int = 512,
    ):
        """Initialize."""
        self.obs_buf = np.zeros([size, epi_len, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros(
            [size, epi_len, obs_dim], dtype=np.float32)
        self.laser_buf = np.zeros([size, epi_len, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, epi_len, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, epi_len], dtype=np.float32)
        self.done_buf = np.zeros([size, epi_len], dtype=np.float32)
        self.epi_ptr = np.zeros([size], dtype=np.int32)
        self.max_size = size
        self.ptr = 0
        self.size = 0

        self.reset_epi_ptr = True

        # all files path
        paths = []
        dirs = os.listdir(main_path)
        dirs.sort()
        for d in dirs:
            dirs1 = os.listdir(main_path+'/'+d)
            dirs1.sort()
            dirs2 = os.listdir(main_path+'/'+d+'/'+dirs1[1])
            dirs2.sort()
            for d2 in dirs2:
                paths.append(main_path+'/'+d+'/'+dirs1[1]+'/'+d2)
                # print(paths[-1])
        print('%d episodes' % len(paths))

        # read files
        for p in tqdm(paths):
            f = open(p, "rb")
            demo = pkl.load(f, encoding="bytes")

            df = pandas.DataFrame(demo)
            df = df.rename(lambda n: n.decode('utf-8'), axis='columns')
            df.drop('next_laser_scan', axis=1, inplace=True)

            for i in range(len(df['mm_scan'])):
                df['mm_scan'][i] = np.append(
                    df['mm_scan'][i], df['pos_diff'][i])

            for i in range(len(df['next_mm_scan'])):
                df['next_mm_scan'][i] = np.append(
                    df['next_mm_scan'][i], df['next_pos_diff'][i])

            df.drop('pos_diff', axis=1, inplace=True)
            df.drop('next_pos_diff', axis=1, inplace=True)

            for i in range(len(df)):
                self.store(obs=df['mm_scan'][i],
                           act=df['action'][i],
                           rew=df['reward'][i],
                           next_obs=df['next_mm_scan'][i],
                           done=df['done'][i],
                           laser=df['laser_scan'][i],
                           )

            f.close()

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        laser: np.ndarray,
    ):
        """Store the transition in buffer."""
        if self.reset_epi_ptr:
            self.epi_ptr[self.ptr] = 0
            self.reset_epi_ptr = False

        # append to obs dimension
        laser = np.append(laser, np.zeros(2))

        self.obs_buf[self.ptr, self.epi_ptr[self.ptr]] = obs
        self.laser_buf[self.ptr, self.epi_ptr[self.ptr]] = laser
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

    def __getitem__(self, index):

        return dict(
            obs=self.obs_buf[index, :self.epi_ptr[index]],
            next_obs=self.next_obs_buf[index, :self.epi_ptr[index]],
            acts=self.acts_buf[index, :self.epi_ptr[index]],
            rews=self.rews_buf[index, :self.epi_ptr[index]],
            done=self.done_buf[index, :self.epi_ptr[index]],
            laser=self.laser_buf[index, :self.epi_ptr[index]],
        )

    def __len__(self) -> int:
        return self.size
