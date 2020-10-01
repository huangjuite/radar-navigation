#!/usr/bin/env python3
# coding: utf-8

import os
import io
import cv2
import copy
import math
import wandb
import random
import datetime
import numpy as np
import pickle as pkl
from collections import deque
from tqdm import tqdm, trange
from typing import Deque, Dict, List, Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

hyper_parameter = dict(
    kernel=3,
    stride=2,
    padding=2,
    latent=128,
    deconv_dim=32,
    deconv_channel=128,
    adjust_linear=235,
    epoch=100,
    learning_rate=0.001,
    batch_size=128,
    lambda_kld=0.5,
    decoder="deconv",
)
ds = datetime.datetime.now()
t = str(ds).split(".")[0].replace(":", "-").replace(" ", "-")
wandb.init(config=hyper_parameter, project="mm_wave-VAE", name=t)
config = wandb.config
print(config)

# data paths ================================================================================
paths = []
main_path = '/media/ray/intelSSD/mmdemo_train'
dirs = os.listdir(main_path)
dirs.sort()
for d in dirs:
    dirs1 = os.listdir(main_path+'/'+d)
    dirs1.sort()
    dirs2 = os.listdir(main_path+'/'+d+'/'+dirs1[1])
    dirs2.sort()
    for d2 in dirs2:
        paths.append(main_path+'/'+d+'/'+dirs1[1]+'/'+d2)
        print(paths[-1])
print('%d episodes' % len(paths))


# mm-wave , laser scan dataset  ==============================================================
class MMDataset(Dataset):
    def __init__(self, paths):
        self.transitions = []

        for p in tqdm(paths):
            with open(p, "rb") as f:
                demo = pkl.load(f, encoding="bytes")
                self.transitions.extend(demo)

    def __getitem__(self, index):
        mm_scan = self.transitions[index][b'mm_scan']
        laser_scan = self.transitions[index][b'laser_scan']
        mm_scan = torch.Tensor(mm_scan).reshape(1, -1)
        laser_scan = torch.Tensor(laser_scan).reshape(1, -1)
        0
        return mm_scan, laser_scan

    def __len__(self):
        return len(self.transitions)


mm_dataset = MMDataset(paths)
# train_sz = int(len(mm_dataset)*config.split_ratio)
# test_sz = len(mm_dataset)-train_sz
# train_set,test_set = random_split(mm_dataset,[train_sz,test_sz])

train_loader = DataLoader(dataset=mm_dataset,
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=4)

# test_loader = DataLoader(dataset=test_set,
#                           batch_size=config.batch_size,
#                           shuffle=False,
#                           num_workers=4)


# visualization function ================================================================
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def laser_visual(lasers=[], show=False, range_limit=6):
    fig = plt.figure(figsize=(8, 8))
    for l in reversed(lasers):
        angle = 120
        xp = []
        yp = []
        for r in l:
            if r <= range_limit:
                yp.append(r * math.cos(math.radians(angle)))
                xp.append(r * math.sin(math.radians(angle)))
            angle -= 1
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.plot(xp, yp, 'x')
    img = get_img_from_fig(fig)
    if not show:
        plt.close()
    return img


# model ================================================================================
class MMvae(nn.Module):
    def __init__(self):
        super(MMvae, self).__init__()
        kernel = 3
        stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel, stride=stride),
            nn.ReLU()
        )

        dim = 64*59
        self.linear1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU()
        )
        self.en_fc1 = nn.Linear(512, config.latent)
        self.en_fc2 = nn.Linear(512, config.latent)

        self.de_fc1 = nn.Sequential(
            nn.Linear(config.latent, config.deconv_channel*config.deconv_dim),
            nn.ReLU()
        )

        self.de_conv = nn.Sequential(
            nn.ConvTranspose1d(config.deconv_channel, config.deconv_channel //
                               2, kernel, stride=stride, padding=config.padding),
            # nn.ReLU(),
            nn.ConvTranspose1d(config.deconv_channel//2, config.deconv_channel //
                               4, kernel, stride=stride, padding=config.padding),
            # nn.ReLU(),
            nn.ConvTranspose1d(config.deconv_channel//4, 1,
                               kernel, stride=stride, padding=config.padding),
            # nn.ReLU(),
        )
        self.adjust_linear = nn.Sequential(
            nn.Linear(config.adjust_linear, 241),
            nn.ReLU()
        )

    def encoder(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        mean = self.en_fc1(x)
        logvar = self.en_fc2(x)
        return mean, logvar

    def reparameter(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def decoder(self, x):
        x = self.de_fc1(x)
        x = x.view(-1, config.deconv_channel, config.deconv_dim)
        x = self.de_conv(x)
        x = self.adjust_linear(x)
        return x

    def forward(self, x):
        mean, logvar = self.encoder(x)
        x = self.reparameter(mean, logvar)
        x = self.decoder(x)
        return x, mean, logvar


# variational auto encoder loss =======================================================

recon_loss = nn.MSELoss()

def loss_function(x, x_hat, mean, logvar):
    # recon = F.binary_cross_entropy(x_hat, x, reduction='sum')
    recon = recon_loss(x_hat, x)

    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return recon + config.lambda_kld*KLD, recon, KLD


# train ================================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device, ', device)
model = MMvae()
model.to(device)
wandb.watch(model)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

iter = 0
for e in range(config.epoch):
    for mm_scan, laser_scan in train_loader:
        mm_scan = mm_scan.to(device)
        laser_scan = laser_scan.to(device)

        optimizer.zero_grad()
        x_hat, mean, logvar = model(mm_scan)
        elbo, recon, kld = loss_function(laser_scan, x_hat, mean, logvar)
        elbo.backward()
        optimizer.step()

        iter += 1
        if iter % 100 == 0:
            metrics = {'total loss': elbo,
                       'reconstruction loss': recon, 'KL-divergence': kld}
            wandb.log(metrics)
            print("iteration %d, loss %.4f" % (iter, elbo.item()))

# visualize and save =====================================================================
model.eval()
vis_number = 12

for mm_scan, laser_scan in train_loader:
    mm_scan = mm_scan.to(device)

    x_hat, mean, logvar = model(mm_scan)

    x_hat = x_hat.detach().cpu().numpy().reshape(config.batch_size, -1)[:vis_number]
    laser_scan = laser_scan.numpy().reshape(config.batch_size, -1)[:vis_number]
    mm_scan = mm_scan.detach().cpu().numpy().reshape(config.batch_size, -1)[:vis_number]

    for i, (x, mm, laser) in enumerate(zip(x_hat, mm_scan, laser_scan)):
        data = laser_visual([mm, x, laser], show=False)
        wandb.log({"example%d"%i: wandb.Image(data)})

    break

torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pth"))
