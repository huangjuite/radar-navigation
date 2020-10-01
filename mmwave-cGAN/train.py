#!/usr/bin/env python
# coding: utf-8


import os
import io
import cv2
import copy
import math
import wandb
import random
import numpy as np
import pickle as pkl
import datetime
from collections import deque
from tqdm import tqdm, trange
from typing import Deque, Dict, List, Tuple
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

seed = 888
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ## hyper parameters

hyper_parameter = dict(
    kernel=3,
    stride=2,
    padding=2,
    deconv_dim=32,
    deconv_channel=128,
    adjust_linear=235,
    epoch=100,
    beta1=0.5,
    learning_rate=0.0002,
    nz=100,
    lambda_l1=1,
    batch_size=128,
    vis_num=4,
    visualize_epoch=10,
)
ds = datetime.datetime.now()
t = str(ds).split(".")[0].replace(":", "-").replace(" ", "-")
wandb.init(config=hyper_parameter,
           project="cGAN-mmwave-to-lidar", name='lsgan_patch_l1*1'+t)
config = wandb.config

# ==========================================================
########################### dataset ###########################
# ==========================================================

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

        return mm_scan, laser_scan

    def __len__(self):
        return len(self.transitions)


mm_dataset = MMDataset(paths)

loader = DataLoader(dataset=mm_dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=4)

# ==========================================================
########################### model ###########################
# ==========================================================


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        kernel = 3
        stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel, stride=stride),
            nn.ReLU()
        )

        dim = 64*59
        self.linear = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

#         self.n_fc1=nn.Linear(config.nz, 128)
#         self.n_fc2=nn.Linear(128, 128)

#         self.fc_combine=nn.Linear(128*2, 128)

        self.de_fc1 = nn.Sequential(
            nn.Linear(128, config.deconv_channel*config.deconv_dim),
            nn.ReLU()
        )

        self.de_conv = nn.Sequential(
            nn.ConvTranspose1d(config.deconv_channel, config.deconv_channel //
                               2, kernel, stride=stride, padding=config.padding),
            nn.ConvTranspose1d(config.deconv_channel//2, config.deconv_channel //
                               4, kernel, stride=stride, padding=config.padding),
            nn.ConvTranspose1d(config.deconv_channel//4, 1,
                               kernel, stride=stride, padding=config.padding),
        )
        self.adjust_linear = nn.Sequential(
            nn.Linear(config.adjust_linear, 241),
            nn.ReLU()
        )

    def encoder(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def decoder(self, x):
        x = self.de_fc1(x)
        x = x.view(-1, config.deconv_channel, config.deconv_dim)
        x = self.de_conv(x)
        x = self.adjust_linear(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
#         n = self.n_fc1(n)
#         n = self.n_fc2(n)

#         x = torch.cat((x,n),dim=-1)
#         x = self.fc_combine(x)

        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        kernel = 3
        stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel, stride=stride),
            nn.ReLU()
        )

        dim = 64*59
        self.linear = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


class DiscriminatorPatch(nn.Module):
    def __init__(self):
        super(DiscriminatorPatch, self).__init__()
        kernel = 3
        stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=kernel, stride=stride),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x = self.conv(x)

        return x


# ==========================================================
########################### visualize ###########################
# ==========================================================

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


# ==========================================================
########################### train config ###########################
# ==========================================================


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device, ', device)

netG = Generator().to(device)
# netD = Discriminator().to(device)
netD = DiscriminatorPatch().to(device)

wandb.watch(netG)
wandb.watch(netD)

# optimizers
optimizer_g = optim.Adam(netG.parameters(),
                         lr=config.learning_rate, betas=(config.beta1, 0.999))
optimizer_d = optim.Adam(netD.parameters(),
                         lr=config.learning_rate, betas=(config.beta1, 0.999))

# criterion
# gan_loss = nn.BCEWithLogitsLoss()
gan_loss = nn.MSELoss()
l1_loss = nn.L1Loss()


def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad

# ==========================================================
########################### train ###########################
# ==========================================================


step = 0
t = trange(config.epoch)

for epoch in t:
    for mm_scan, laser_scan in loader:
        x = mm_scan.to(device)
        y = laser_scan.to(device)

        # patch size 14
        fake_label = Variable(torch.Tensor(
            np.zeros((x.size(0), 1, 14))), requires_grad=False).to(device)
        real_label = Variable(torch.Tensor(
            np.ones((x.size(0), 1, 14))), requires_grad=False).to(device)

        fake_y = netG(x)

        ########################### train D ############################

        set_requires_grad(netD, True)
        optimizer_d.zero_grad()

        # fake
        fake_xy = torch.cat((x, fake_y), dim=1)
        pred_fake = netD(fake_xy.detach())
        loss_D_fake = gan_loss(pred_fake, fake_label)

        # real
        real_xy = torch.cat((x, y), dim=1)
        pred_real = netD(real_xy)
        loss_D_real = gan_loss(pred_real, real_label)

        # train
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_d.step()

        ########################### train G ############################

        set_requires_grad(netD, False)
        optimizer_g.zero_grad()

        pred_fake = netD(fake_xy)
        loss_G_gan = gan_loss(pred_fake, real_label)
        loss_G_l1 = l1_loss(fake_y, y) * config.lambda_l1
        loss_G = loss_G_gan + loss_G_l1
        loss_G.backward()
        optimizer_g.step()

        ########################### log ##################################

        metrics = {
            'loss_D_real': loss_D_real,
            'loss_D_fake': loss_D_fake,
            'loss_D': loss_D,
            'loss_G_gan': loss_G_gan,
            'loss_G_l1': loss_G_l1,
            'loss_G': loss_G,
        }
        wandb.log(metrics)
        step += 1
        t.set_description('setp: %d' % step)

        ########################### visualize ##################################

        if (step / len(loader)) % config.visualize_epoch == 0:
            fake_y = fake_y.detach().cpu().numpy().reshape(
                config.batch_size, -1)[:config.vis_num]
            laser_scan = laser_scan.detach().cpu().numpy().reshape(
                config.batch_size, -1)[:config.vis_num]
            mm_scan = mm_scan.detach().cpu().numpy().reshape(
                config.batch_size, -1)[:config.vis_num]

            examples = []
            for i, (y, mm, laser) in enumerate(zip(fake_y, mm_scan, laser_scan)):
                examples.append(laser_visual([mm, y, laser], show=False))

            wandb.log({"example_%d" %
                       step: [wandb.Image(img) for img in examples]})


################## save model #################
torch.save(netG.state_dict(), os.path.join(
    wandb.run.dir, "model.pth"))
