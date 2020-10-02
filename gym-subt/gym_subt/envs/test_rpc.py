import zerorpc
import numpy as np
import pickle
import time
import torch
import torchvision
import rospy
from sensor_msgs.msg import LaserScan, Imu, Joy


env = zerorpc.Client()
try:
    env.connect("tcp://140.113.148.104:4242")
except Exception as e:
    print(e)

env.reset()

for i in range(100):
    action = [1,0]
    data = env.step(action)
    s,r,d,i = pickle.loads(data)
    s = torch.Tensor([s])
    s, track = torch.split(s, s.shape[1]-30, dim=1)
    s = s.reshape(1, 4, -1)
    track = track.reshape(1, 10, -1)
    print(track)

    if d:
        env.reset()
    