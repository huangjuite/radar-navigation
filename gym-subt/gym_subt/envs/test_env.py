import rospy
import time
import numpy as np
import math
import sys
from matplotlib import pyplot as plt
from sensor_msgs.msg import LaserScan, Imu, Joy
from std_srvs.srv import Empty
from std_msgs.msg import Int64
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
import gym
import torch

env_id = "gym_subt:subt-cave-forward-v0"
env = gym.make(env_id)
env.max_step = 512


def cb_joy(msg):
    action = [msg.axes[4], msg.axes[3]]
    s, r, d, info = env.step(action)
    print(r)

    s = torch.Tensor([s])
    s, pos = torch.split(s, s.shape[1]-2, dim=1)
    # s = s.reshape(1, 4, -1)
    # track = track.reshape(1, 10, -1)

    if d:
        env.reset()


env.reset()
sub_joy = rospy.Subscriber('/joy', Joy, cb_joy, queue_size=1)

rospy.spin()
