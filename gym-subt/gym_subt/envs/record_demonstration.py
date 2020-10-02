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
import h5py
import pickle as pkl
import datetime as dt

# f = open("demo.pkl",'rb')
# data = pkl.load(f)
# print(data[0])


class Demonstrator():
    def __init__(self):
        self.demonstration = []

        env_id = "gym_subt:subt-cave-forward-goal-v0"
        self.env = gym.make(env_id)
        self.env.reset()
        self.s_ = None
        self.epi = 0
        self.flag = False
        self.frames = 0

        sub_joy = rospy.Subscriber('/joy', Joy, self.cb_joy, queue_size=1)

    def cb_joy(self, msg):
        if not self.flag:
            self.flag = msg.buttons[6]
            print('start')
        else:
            action = np.array([msg.axes[4], msg.axes[3]])
            s, r, d, info = self.env.step(action)
            self.frames += 1

            if self.s_ is not None:
                trans = [s, action, r, self.s_, d]
                self.demonstration.append(trans)

            self.s_ = s

            if d:
                print('epi:%d, step:%d' % (self.epi, self.frames))
                self.epi += 1
                self.s_ = None
                self.flag = False
                self.env.reset()

    def on_shutdown(self):
        ds = dt.datetime.now()
        t = str(ds).split(".")[0].replace(":", "-").replace(" ", "-")
        output = open('demo_%s.pkl' % t, 'wb')
        pkl.dump(self.demonstration, output)
        output.close()


if __name__ == "__main__":
    demonstrator = Demonstrator()
    rospy.on_shutdown(demonstrator.on_shutdown)
    rospy.spin()
