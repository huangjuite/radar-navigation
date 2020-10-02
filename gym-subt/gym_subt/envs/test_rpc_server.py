import zerorpc
import numpy as np
import pickle
import torch
import torchvision
import rospy
import time
import numpy as np
import math
import sys
import gym


class GymRPCserver(object):
    def __init__(self):
        env_id = "gym_subt:subt-cave-forward-goal-v0"
        self.env = gym.make(env_id)

        self.model = torchvision.models.densenet121()
        # print(self.model)
        # torch.save(model.state_dict(),'./test_model.pth')
    
    def step(self, action):
        print('step')
        data = self.env.step(action)
        return pickle.dumps(data)

    def reset(self):
        print('reset')
        self.env.reset()
        return
        
    def update_model(self):
        # model = open("./test_model.pth","r")
        # print(model.read())
        return pickle.dumps(self.model)


    def giant_data(self,shape=(100,100)):
        a = np.ones(shape)
        serialized = pickle.dumps(a)
        
        return serialized

s = zerorpc.Server(GymRPCserver())
s.bind("tcp://0.0.0.0:4242")
print("rpc running")
s.run()
