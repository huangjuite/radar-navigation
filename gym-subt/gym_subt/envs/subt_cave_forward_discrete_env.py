import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import rospy
import time
import numpy as np
import math
from matplotlib import pyplot as plt
from sensor_msgs.msg import LaserScan, Imu
from std_srvs.srv import Empty
from std_msgs.msg import Int64
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from scipy.spatial.transform import Rotation as R


full_path = os.path.realpath(__file__)
INITIAL_STATES = np.genfromtxt(os.path.dirname(
    full_path) + '/init_pos.csv', delimiter=',')

ACTIONS = [
    [0, 0.25],
    [0.25, 0.25],
    [0.5, 0.25],
    [0.5, 0.5],
    [0.5, 0.75],
    [0.75, 0.25],
    [1, 0.25],
    [1, 0],
    [1, -0.25],
    [0.75, -0.25],
    [0.5, -0.75],
    [0.5, -0.5],
    [0.5, -0.25],
    [0.25, -0.25],
    [0, -0.25], ]


class SubtCaveForwardDiscreteEnv(gym.Env):
    metadata = {'render.modes': ['laser']}

    def __init__(self):
        rospy.init_node('gym_subt')
        self.laser_upper = LaserScan()
        self.laser_lower = LaserScan()

        self.action_bound = {'linear': 1.5, 'angular': 1}

        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_model = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        self.get_model = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)

        self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics = rospy.ServiceProxy(
            '/gazebo/unpause_physics', Empty)

        self.pub_twist = rospy.Publisher('/X1/cmd_vel', Twist, queue_size=1)

        self.total_dis = 0
        self.last_pos = None
        self.max_dis = 5
        self.max_step = 1024
        self.epi = 0
        self.frame = 0
        self.last_action = [0, 0]
        self.pub_episode = rospy.Publisher('/RL/epi', Int64, queue_size=1)
        self.sub_laser_upper = rospy.Subscriber(
            '/RL/scan', LaserScan, self.cb_laser, queue_size=1)

        self.pause_gym(False)
        sample = self.get_observation()
        self.observation_space = spaces.Box(
            high=self.max_dis, low=0, shape=sample.shape)
        self.action_space = spaces.Discrete(len(ACTIONS))

    def seed(self, seed=None):
        np.random.seed(seed)
        return super().seed(seed=seed)

    def get_initial_state(self, name, id):
        # start position
        state_msg = ModelState()
        state_msg.model_name = name
        state_msg.pose.position.x = INITIAL_STATES[id][0]
        state_msg.pose.position.y = INITIAL_STATES[id][1]
        state_msg.pose.position.z = INITIAL_STATES[id][2]
        r = R.from_euler('z', np.random.uniform(-np.pi, np.pi))
        quat = r.as_quat()
        state_msg.pose.orientation.x = quat[0]
        state_msg.pose.orientation.y = quat[1]
        state_msg.pose.orientation.z = quat[2]
        state_msg.pose.orientation.w = quat[3]
        return state_msg

    def cb_laser(self, msg):
        pass

    def set_max_dis(self, max_d):
        self.max_dis = max_d

    def scale_linear(self, n, bound):
        return (max(min(n, 1), 0))*bound

    def scale_angular(self, n, bound):
        return (max(min(n, 1), -1))*bound

    def pause_gym(self, pause=True):
        srv_name = '/gazebo/pause_physics' if pause else '/gazebo/unpause_physics'
        rospy.wait_for_service(srv_name)
        try:
            if pause:
                self.pause_physics()
            else:
                self.unpause_physics()

        except (rospy.ServiceException) as e:
            print(e)

    def step(self, action):
        self.pause_gym(False)

        action = ACTIONS[action]
        self.last_action = action

        cmd_vel = Twist()
        cmd_vel.linear.x = self.scale_linear(
            action[0], self.action_bound['linear'])
        cmd_vel.angular.z = self.scale_angular(
            action[1], self.action_bound['angular'])
        self.pub_twist.publish(cmd_vel)

        state = self.get_observation()
        laser = state[:-2]

        self.pause_gym(True)

        self.frame += 1
        done = self.frame >= self.max_step
        info = {}
        self.pub_episode.publish(self.epi)
        ##################reward design##################

        # action reward [0,32]
        angular_factor = (1-abs(action[1])) * 5
        r_act = pow(2, angular_factor)
        r_act *= action[0]

        # dis reward [0,-32]
        min_avoid = 1
        min_allow = 0.75
        min_dis = np.min(laser)
        r_dis = 0
        
        if min_dis <= min_avoid:
            r_dis = -32 * ((min_avoid-min_dis)/(min_avoid-min_allow))
            done = True if min_dis <= min_allow else done

        # total, clip
        # r_max = (2**5)
        # r_min = -32
        reward = r_act + r_dis
        # reward = (reward/abs(reward))*math.log(1+abs(reward))
        reward = math.tanh(reward/24)
        
        ##################reward design###################

        if done:
            self.frame = 0
            self.epi += 1

        return state, reward, done, info

    def reset(self):
        states = np.arange(0, len(INITIAL_STATES)).tolist()
        agent_idx = states.pop(np.random.randint(0, len(states)))
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.reset_model(self.get_initial_state('X1', agent_idx))
        except(rospy.ServiceException) as e:
            print(e)

        self.pause_gym(False)

        self.reward = 0
        self.total_dis = 0
        self.last_pos = None
        state = self.get_observation()

        return state

    def scan_once(self):
        data1 = None
        while data1 is None:
            try:
                data1 = rospy.wait_for_message(
                    '/RL/scan', LaserScan, timeout=5)
            except:
                print('fail to receive message')

        ranges = np.array(data1.ranges)
        ranges = np.clip(ranges, 0, self.max_dis)

        return ranges

    def get_observation(self):
        agent = ModelState()
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            agent = self.get_model('X1', '')
        except (rospy.ServiceException) as e:
            print(e)

        new_pos = np.array(
            [agent.pose.position.x, agent.pose.position.y, agent.pose.position.z])
        if self.last_pos is not None:
            self.total_dis += np.linalg.norm(new_pos-self.last_pos)
        self.last_pos = new_pos

        state = self.scan_once()
        state = np.append(state, self.last_action)

        return state

    def close(self):
        self.pause_gym(False)
        rospy.signal_shutdown('WTF')
