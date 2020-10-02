import gym
from gym import error, spaces, utils
from gym.utils import seeding
import rospy
import time
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan, Image, CompressedImage, Imu
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
# from tf.transformations import quaternion_from_euler, euler_from_quaternion


# [x,y,z]
INITIAL_STATES = [[0, 0, 0],
                  [15.69, 20.81, 0.13],
                  [60.64, 99.35, 0.13],
                  [119.44, 18.53, 0.13],
                  [140.14, -21.77, 0.13],
                  [158.35, -80.51, 0.13],
                  [199.64, 40.92, 0.13]]


class SubtCaveBaselineEnv(gym.Env):
    metadata = {'render.modes': ['laser']}

    def __init__(self):
        rospy.init_node('gym_subt')
        self.laser_upper = LaserScan()
        self.laser_lower = LaserScan()
        self.image = CompressedImage()
        self.cv_bridge = CvBridge()
        # self.fig, self.axs = plt.subplots(
        #     2, 1, subplot_kw=dict(projection='polar'))

        self.action_bound = {'linear': 1.5, 'angular': 0.8}

        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_model = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        self.get_model = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)

        self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics = rospy.ServiceProxy(
            '/gazebo/unpause_physics', Empty)

        self.pub_twist = rospy.Publisher('/X1/cmd_vel', Twist, queue_size=1)

        self.max_dis = 4
        self.goal = [0, 0]
        self.last_dist = 0
        self.last_action = [0, 0]
        self.sub_laser_lower = rospy.Subscriber(
            '/RL/scan/mid', LaserScan, self.cb_laser_mid, queue_size=1)

        self.memory = None
        for i in range(3):
            if self.memory is None:
                self.memory = self.scan_once()
            elif self.memory.shape[1] < 3:
                self.memory = np.hstack((self.memory, self.scan_once()))

    def get_initial_state(self, name, id):
        # start position
        state_msg = ModelState()
        state_msg.model_name = name
        state_msg.pose.position.x = INITIAL_STATES[id][0]
        state_msg.pose.position.y = INITIAL_STATES[id][1]
        state_msg.pose.position.z = INITIAL_STATES[id][2]
        quat = quaternion_from_euler(0, 0, np.random.uniform(-np.pi, np.pi))
        state_msg.pose.orientation.x = quat[0]
        state_msg.pose.orientation.y = quat[1]
        state_msg.pose.orientation.z = quat[2]
        state_msg.pose.orientation.w = quat[3]
        return state_msg

    def cb_imu(self, msg):
        pass

    def cb_laser_upper(self, msg):
        pass

    def cb_laser_mid(self, msg):
        pass

    def cb_laser_lower(self, msg):
        pass

    def set_max_dis(self, max_d):
        self.max_dis = max_d

    def seed(self, seed):
        np.random.seed(seed)

    def scale_linear(self, n, bound):
        return (max(min(n, 1), -1)+1)/2.0*bound

    def scale_angular(self, n, bound):
        return max(min(n, 1), -1)*bound

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        cmd_vel = Twist()
        cmd_vel.linear.x = self.scale_linear(
            action[0], self.action_bound['linear'])
        cmd_vel.angular.z = self.scale_angular(
            action[1], self.action_bound['angular'])
        self.last_action = action
        self.pub_twist.publish(cmd_vel)

        state = self.get_observation()
        laser = state['pc'][:, -1]
        goal = state['goal']
        speed = state['speed']

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        done = False
        info = {}

        ##################reward design##################
        r_a = abs(speed[1]) * -0.1 if abs(speed[1]) > 0.7 else 0
        r_c = 0

        dis2g = math.sqrt(goal[0]**2+goal[1]**2)
        r_g = 2.5 * (self.last_dist - dis2g)
        self.last_dist = dis2g
        if dis2g < 0.1:
            r_g = 15
            done = True

        min_dis = 100
        for x, dis in np.ndenumerate(laser):
            if dis < min_dis:
                min_dis = dis
            if dis < 0.75:
                r_c = -15
                done = True

        # print r_a, r_c, r_g
        reward = r_a + r_c + r_g
        ##################reward design###################

        return state, reward, done, info

    def reset(self):
        states = np.arange(0, len(INITIAL_STATES)).tolist()
        agent_idx = states.pop(np.random.randint(0, len(states)))
        goal_idx = states.pop(np.random.randint(0, len(states)))
        self.goal = INITIAL_STATES[goal_idx][:-1]
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.reset_model(self.get_initial_state('X1', agent_idx))
            self.reset_model(self.get_initial_state('Radio', goal_idx))
        except(rospy.ServiceException) as e:
            print(e)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except(rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        self.reward = 0
        for i in range(3):
            if self.memory is None:
                self.memory = self.scan_once()
            elif self.memory.shape[1] < 3:
                self.memory = np.hstack((self.memory, self.scan_once()))
            else:
                self.memory = np.hstack(
                    (self.memory[:, 1:], self.scan_once()))

        state = self.get_observation()
        goal = state['goal']
        self.last_dist = math.sqrt(goal[0]**2+goal[1]**2)
        self.last_action = [0, 0]

        time.sleep(0.5)

        return state

    def scan_once(self):
        data1 = None
        while data1 is None:
            try:
                data1 = rospy.wait_for_message(
                    '/RL/scan/mid', LaserScan, timeout=5)
            except:
                print('fail to receive message')

        laser1 = []
        for i, dis in enumerate(list(data1.ranges)):
            if dis > self.max_dis:
                dis = self.max_dis
            laser1.append([dis])

        return np.array(laser1)

    def get_observation(self):
        agent = ModelState()
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            agent = self.get_model('X1', '')
        except (rospy.ServiceException) as e:
            print("/gazebo/get_model_state service call failed")

        self.memory = np.hstack((self.memory[:, 1:], self.scan_once()))

        relative_goal = [self.goal[0]-agent.pose.position.x,
                         self.goal[1]-agent.pose.position.y]

        state = {'pc': self.memory, 'goal': np.array(
            relative_goal), 'speed': np.array(self.last_action)}

        return state

    def close(self):
        self.unpause_physics()
        rospy.signal_shutdown('WTF')
