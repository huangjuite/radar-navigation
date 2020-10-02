import gym
from gym import error, spaces, utils
from gym.utils import seeding
import rospy
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan, Image, CompressedImage
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


# [x,y,z,x,y,z,w]
INITIAL_STATES = [[14.38, -0.05, -0.74, 0, 0.14, 0, 1],
                  [89.1, 0, -19.87, 0, 0, 0, -1],
                  [100, 22.5, -20, 0, 0, -0.7, -0.7],
                  [100, 63.13, -20, 0, 0, -0.7, -0.7],
                  [80, 51, -20, 0, 0, -0.74, 0.66],
                  [159, 0, -20, 0, 0, 0, -1],
                  [80, 60, -20, 0, 0, -0.7, -0.7],
                  [116, 40, -20, 0, 0, 1, 0],
                  [140, 43, -20, 0, 0, 0.7, 0.7]]


class SubtTunnelEnv(gym.Env):
    metadata = {'render.modes': ['laser']}

    def __init__(self):
        rospy.init_node('gym_subt')
        self.laser_upper = LaserScan()
        self.laser_lower = LaserScan()
        self.image = CompressedImage()
        self.cv_bridge = CvBridge()
        # self.fig, self.axs = plt.subplots(
        #     2, 1, subplot_kw=dict(projection='polar'))

        # [Twist.linear.x, Twist.angular.z]
        self.actions = [[0.5, -0.8],
                        [1.5, -0.8],
                        [1.5, -0.4],
                        [1.5, 0.0],
                        [1.5, 0.4],
                        [1.5, 0.8],
                        [0.5, 0.8]]

        # self.laser_shape = [None, 2, 21, 1]

        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_model = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics = rospy.ServiceProxy(
            '/gazebo/unpause_physics', Empty)

        self.pub_twist = rospy.Publisher('/X1/cmd_vel', Twist, queue_size=1)

        self.sub_laser_upper = rospy.Subscriber(
            '/RL/scan/upper', LaserScan, self.cb_laser_upper, queue_size=1)
        self.sub_laser_lower = rospy.Subscriber(
            '/RL/scan/lower', LaserScan, self.cb_laser_lower, queue_size=1)
        # self.sub_image_raw = rospy.Subscriber('X1/rgbd_camera/rgb/image_raw/compressed',CompressedImage,self.cb_image,queue_size=1)

        self.reset()

    def get_initial_state(self, id):
        # start position
        state_msg = ModelState()
        state_msg.model_name = 'X1'
        state_msg.pose.position.x = INITIAL_STATES[id][0]
        state_msg.pose.position.y = INITIAL_STATES[id][1]
        state_msg.pose.position.z = INITIAL_STATES[id][2]
        state_msg.pose.orientation.x = INITIAL_STATES[id][3]
        state_msg.pose.orientation.y = INITIAL_STATES[id][4]
        state_msg.pose.orientation.z = INITIAL_STATES[id][5]
        state_msg.pose.orientation.w = INITIAL_STATES[id][6]
        return state_msg

    def cb_image(self, msg):
        self.image = msg

    def cb_laser_upper(self, msg):
        self.laser_upper = msg

    def cb_laser_lower(self, msg):
        self.laser_lower = msg

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        cmd_vel = Twist()
        cmd_vel.linear.x = self.actions[action][0]
        cmd_vel.angular.z = self.actions[action][1]
        self.pub_twist.publish(cmd_vel)

        laser = self.get_observation()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        done = False
        info = {}
        factor = len(self.actions)//2-abs(action-len(self.actions)//2)
        reward = 2**factor

        too_dam_close = False
        for i, dis in enumerate(laser):
            if dis < 0.9:
                too_dam_close = True
            if dis < 0.75:
                done = True

        if too_dam_close:
            reward = -50
        if done:
            reward = -200

        return laser, reward, done, info

    def reset(self):
        self.reset_model(self.get_initial_state(
            np.random.randint(0, len(INITIAL_STATES))))
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        self.reward = 0
        rospy.loginfo('reset model')
        laser = self.get_observation()

        time.sleep(0.5)

        return laser

    def get_observation(self):
        data1 = None
        data2 = None
        while data1 is None or data2 is None:
            try:
                data1 = rospy.wait_for_message(
                    '/RL/scan/upper', LaserScan, timeout=5)
                data2 = rospy.wait_for_message(
                    '/RL/scan/lower', LaserScan, timeout=5)
            except:
                pass

        laser = []
        max_dis = 1.5
        for i, dis in enumerate(list(data1.ranges)):
            if dis > max_dis:
                dis = max_dis
            laser.append(dis)
        for i, dis in enumerate(list(data2.ranges)):
            if dis > max_dis:
                dis = max_dis
            laser.append(dis)
        return np.array(laser)

    def render(self, mode='laser'):
        pass
        # observation = self.get_observation()
        # theta = np.arange(0, np.pi, 0.157)
        # self.axs[0].set_title("upper laser")
        # self.axs[1].set_title("lower laser")
        # self.axs[0].clear()
        # self.axs[0].plot(theta, observation[0])
        # self.axs[1].clear()
        # self.axs[1].plot(theta, observation[1])
        # plt.pause(0.001)

    def close(self):
        self.unpause_physics()
        rospy.signal_shutdown('WTF')
