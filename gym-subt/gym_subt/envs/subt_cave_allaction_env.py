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
# from tf.transformations import quaternion_from_euler


# [x,y,z]
INITIAL_STATES = [[0,0,0],
                  [15.69,20.81,0.13],
                  [60.64,99.35,0.13],
                  [119.44,18.53,0.13],
                  [140.14,-21.77,0.13],
                  [158.35,-80.51,0.13],
                  [199.64,40.92,0.13]]


class SubtCaveAllactionEnv(gym.Env):
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
        self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics = rospy.ServiceProxy(
            '/gazebo/unpause_physics', Empty)

        self.pub_twist = rospy.Publisher('/X1/cmd_vel', Twist, queue_size=1)

        self.max_dis = 2
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
        quat = quaternion_from_euler(0, 0, np.random.uniform(-np.pi, np.pi))
        state_msg.pose.orientation.x = quat[0]
        state_msg.pose.orientation.y = quat[1]
        state_msg.pose.orientation.z = quat[2]
        state_msg.pose.orientation.w = quat[3]
        return state_msg

    def cb_image(self, msg):
        self.image = msg

    def cb_laser_upper(self, msg):
        self.laser_upper = msg

    def cb_laser_lower(self, msg):
        self.laser_lower = msg
    
    def set_max_dis(self,max_d):
        self.max_dis = max_d

    def scale(self, n, bound):
        return max(min(n, 1), -1)*bound

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        cmd_vel = Twist()
        cmd_vel.linear.x = self.scale(action[0], self.action_bound['linear'])
        cmd_vel.angular.z = self.scale(action[1], self.action_bound['angular'])
        self.pub_twist.publish(cmd_vel)

        laser = self.get_observation()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        done = False
        info = {}

        ##################reward design##################
        angular_factor = (1-abs(action[1])) * 5

        reward = pow(2, angular_factor)
        # reward *= 4 if (action[0] > 0) else 1
        reward *= action[0]

        min_dis = 100
        for i, dis in enumerate(laser):
            if dis < min_dis:
                min_dis = dis
            if dis < 0.75:
                done = True

        if min_dis < 1.2:
            reward = -50 - 300*(1.2 - min_dis)

        max_r = 32.0 # 2^5*4
        min_r = -185.0
        norm_reward = (reward-min_r)/(max_r-min_r)

        ##################reward design###################

        return laser, norm_reward, done, info

    def reset(self):
        self.reset_model(self.get_initial_state(
            np.random.randint(0, len(INITIAL_STATES))))
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        self.reward = 0
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
        for i, dis in enumerate(list(data1.ranges)):
            if dis > self.max_dis:
                dis = self.max_dis
            laser.append(dis)
        for i, dis in enumerate(list(data2.ranges)):
            if dis > self.max_dis:
                dis = self.max_dis
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
