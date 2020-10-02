import gym
from gym import error, spaces, utils
from gym.utils import seeding
import rospy
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan, Image, CompressedImage, Imu
from subt_rl.msg import State
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
# from tf.transformations import quaternion_from_euler, euler_from_quaternion


# [x,y,z]
INITIAL_STATES = [
    [0, 0, 0],
    [15.69, 20.81, 0.13],
    [60.64, 99.35, 0.13],
    [119.44, 18.53, 0.13],
    [140.14, -21.77, 0.13],
    [158.35, -80.51, 0.13],
    [199.64, 40.92, 0.13],
    [153.95, -20.35, 0.13],
    [127.20, -41.11, 0.13],
    [110.64, -20.08, 0.13],
    [96.67, -4.71, 0.13],
    [78.82, 12.17, 0.13]]


class SubtCave3DEnv(gym.Env):
    metadata = {'render.modes': ['laser']}

    def __init__(self):
        rospy.init_node('gym_subt')
        self.laser_upper = LaserScan()
        self.laser_lower = LaserScan()
        self.image = CompressedImage()

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

        self.total_dis = 0
        self.last_pos = None
        self.sub_state = rospy.Subscriber(
            '/RL/state', State, self.cb_state, queue_size=1)

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

    def cb_state(self, msg):
        pass

    def scale_linear(self, n, bound):
        return (max(min(n, 1), 0))*bound

    def scale_angular(self, n, bound):
        return (max(min(n, 1), -1))*bound

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
        self.pub_twist.publish(cmd_vel)

        state, min_dis = self.get_observation()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        done = False
        info = {}
        norm_reward = 0

        ##################reward design##################
        angular_factor = (1-abs(action[1])) * 5
        reward = pow(2, angular_factor)
        reward *= action[0]

        if min_dis < 0.76:
            done = True
            reward = 0

        max_r = (2**5)
        min_r = 0
        norm_reward = (reward-min_r)/(max_r-min_r)

        ##################reward design###################

        return state, norm_reward, done, info

    def reset(self):
        states = np.arange(0, len(INITIAL_STATES)).tolist()
        agent_idx = states.pop(np.random.randint(0, len(states)))
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.reset_model(self.get_initial_state('X1', agent_idx))
        except(rospy.ServiceException) as e:
            print(e)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except(rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        self.reward = 0
        self.total_dis = 0
        self.last_pos = None
        state, close_d = self.get_observation()

        time.sleep(0.5)

        return state

    def get_observation(self):
        agent = ModelState()
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            agent = self.get_model('X1', '')
        except (rospy.ServiceException) as e:
            print("/gazebo/get_model_state service call failed")

        new_pos = np.array(
            [agent.pose.position.x, agent.pose.position.y, agent.pose.position.z])
        if self.last_pos is not None:
            self.total_dis += np.linalg.norm(new_pos-self.last_pos)
        self.last_pos = new_pos

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(
                    '/RL/state', State, timeout=5)
            except:
                print('fail to receive message')

        close_dis = data.closestDistance
        observation = data.observation.data
        dim = data.observation.layout.dim
        state = np.array(observation).reshape(
            dim[0].size, dim[1].size, dim[2].size)

        state = np.expand_dims(state, 0)

        return state, close_dis

    def close(self):
        self.unpause_physics()
        rospy.signal_shutdown('WTF')
