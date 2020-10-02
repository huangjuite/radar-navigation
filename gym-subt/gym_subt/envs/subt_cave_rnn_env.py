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


class SubtCaveRnnEnv(gym.Env):
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
        self.max_dis = 5
        self.sub_laser_upper = rospy.Subscriber(
            '/RL/scan', LaserScan, self.cb_laser, queue_size=1)

        # self.memory = self.scan_once()
        # for i in range(2):
        #     self.memory = np.vstack((self.memory, self.scan_once()))

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

    def cb_laser(self, msg):
        pass

    def set_max_dis(self, max_d):
        self.max_dis = max_d

    def scale_linear(self, n, bound):
        return (max(min(n, 1), -1))*bound

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

        state = self.get_observation()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        done = False
        info = {}

        ##################reward design##################
        angular_factor = (1-abs(action[1])) * 7
        reward = pow(2, angular_factor)
        reward *= action[0]

        min_dis = 100
        for i, dis in np.ndenumerate(state):
            if dis < min_dis:
                min_dis = dis
            if dis < 0.75:
                done = True

        if min_dis < 1.2:
            reward = -150

        if done:
            reward = -200

        max_r = (2**7)
        min_r = -200
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
        # self.memory = self.scan_once()
        # for i in range(2):
        #     self.memory = np.vstack((self.memory, self.scan_once()))

        state = self.get_observation()

        time.sleep(0.5)

        return state

    def scan_once(self):
        data1 = None
        while data1 is None:
            try:
                data1 = rospy.wait_for_message(
                    '/RL/scan', LaserScan, timeout=5)
            except:
                print('fail to receive message')

        laser1 = []
        for i, dis in enumerate(list(data1.ranges)):
            if dis > self.max_dis:
                dis = self.max_dis
            laser1.append(dis)
        return np.array(laser1)

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

        # self.memory = np.vstack((self.memory[1:, :], self.scan_once()))

        # state = np.reshape(self.memory,(-1))
        # state = self.memory
        state = self.scan_once()
        return state

    def get_goal_angle(self, robot_yaw, robot, goal):
        robot_angle = np.degrees(robot_yaw)
        p1 = [robot[0], robot[1]]
        p2 = [robot[0], robot[1]+1.]
        p3 = goal
        angle = self.get_angle(p1, p2, p3)
        result = angle - robot_angle
        result = self.angle_range(-(result + 90.))
        return result

    def get_angle(self, p1, p2, p3):
        v0 = np.array(p2) - np.array(p1)
        v1 = np.array(p3) - np.array(p1)
        angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
        return np.degrees(angle)

    # limit the angle to the range of [-180, 180]
    def angle_range(self, angle):
        if angle > 180:
            angle = angle - 360
            angle = self.angle_range(angle)
        elif angle < -180:
            angle = angle + 360
            angle = self.angle_range(angle)
        return angle

    def close(self):
        self.unpause_physics()
        rospy.signal_shutdown('WTF')
