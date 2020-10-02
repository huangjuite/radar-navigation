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
from scipy.spatial.transform import Rotation as R
# from tf.transformations import quaternion_from_euler, euler_from_quaternion


# [x,y,z]
INITIAL_STATES = [[0, 0, 0],
                  [15.69, 20.81, 0.13],
                  [60.64, 99.35, 0.13],
                  [119.44, 18.53, 0.13],
                  [140.14, -21.77, 0.13],
                  [158.35, -80.51, 0.13],
                  [199.64, 40.92, 0.13]]


class SubtCaveGoalEnv(gym.Env):
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

        self.max_dis = 5
        self.goal = [0, 0]
        self.sub_laser_upper = rospy.Subscriber(
            '/RL/scan/upper', LaserScan, self.cb_laser_upper, queue_size=1)
        self.sub_laser_lower = rospy.Subscriber(
            '/RL/scan/mid', LaserScan, self.cb_laser_mid, queue_size=1)
        self.sub_laser_lower = rospy.Subscriber(
            '/RL/scan/lower', LaserScan, self.cb_laser_lower, queue_size=1)
        # self.sub_imu = rospy.Subscriber(
        #     '/X1/imu/data', Imu, self.cb_imu, queue_size=1)
        # self.sub_image_raw = rospy.Subscriber('X1/rgbd_camera/rgb/image_raw/compressed',CompressedImage,self.cb_image,queue_size=1)

        # self.reset()

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

    def cb_image(self, msg):
        self.image = msg
    
    def seed(self, seed):
        np.random.seed(seed)

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

    def scale_linear(self, n, bound):
        return max(min(n, 1), -1)*bound
        # return (max(min(n, 1), -1)+1)/2.0*bound

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
        self.pub_twist.publish(cmd_vel)

        state = self.get_observation()
        laser = state['pc']
        goal_angle = state['goal'][0]

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
        reward *= np.pi-abs(goal_angle) if (action[0] > 0) else 1
        # reward *= action[0]

        min_dis = 100
        for (x,y,z), dis in np.ndenumerate(laser):
            dis = np.squeeze(dis)
            if dis < min_dis:
                min_dis = dis
            if dis < 0.75:
                done = True

        if min_dis < 1.2:
            reward = -50 - 300*(1.2 - min_dis)

        max_r = (2**5)*np.pi
        min_r = -50 - 300*(1.2 - 0.75)
        # norm_reward = (reward-min_r)/(max_r-min_r)*0.6 + 0.4*(np.pi-abs(goal_angle))/np.pi
        norm_reward = (reward-min_r)/(max_r-min_r)
        ##################reward design###################

        return state, norm_reward, done, info

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
        state = self.get_observation()

        time.sleep(0.5)

        return state

    def get_observation(self):
        agent = ModelState()
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            agent = self.get_model('X1', '')
        except (rospy.ServiceException) as e:
            print("/gazebo/get_model_state service call failed")

        data1 = None
        data2 = None
        data3 = None
        while data1 is None or data2 is None or data3 is None:
            try:
                data1 = rospy.wait_for_message(
                    '/RL/scan/upper', LaserScan, timeout=5)
                data2 = rospy.wait_for_message(
                    '/RL/scan/mid', LaserScan, timeout=5)
                data3 = rospy.wait_for_message(
                    '/RL/scan/lower', LaserScan, timeout=5)
            except:
                print('fail to receive message')

        pc = []
        laser1 = []
        for i, dis in enumerate(list(data1.ranges)):
            if dis > self.max_dis:
                dis = self.max_dis
            laser1.append([dis])
        pc.append(laser1)
        laser2 = []
        for i, dis in enumerate(list(data2.ranges)):
            if dis > self.max_dis:
                dis = self.max_dis
            laser2.append([dis])
        pc.append(laser2)
        laser3 = []
        for i, dis in enumerate(list(data3.ranges)):
            if dis > self.max_dis:
                dis = self.max_dis
            laser3.append([dis])
        pc.append(laser3)


        quat = (agent.pose.orientation.x, agent.pose.orientation.y,
                agent.pose.orientation.z, agent.pose.orientation.w)
        _, _, yaw = euler_from_quaternion(quat)
        robot_position = [agent.pose.position.x, agent.pose.position.y]
        goal_angle = self.get_goal_angle(yaw, robot_position, self.goal)
        state = {'pc':np.array(pc),'goal':np.array([goal_angle/180.0*np.pi])}

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
