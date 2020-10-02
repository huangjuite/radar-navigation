import gym
from gym import error, spaces, utils
from gym.utils import seeding
import rospy
import time
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan, Image, CompressedImage, Imu
from std_srvs.srv import Empty
from std_msgs.msg import Int64
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState, GetPhysicsProperties, SetPhysicsProperties, SetPhysicsPropertiesRequest
from scipy.spatial.transform import Rotation as R


full_path = os.path.realpath(__file__)
INITIAL_STATES = np.genfromtxt(os.path.dirname(
    full_path) + '/init_pos.csv', delimiter=',')


class SubtCaveNobackEnv(gym.Env):
    metadata = {'render.modes': ['laser']}

    def __init__(self):
        rospy.init_node('gym_subt')
        self.laser_upper = LaserScan()
        self.laser_lower = LaserScan()
        self.image = CompressedImage()

        self.max_dis = 5
        self.max_step = 4096
        self.action_scale = {'linear': 1.5, 'angular': 0.8}

        self.total_dis = 0
        self.last_pos = None
        self.step_num = 0
        self.epi = 0

        # ServiceProxy
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_model = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        self.get_model = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)
        self.set_physics = rospy.ServiceProxy(
            '/gazebo/set_physics_properties', SetPhysicsProperties)
        self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics = rospy.ServiceProxy(
            '/gazebo/unpause_physics', Empty)

        # publisher subscriber
        self.pub_twist = rospy.Publisher('/X1/cmd_vel', Twist, queue_size=1)
        self.pub_episode = rospy.Publisher('/RL/epi', Int64, queue_size=1)
        self.sub_laser_upper = rospy.Subscriber(
            '/RL/scan', LaserScan, self.cb_laser, queue_size=1)

        # unpause physics
        self.pause_gym(False)

        # set real time factor
        physics = SetPhysicsPropertiesRequest()
        physics.time_step = 0.001
        physics.max_update_rate = 0.0
        physics.gravity.x = 0.0
        physics.gravity.y = 0.0
        physics.gravity.z = -9.8
        physics.ode_config.auto_disable_bodies: False
        physics.ode_config.sor_pgs_precon_iters = 0
        physics.ode_config.sor_pgs_iters = 50
        physics.ode_config.sor_pgs_w = 1.3
        physics.ode_config.sor_pgs_rms_error_tol = 0.0
        physics.ode_config.contact_surface_layer = 0.001
        physics.ode_config.contact_max_correcting_vel = 100.0
        physics.ode_config.cfm = 0.0
        physics.ode_config.erp = 0.2
        physics.ode_config.max_contacts = 20
        self.set_physics(physics)

        # action info
        state = self.get_observation()
        self.observation_space = spaces.Box(
            high=self.max_dis, low=0, shape=state.shape)
        self.action_space = spaces.Box(low=np.array(
            [0, -1]), high=np.array([1, 1]), dtype=np.float32)

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
        return np.clip(n, self.action_space.low[0], self.action_space.high[0])*bound

    def scale_angular(self, n, bound):
        return np.clip(n, self.action_space.low[1], self.action_space.high[1])*bound

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
        self.step_num += 1

        action[0] = self.scale_linear(action[0], self.action_scale['linear'])
        action[1] = self.scale_angular(action[1], self.action_scale['angular'])
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]

        # step
        self.pause_gym(False)
        self.pub_twist.publish(cmd_vel)
        state = self.get_observation()
        self.pause_gym(True)

        done = False
        info = {}
        self.pub_episode.publish(self.epi)
        ##################reward design##################
        angular_factor = (1-abs(action[1])) * 5
        reward = pow(2, angular_factor)
        reward *= action[0]

        min_dis = np.min(state)
        if min_dis < 0.75:
            done = True

        if min_dis < 1:
            reward = -50 - 300*(1 - min_dis)

        max_r = (2**5)
        min_r = -50 - 300*(1 - 0.75)
        norm_reward = (reward-min_r)/(max_r-min_r)
        ##################reward design###################
        if self.step_num >= self.max_step:
            done = True

        if done:
            self.epi += 1

        return state, norm_reward, done, info

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
        self.step_num = 0
        self.last_pos = None
        state = self.get_observation()

        return state

    def scan_once(self):
        data1 = None
        try:
            data1 = rospy.wait_for_message(
                '/RL/scan', LaserScan, timeout=5)
        except:
            print('fail to receive message')

        return np.array(data1.ranges)

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

        state = self.scan_once()
        return state

    def close(self):
        self.pause_gym(False)
        rospy.signal_shutdown('WTF')
