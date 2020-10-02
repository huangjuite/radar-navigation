import gym
from gym import error, spaces, utils
from gym.utils import seeding
import rospy
import time
import numpy as np
import math
import sys
import os
from matplotlib import pyplot as plt
from sensor_msgs.msg import LaserScan, Imu
from std_srvs.srv import Empty
from std_msgs.msg import Int64
from geometry_msgs.msg import Twist, Vector3
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState, GetPhysicsProperties, SetPhysicsProperties, SetPhysicsPropertiesRequest
from scipy.spatial.transform import Rotation as R

full_path = os.path.realpath(__file__)
INITIAL_STATES = np.genfromtxt(os.path.dirname(
    full_path) + '/init_pos.csv', delimiter=',')


class SubtCaveForwardGoalEnv(gym.Env):
    metadata = {'render.modes': ['laser']}

    def __init__(self):
        rospy.init_node('gym_subt')

        # env parameter
        self.max_dis = 10
        self.max_step = 4096
        self.track_num = 10
        self.state_num = 4
        self.action_scale = {'linear': 1.5, 'angular': 0.8}

        # global variable
        self.epi = 0
        self.frame = 0
        self.last_dis = 0
        self.total_dis = 0
        self.goal = np.array([0, 0])
        self.last_action = [0, 0]
        self.last_pos = None
        self.pos_track = None
        self.state_stack = None

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
        # physics.max_update_rate = 1500.0
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

        # state info
        self.info = [(self.state_num, 241),
                     (self.track_num, 3)]

    def seed(self, seed=None):
        np.random.seed(seed)
        return super().seed(seed=seed)

    def get_initial_state(self, name, position):
        # start position
        state_msg = ModelState()
        state_msg.model_name = name
        state_msg.pose.position.x = position[0]
        state_msg.pose.position.y = position[1]
        state_msg.pose.position.z = position[2]
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

    def scale_pose(self, value):
        if value > 0:
            return math.log(1 + value)
        elif value < 0:
            return -math.log(1 + abs(value))

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

        # action[0] ,map to [0, 1]
        # action[0] = (action[0]+1)/2
        # self.last_action = action
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

        laser = self.state_stack[-1]

        # calculate relative pos
        location_dif = self.goal - self.last_pos[:2]

        self.frame += 1
        done = self.frame >= self.max_step

        self.pub_episode.publish(self.epi)
        ##################reward design##################

        # r_goal
        dis_to_g = np.linalg.norm(location_dif)
        r_goal = 0.2 if dis_to_g < self.last_dis-0.01 else -0.2

        # r_move speed bonus > 0.03 3cm between frame
        speed = abs(self.last_dis-dis_to_g)
        r_move = 0.1 if speed > 0.03 else -0.1

        # r_reach
        r_reach = 0
        if dis_to_g < 3:
            r_reach = 1500
            done = True

        # r_dis
        min_allow = 0.75
        min_dis = np.min(laser)
        r_dis = 0
        if min_dis < min_allow:
            r_dis = -10
            done = True

        # totals
        reward = r_goal + r_dis + r_reach + r_move

        # if reward > 0:
        #     reward = math.log(1+abs(reward))
        # elif reward < 0:
        #     reward = -1 * math.log(1+abs(reward))

        ##################reward design###################

        self.last_dis = dis_to_g

        if done:
            self.frame = 0
            self.epi += 1

        return state, reward, done, self.info

    def reset(self):
        agent_idx = np.random.choice(INITIAL_STATES.shape[0], 1)[0]
        poses = np.tile(INITIAL_STATES[agent_idx],
                        (INITIAL_STATES.shape[0], 1))
        dis = np.linalg.norm(INITIAL_STATES-poses, axis=1)
        goal_idx = np.argsort(dis)[1]

        self.goal = INITIAL_STATES[goal_idx][:-1]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.reset_model(self.get_initial_state(
                'X1', INITIAL_STATES[agent_idx]))
            self.reset_model(self.get_initial_state(
                'Radio', INITIAL_STATES[goal_idx]))
        except(rospy.ServiceException) as e:
            print(e)

        self.pause_gym(False)

        self.reward = 0
        self.last_dis = 0
        self.total_dis = 0
        self.last_pos = None
        self.state_stack = None
        self.pos_track = None
        state = self.get_observation()

        return state

    def scan_once(self):
        data = LaserScan()
        try:
            data = rospy.wait_for_message(
                '/RL/scan', LaserScan, timeout=5)
        except:
            print('fail to receive message')

        ranges = np.array(data.ranges)
        ranges = np.clip(ranges, 0, self.max_dis)

        return ranges

    def get_observation(self):
        # obtain
        agent = ModelState()
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            agent = self.get_model('X1', '')
        except (rospy.ServiceException) as e:
            print(e)
        new_pos = np.array(
            [agent.pose.position.x, agent.pose.position.y, agent.pose.position.z])

        # add travel distance
        if self.last_pos is not None:
            self.total_dis += np.linalg.norm(new_pos-self.last_pos)
        self.last_pos = new_pos

        # caculate angle diff
        diff = self.goal - self.last_pos[:2]
        r = R.from_quat([agent.pose.orientation.x,
                         agent.pose.orientation.y,
                         agent.pose.orientation.z,
                         agent.pose.orientation.w])
        yaw = r.as_euler('zyx')[0]
        angle = math.atan2(diff[1], diff[0]) - yaw
        if angle >= np.pi:
            angle -= 2*np.pi
        elif angle <= -np.pi:
            angle += 2*np.pi

        # update pose tracker
        diff = np.array([self.scale_pose(v) for v in diff])
        track_pos = np.append(diff, angle)
        if self.pos_track is None:
            self.pos_track = np.tile(track_pos, (self.track_num, 1))
        else:
            self.pos_track[:-1] = self.pos_track[1:]
            self.pos_track[-1] = track_pos

        # prepare laser scan stack
        scan = self.scan_once()
        if self.state_stack is None:
            self.state_stack = np.tile(scan, (self.state_num, 1))
        else:
            self.state_stack[:-1] = self.state_stack[1:]
            self.state_stack[-1] = scan

        # reshape
        laser = self.state_stack.reshape(-1)
        track = self.pos_track.reshape(-1)

        state = laser
        state = np.append(state, track)

        return state

    def close(self):
        self.pause_gym(False)
        rospy.signal_shutdown('WTF')
