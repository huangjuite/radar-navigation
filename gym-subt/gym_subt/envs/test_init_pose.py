import rospy
import time
import numpy as np
import math
import sys
from matplotlib import pyplot as plt
from sensor_msgs.msg import LaserScan, Imu, Joy
from std_srvs.srv import Empty
from std_msgs.msg import Int64
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from scipy.spatial.transform import Rotation as R
import os

full_path = os.path.realpath(__file__)
INITIAL_STATES = np.genfromtxt(os.path.dirname(
    full_path) + '/init_pos.csv', delimiter=',')
print(INITIAL_STATES.shape)


def get_initial_state(name, id):
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

rospy.init_node('test_init')

reset_model = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

for i in range(INITIAL_STATES.shape[0]):
    
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        reset_model(get_initial_state('X1', i))
    except(rospy.ServiceException) as e:
        print(e)
    data = LaserScan()
    try:
        data = rospy.wait_for_message(
            '/RL/scan', LaserScan, timeout=5)
    except:
        print('fail to receive message')

    ranges = np.array(data.ranges)
    minrange = np.min(ranges)
    if minrange < 1:
        print(minrange)
        print(INITIAL_STATES[i])

    rospy.sleep(0.5)
