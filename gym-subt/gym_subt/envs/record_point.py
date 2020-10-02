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


class Recorder(object):
    def __init__(self):
        self.al_pos = []
        self.get_model = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)
        self.sub_joy = rospy.Subscriber('/joy', Joy, self.cb_joy, queue_size=1)

    def print_al(self):
        print('----------------------')
        for pos in self.al_pos:
            print(pos)

    def cb_joy(self, msg):
        if msg.buttons[0] == 1:
            try:
                agent = self.get_model('X1', '')
                pos = [agent.pose.position.x,
                       agent.pose.position.y,
                       agent.pose.position.z]
                self.al_pos.append(pos)
            except (rospy.ServiceException) as e:
                print(e)
            self.print_al()

        elif msg.axes[6] == 1:
            self.al_pos.pop(-1)
            self.print_al()

        rospy.sleep(0.5)

    def shutdown(self):
        pass


if __name__ == "__main__":
    rospy.init_node('recorder')
    rec = Recorder()
    rospy.on_shutdown(rec.shutdown)
    rospy.spin()
