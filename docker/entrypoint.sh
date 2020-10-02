#!/bin/bash
# set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
cd /home/argsubt/radar-navigation

exec "$@"
