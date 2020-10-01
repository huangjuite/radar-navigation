#!/usr/bin/env bash
#
# Typical usage: ./join.bash subt
#

if [ ! -z "$1" ]; then
    ROS_MASTER_URI=http://$1:11311
    echo "ROS_MASTER $1"
fi

if [ ! -z "$2" ]; then
    ROS_IP=$2
    echo "ROS_IP $2"
fi

BASH_OPTION=bash
if [ ! -z "$3" ]; then
    if [ $3 = "husky1" ]; then
        BASH_OPTION="bash -c ~/subt-system/scripts/husky1_gui.sh"
    fi
    if [ $3 = "husky2" ]; then
        BASH_OPTION="bash -c ~/subt-system/scripts/husky2_gui.sh"
    fi
    if [ $3 = "df_gui" ]; then
        BASH_OPTION="bash -c ~/subt-system/scripts/df_gui.sh"
    fi
    echo "run $3"
fi

IMG=argnctu/subt:core

xhost +
containerid=$(docker ps -aqf "ancestor=${IMG}") && echo $containerid
docker exec -it \
    --privileged \
    -e ROS_MASTER_URI=$ROS_MASTER_URI \
    -e ROS_IP=$ROS_IP \
    -e DISPLAY=${DISPLAY} \
    -e LINES="$(tput lines)" \
    ${containerid} \
    $BASH_OPTION
xhost -
