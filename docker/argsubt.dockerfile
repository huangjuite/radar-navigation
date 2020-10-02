FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO melodic

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
    && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \
    locales \
    cmake \
    git \
    vim \
    gedit \
    lsb-release \
    wget \
    sudo \
    build-essential \
    dirmngr \
    gnupg2 \
    mercurial \
    python-gtk2 \
    python-gobject \
    python-tk \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    python-pip \
    python-setuptools \
    python3-pip \
    python3-setuptools \
    python3-numpy \
    python3-empy  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV SHELL=/bin/bash \
    NB_USER=argsubt \
    NB_UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} 

RUN echo "root:root" | chpasswd
RUN echo "${NB_USER}:111111" | chpasswd

###################################### ROS #####################################

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list \
    && echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list

# setup keys
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
    && wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add -


# install bootstrap tools
RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -o Acquire::ForceIPv4=true install --no-install-recommends -y \
    ros-$ROS_DISTRO-desktop-full \
    ros-$ROS_DISTRO-joy \
    ros-$ROS_DISTRO-joystick-drivers \
    ros-$ROS_DISTRO-pointcloud-to-laserscan \
    ros-$ROS_DISTRO-robot-localization \
    ros-$ROS_DISTRO-spacenav-node \
    ros-$ROS_DISTRO-tf2-sensor-msgs \
    ros-$ROS_DISTRO-twist-mux \
    ros-$ROS_DISTRO-velodyne-simulator \
    ros-$ROS_DISTRO-serial \
    ros-$ROS_DISTRO-soem \
    ros-$ROS_DISTRO-openslam-gmapping \
    ros-$ROS_DISTRO-geodesy \
    ros-$ROS_DISTRO-cartographer-* \
    ros-$ROS_DISTRO-ddynamic-reconfigure \
    ros-$ROS_DISTRO-ros-numpy \
    python-rosdep \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    && rm -rf /var/lib/apt/lists/*

####################################### procman ###########################################

RUN cd ${HOME} && git clone https://github.com/lcm-proj/lcm \
    && cd lcm \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install

RUN cd ${HOME} && git clone http://github.com/ARG-NCTU/procman \
    && cd procman \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install

################################### SOURCE #####################################
ARG username
ARG password

RUN git clone https://${username}:${password}@github.com/ARG-NCTU/subt-gazebo /subt_ws \
    && cd /subt_ws \
    && mkdir -p ${HOME}/catkin_ws/src \
    && apt-get -o Acquire::ForceIPv4=true update \
    && cp -R /subt_ws ${HOME}/catkin_ws/src/. \
    && cd ${HOME}/catkin_ws \
    && wget https://s3.amazonaws.com/osrf-distributions/subt_robot_examples/releases/subt_robot_examples_latest.tgz \
    && tar xvf subt_robot_examples_latest.tgz \
    && /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && rosdep init" \
    && /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && rosdep update && rosdep install --as-root apt:false --from-paths src --ignore-src -r -y" \
    && rm -rf /var/lib/apt/lists/* \
    && /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make install" \
    && rm -fr /subt_ws 

RUN echo "source ~/catkin_ws/install/setup.bash" >> ${HOME}/.bashrc

##################################### PIP ######################################
RUN pip2 install --upgrade pip setuptools

RUN pip2 install --upgrade \
    h5py \
    tqdm \
    matplotlib \
    open3d 


RUN pip3 install --upgrade pip setuptools

RUN pip3 install --upgrade \
    gym \
    tensorflow \
    matplotlib \
    pandas \
    pypozyx \
    requests \
    jupyter \
    ipykernel \
    rospkg \
    catkin-tools \
    scikit-image \
    scikit-learn \
    zerorpc \
    tqdm \
    numpy 

RUN pip3 install \
    torch==1.5.0+cu101 \
    torchvision==0.6.0+cu101 \
    -f https://download.pytorch.org/whl/torch_stable.html


########################################################### extra ###########################################################

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -o Acquire::ForceIPv4=true install --no-install-recommends -y \
    python3-opencv \
    net-tools \
    && rm -rf /var/lib/apt/lists/*


##################################### TAIL #####################################

RUN chown -R ${NB_UID} ${HOME}/
RUN echo "${NB_USER} ALL=(ALL)  ALL" > /etc/sudoers

RUN echo "cd ~/radar-navigation" >> ${HOME}/.bashrc

# Support of nvidia-docker 2.0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

USER ${NB_USER}

WORKDIR ${HOME}
