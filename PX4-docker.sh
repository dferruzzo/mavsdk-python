#!/bin/bash

# Run the script with the following command
# ./PX4-docker.sh "make px4_sitl gazebo-classic" 

# enable access to xhost from the container
xhost +

PX4_DIR=/home/diego/PX4-Autopilot # Change this to your PX4 directory

# Run docker
docker run -it --rm --privileged \
    -w /PX4-Autopilot \
    --env=LOCAL_USER_ID="$(id -u)" \
    -v ${PX4_DIR}:/PX4-Autopilot:rw \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -e DISPLAY=:0 \
    -p 14570:14570/udp \
    --network host \
    --name=PX4  px4io/px4-dev-ros-noetic:latest /bin/bash -c "$1 $2 $3"
