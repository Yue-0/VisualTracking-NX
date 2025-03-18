cmake_minimum_required(VERSION 3.0.2)
project(client)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
    rospy
    roscpp
    std_msgs
    sensor_msgs
)

# System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)

###################################
## catkin specific configuration ##
###################################

catkin_package()

###########
## Build ##
###########

# Specify additional locations of header files
include_directories(
    ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

# Mark executable scripts (Python etc.) for installation
catkin_install_python(PROGRAMS
    scripts/window.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
