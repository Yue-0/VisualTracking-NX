cmake_minimum_required(VERSION 3.0.2)
project(tracking_msgs)

# Compile as C++11
add_compile_options(-std=c++11)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  message_generation
)

################################################
## Declare ROS messages, services and actions ##
################################################

# Generate services in the 'srv' folder
add_service_files(
  FILES
  Image.srv
  Point2Box.srv
  ImageWithBoxes.srv
)

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
)

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
