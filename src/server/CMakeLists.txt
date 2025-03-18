cmake_minimum_required(VERSION 3.0.2)
project(server)

# Compile as C++11
add_compile_options(-std=c++11)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
    rospy  
    roscpp
    std_msgs
    std_srvs
    tracking_msgs
)

# System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

###################################
## catkin specific configuration ##
###################################

# The catkin_package macro generates cmake config files for package
catkin_package(INCLUDE_DIRS include)

###########
## Build ##
###########

# Specify additional locations of header files
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/cuda/include
    /usr/include/x86_64-linux-gnu/
)
link_directories(/usr/local/cuda/lib64)
link_directories(/usr/lib/x86_64-linux-gnu/)

# Declare a C++ executable
add_executable(server src/server.cpp)

# Specify libraries to link a library or executable target against
target_link_libraries(server
    cudart
    nvinfer
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    libnvinfer_plugin.so
)
add_definitions(-O2 -pthread)

# Mark executable scripts for installation
catkin_install_python(PROGRAMS
    scripts/point2box.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
