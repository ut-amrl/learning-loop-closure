PROJECT(lidar_pointcloud)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

MESSAGE(STATUS "Compiling Lidar To PointCloud Implementation")
MESSAGE(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")
MESSAGE(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")

SET(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-error=deprecated-copy -Werror")

IF(${CMAKE_BUILD_TYPE} MATCHES "Release")
    MESSAGE(STATUS "Additional Flags for Release mode")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -O3 -DNDEBUG")
ELSEIF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    MESSAGE(STATUS "Additional Flags for Debug mode")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")
ENDIF()

INCLUDE($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

SET(ROS_BUILD_STATIC_LIBS true)
SET(ROS_BUILD_SHARED_LIBS false)

MESSAGE(STATUS "ROS-Overrride Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

rosbuild_genmsg()

SET(libs roslib roscpp rosbag console_bridge glog gflags)

ROSBUILD_ADD_EXECUTABLE(lidar_pointcloud
        src/main.cc
        src/convert_pointcloud.cc
        src/kdtree.cc
        src/pointcloud_helpers.cc)

TARGET_LINK_LIBRARIES(lidar_pointcloud
        ${libs})

