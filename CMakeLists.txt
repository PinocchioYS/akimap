cmake_minimum_required(VERSION 3.0.2)
project(akimap)


SET(CMAKE_BUILD_TYPE Release)
# SET(CMAKE_BUILD_TYPE Debug)
ADD_COMPILE_OPTIONS(-std=c++11)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    ADD_COMPILE_OPTIONS(-O3)
endif()


FIND_PACKAGE(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    pcl_ros
)

FIND_PACKAGE(Eigen3 REQUIRED)

INCLUDE(FindOpenMP)
IF (OPENMP_FOUND)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()
add_definitions(-DOPENMP)


CATKIN_PACKAGE(
    INCLUDE_DIRS include
)

INCLUDE_DIRECTORIES(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

ADD_EXECUTABLE(akimap_example
    src/akimap_example.cpp

    src/akimap/akimap.cpp
    src/utils/RTree.cpp

    include/akimap/akimap.h
    include/akimap/akimap_node.h
    include/akimap/akimap_key.h

    include/utils/RTree.h
)
TARGET_LINK_LIBRARIES(akimap_example
    ${catkin_LIBRARIES}
)