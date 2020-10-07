AKIMap - Occupancy Grids based on Adaptive Kernel Inference 
======

This ROS package includes the implementation about the occupancy grids based on adaptive kernel inference and the example of on-the-fly mapping.
AKIMap predicts an occupancy probability of an environment given sparse point clouds. 
You can see technical details of the map on the [project page](http://sglab.kaist.ac.kr/publication/akimap).

BUILD
-----
You can build this package through the following commands (check your directory of catkin workspace).
We implemented and tested this package on ROS Melodic.

    cd ~/catkin_ws/src
    git clone https://github.com/PinocchioYS/akimap.git
    cd ~/catkin_ws/src && catkin_make
 
RUN EXAMPLE
-----
This package includes a bag file in the 'sample' directory for a simple test.

    roslaunch akimap example.launch
    rosbag play $(rospack find akimap)/sample/example.bag