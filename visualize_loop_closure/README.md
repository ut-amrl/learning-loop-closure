# Visualize Loop Closure

This project aims to help visualize if our loop closure training is working.

### Compiling:

Just run:
```make```

Do not run cmake directly.

### How to run:

On linux:
```./bin/vis_loop_closure --bag_path <bags> --diff_threshold 10```

### Arguments:

- ```--odom_topic <name>``` this will look for odometry messages on the ROS topic named ```<name>```.
- ```--lidar_topic <name>``` this will look for lidar messages on the ROS topic named ```<name>```.
- ```--pose_num <num>``` this will set the number of poses to process to ```<num>```.
- ```--diff_threshold``` difference between embeddings to consider pointclouds different.


