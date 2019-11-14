//
// Created by jack on 9/15/19.
//

#ifndef SRC_POINTCLOUD_HELPERS_H_
#define SRC_POINTCLOUD_HELPERS_H_

#include <sensor_msgs/LaserScan.h>
#include "ros/package.h"
#include "ros/ros.h"
#include "eigen3/Eigen/Dense"

using Eigen::Vector2f;

namespace pointcloud_helpers {
  std::vector<Vector2f>
      LaserScanToPointCloud(sensor_msgs::LaserScan &laser_scan);

};
#endif // SRC_POINTCLOUD_HELPERS_H_
