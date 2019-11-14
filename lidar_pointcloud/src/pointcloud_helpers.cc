//
// Created by jack on 9/15/19.
//

#include "./pointcloud_helpers.h"

#include "ros/package.h"
#include "eigen3/Eigen/Dense"
#include "sensor_msgs/PointCloud2.h"

using Eigen::Vector2f;
using Eigen::Matrix2f;
using Eigen::Rotation2D;

std::vector<Vector2f>
pointcloud_helpers::LaserScanToPointCloud(sensor_msgs::LaserScan &laser_scan) {
  std::vector<Vector2f> pointcloud;
  float angle_offset = 0.0f;
  float range_max = fmin(laser_scan.range_max, 30);
  for (float range : laser_scan.ranges) {
    if (range >= laser_scan.range_min && range <= range_max) {
      // Only accept valid ranges.
      // Then we must rotate the point by the specified angle at that distance.
      Vector2f point(range, 0.0);
      Matrix2f rot_matrix =
              Rotation2D<float>(laser_scan.angle_min + angle_offset)
                      .toRotationMatrix();
      point = rot_matrix * point;
      pointcloud.push_back(point);
    }
    angle_offset += laser_scan.angle_increment;
  }
  return pointcloud;
}
