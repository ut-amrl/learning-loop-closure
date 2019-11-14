#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/Odometry.h"
#include "eigen3/Eigen/Geometry"

#include "convert_pointcloud.h"
#include "./pointcloud_helpers.h"

using Eigen::Vector2f;
using std::string;

struct PointCloud {
  std::vector<Vector2f> points;
  float timestamp;

  PointCloud(std::vector<Vector2f> p, float t) : points(p), timestamp(t) {};
};

struct LocalizationInformation {
  Vector2f location;
  float timestamp;
  LocalizationInformation(Vector2f l, float t) : location(l), timestamp(t) {};
};

/*
  Looks at a ros bag, and inspects the lidar and localization topics to construct training data.
*/
void ExtractDataFromBagFile(const char* bag_path, const char* lidar_topic, const char* localization_topic, string dataset_name) {
  printf("Loading bag file... ");
  fflush(stdout);
  rosbag::Bag bag;
  try {
    bag.open(bag_path, rosbag::bagmode::Read);
  } catch (rosbag::BagException& exception) {
    printf("Unable to read %s, reason %s:", bag_path, exception.what());
  }

  // Set up the directory structure
  string dataLocation = "../" + dataset_name + "/";


  // Get the topics we want
  std::vector<string> topics;
  topics.emplace_back(lidar_topic);
  topics.emplace_back(localization_topic);
  rosbag::View view(bag, rosbag::TopicQuery(topics));

  std::vector<PointCloud> pointClouds;
  std::vector<LocalizationInformation> locations;

  // Iterate through the bag
  for (rosbag::View::iterator it = view.begin();
       ros::ok() && it != view.end();
       ++it) {
    
    const rosbag::MessageInstance &message = *it;
    {
      sensor_msgs::LaserScanPtr laser_scan =
              message.instantiate<sensor_msgs::LaserScan>();
      if (laser_scan != nullptr) {
        std::vector<Vector2f> points = pointcloud_helpers::LaserScanToPointCloud(*laser_scan);
        pointClouds.push_back(PointCloud(points, message.getTime().toSec()));
      }

      // store the localization timestamps & guessed locations as well (How do I do this? ROS struggles...)
    }
  }



  /* The next step is to add all the LocalizationInformations to a 1-dimensional KDTree, for fast lookup.
    Then, we loop over all the PointCloud structs. For Each One:
    -- find the nearest Localization in time-space (using the KD-Tree above)
    -- write out location and point cloud information.
      Either as 2 files (e.g. points_$timestamp$.data and points_$timestamp$.location)
      Or as one file with some consistent pattern we can consume in python land.
  */


    // std::ofstream pointFile; 
    // string pointFileName = "points_" + std::to_string(i) + ".data";
    // pointFile.open(dataLocation + pointFileName); 
    // pointFile << std::to_string(v[0]) + " " + std::to_string(v[1]) + " 1\n";
    // pointFile.close();
}