// TODO: Figure out how to share this code, so we don't need a second copy of this here...
#include "./pointcloud_helpers.h"

#include <csignal>
#include <string>
#include "ros/node_handle.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/Odometry.h"
#include "./convert_pointcloud.h"

using std::string;
using std::vector;

DEFINE_string(
  bag_path,
  "",
  "The location of the bag file to convert to pointcloud data.");
DEFINE_string(
  lidar_topic,
  "/Cobot/Laser",
  "The topic that lidar messages are published over.");
DEFINE_string(
  localization_topic,
  "/Cobot/Localization",
  "The topic that localization messages are published over.");


void SignalHandler(int signum) {
  printf("Exiting with %d\n", signum);
  exit(0);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(*argv);
  google::ParseCommandLineFlags(&argc, &argv, false);
  if (FLAGS_bag_path == "") {
    printf("Must specify an input bag!");
    exit(0);
  }
  ros::init(argc, argv, "lidar_pointcloud");
  ros::NodeHandle n;
  signal(SIGINT, SignalHandler);

  string dataset_name;
  std::cout << "Dataset name:";
  std::cin >> dataset_name;

  ExtractDataFromBagFile(FLAGS_bag_path.c_str(), FLAGS_lidar_topic.c_str(), FLAGS_localization_topic.c_str(), dataset_name);

  return 0;
}