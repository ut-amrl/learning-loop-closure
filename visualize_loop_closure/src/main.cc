#include <csignal>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "ros/node_handle.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/PointCloud2.h"

#include "./slam_type_builder.h"
#include "./slam_types.h"
#include "lidar_slam/CobotOdometryMsg.h"
#include "pointcloud_helpers.h"

using std::string;
using std::vector;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using slam_types::SLAMNode2D;
using lidar_slam::CobotOdometryMsg;
using std::ofstream;
using std::ifstream;

DEFINE_string(
  bag_path,
  "",
  "The location of the bag file to run SLAM on.");
DEFINE_string(
  odom_topic,
  "/odometry/filtered",
  "The topic that odometry messagse are published over.");
DEFINE_string(
  lidar_topic,
  "/velodyne_2dscan_high_beams",
  "The topic that lidar messages are published over.");
DEFINE_int64(
  pose_num,
  30,
  "The number of poses to process.");
DEFINE_bool(
  diff_odom,
  false,
  "Is the odometry differential (True for CobotOdometryMsgs)?");
DEFINE_double(
  embedding_distance,
  10,
  "The threshold to judge if two embeddings are different or not.");
DEFINE_string(
  model,
  "",
  "The path to the model.");

SLAMProblem2D ProcessBagFile(const char* bag_path,
                             const ros::NodeHandle& n) {
  /*
   * Loads and processes the bag file pulling out the lidar data
   * and the odometry data. Keeps track of the current pose and produces
   * a list of poses / pointclouds. Also keeps a list of the odometry data.
   */
  printf("Loading bag file... ");
  fflush(stdout);
  rosbag::Bag bag;
  try {
    bag.open(bag_path, rosbag::bagmode::Read);
  } catch (rosbag::BagException& exception) {
    printf("Unable to read %s, reason %s:", bag_path, exception.what());
    return slam_types::SLAMProblem2D();
  }
  // Get the topics we want
  vector<string> topics;
  topics.emplace_back(FLAGS_odom_topic.c_str());
  topics.emplace_back(FLAGS_lidar_topic.c_str());
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  SLAMTypeBuilder slam_builder(FLAGS_pose_num, FLAGS_diff_odom);
  // Iterate through the bag
  for (rosbag::View::iterator it = view.begin();
       ros::ok() && it != view.end() && !slam_builder.Done();
       ++it) {
    const rosbag::MessageInstance &message = *it;
    {
      // Load all the point clouds into memory.
      sensor_msgs::LaserScanPtr laser_scan =
              message.instantiate<sensor_msgs::LaserScan>();
      if (laser_scan != nullptr) {
        slam_builder.LidarCallback(*laser_scan);
      }
    }
    {
      nav_msgs::OdometryPtr odom = message.instantiate<nav_msgs::Odometry>();
      if (odom != nullptr) {
        slam_builder.OdometryCallback(*odom);
      }
    }
    {
      lidar_slam::CobotOdometryMsgPtr odom = message.instantiate<CobotOdometryMsg>();
      if (odom != nullptr) {
        slam_builder.OdometryCallback(*odom);
      }
    }
  }
  bag.close();
  printf("Done.\n");
  fflush(stdout);
  return slam_builder.GetSlamProblem();
}

#define TEMP_ASCII_FILENAME "temp_pc_file.txt"
#define TEMP_OUTPUT_FILENAME "output.txt"
#define START_IDX 15

void WritePointCloudToAscii(vector<Vector2f>& pointcloud) {
  remove(TEMP_ASCII_FILENAME); // Delete the last file.
  ofstream pc_file;
  pc_file.open(TEMP_ASCII_FILENAME);
  for (Vector2f point : pointcloud) {
    pc_file << point.x() << " " << point.y() << " 0.0" << std::endl;
  }
  pc_file.close();
}

Eigen::MatrixXd LoadEmbedding() {
  ifstream embed_file;
  embed_file.open(TEMP_OUTPUT_FILENAME);
  Eigen::MatrixXd m(1024, 1);
  for (size_t i = 0; i < 1024; i++) {
    double val;
    embed_file >> val;
    m(i, 0) = val;
  }
  embed_file.close();
  return m;
}

void WriteEmbeddingToFile(vector<Vector2f>& pointcloud, string fileName=TEMP_OUTPUT_FILENAME) {
  WritePointCloudToAscii(pointcloud);
  remove(fileName.c_str());
  std::stringstream command;
  command << "python3 ../learning/embed.py --model ";
  command << FLAGS_model << " ";
  command << "--data_path " << TEMP_ASCII_FILENAME << " ";
  command << "--out_path " << fileName;
  int ret_val = system(command.str().c_str());
  (void) ret_val; // TODO: For now we are just gonna ignore this.
}

Eigen::MatrixXd GetEmbedding(vector<Vector2f>& pointcloud) {
  WriteEmbeddingToFile(pointcloud);
  return LoadEmbedding();
}

vector<Eigen::Vector2f> AugmentPoints(vector<Eigen::Vector2f>& pointcloud, RobotPose2D& pose) {
  vector<Eigen::Vector2f> augmented_point_cloud;
  Eigen::Affine2f point_to_world = pose.RobotToWorldTf();
  for (Eigen::Vector2f point : pointcloud) {
    Eigen::Vector2f aug_point = point_to_world * point;
    augmented_point_cloud.push_back(aug_point);
  }
  return augmented_point_cloud;
}

void VisualizePointClouds(SLAMProblem2D problem, ros::NodeHandle& n) {
  ros::Publisher pointcloud_pub =
    n.advertise<sensor_msgs::PointCloud2>("/points", 10);
  PointCloud2 vis_points_marker;
  vector<Eigen::Vector2f> vis_points;
  pointcloud_helpers::InitPointcloud(&vis_points_marker);
  CHECK_GT(problem.nodes.size(), START_IDX);
  Eigen::MatrixXd last_vis_embedding =
    GetEmbedding(problem.nodes[START_IDX].lidar_factor.pointcloud);
  int last_embedding_idx = START_IDX;
  vector<Eigen::Vector2f> first_scan =
          AugmentPoints(problem.nodes[START_IDX].lidar_factor.pointcloud,
                        problem.nodes[START_IDX].pose);
  vis_points.insert(vis_points.end(),
                    first_scan.begin(),
                    first_scan.end());
  for (unsigned int i = START_IDX; i < problem.nodes.size(); i++) {
    SLAMNode2D node = problem.nodes[i];
    Eigen::MatrixXd current_embedding =
      GetEmbedding(node.lidar_factor.pointcloud);
    double norm = (last_vis_embedding - current_embedding).norm();
    if (isinf(norm)) {
      std::cout << "Found infinite difference" << std::endl;
      WriteEmbeddingToFile(problem.nodes[last_embedding_idx].lidar_factor.pointcloud, "inf_" + std::to_string(last_embedding_idx) + ".txt");
      WriteEmbeddingToFile(node.lidar_factor.pointcloud, "inf_" + std::to_string(i) + ".txt");
    }
    
    std::cout << "Current difference: " << norm;
    if (norm > FLAGS_embedding_distance || i == START_IDX) {
      last_vis_embedding = current_embedding;
      last_embedding_idx = i;
      vector<Eigen::Vector2f> augmented_points =
        AugmentPoints(node.lidar_factor.pointcloud, node.pose);
      vis_points.insert(vis_points.end(),
                        augmented_points.begin(),
                        augmented_points.end());
      pointcloud_helpers::PublishPointcloud(vis_points, vis_points_marker, pointcloud_pub);
      std::cout << " CHOSEN";
    }
    std::cout << std::endl;
  }
  pointcloud_helpers::PublishPointcloud(vis_points, vis_points_marker, pointcloud_pub);
}

void SignalHandler(int signum) {
  printf("Exiting with %d\n", signum);
  exit(0);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(*argv);
  google::ParseCommandLineFlags(&argc, &argv, false);
  if (FLAGS_bag_path.compare("") == 0) {
    printf("Must specify an input bag!");
    exit(0);
  }
  if (FLAGS_model.compare("") == 0) {
    printf("Must specify a model");
    exit(0);
  }
  ros::init(argc, argv, "lidar_slam");
  ros::NodeHandle n;
  signal(SIGINT, SignalHandler);
  // Load and pre-process the data.
  SLAMProblem2D slam_problem =
          ProcessBagFile(FLAGS_bag_path.c_str(), n);
  CHECK_GT(slam_problem.nodes.size(), 1)
    << "Not enough nodes were processed"
    << "you probably didn't specify the correct topics!\n";
  VisualizePointClouds(slam_problem, n);
  return 0;
}
