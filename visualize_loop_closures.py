import rosbag
import rospy
import numpy as np
import json
import argparse
from sensor_msgs.msg import PointCloud2
from helpers import get_scans_and_localizations_from_bag, embedding_for_scan, create_ros_pointcloud, publish_ros_pointcloud
from scipy import spatial

TIMESTEP = 0.1

parser = argparse.ArgumentParser(
    description='Visualize loop closure locations')
parser.add_argument('--loop_closure_file', type=str,
                    help='path to the loop file containing')
parser.add_argument('--bag_file', type=str,
                    help='path to the bag file containing scan data')
parser.add_argument('--lidar_topic', type=str,
                    help='name of topic containing lidar information')
parser.add_argument('--localization_topic', type=str,
                    help='name of topic containing localization information')

args = parser.parse_args()
bag = rosbag.Bag(args.bag_file)

point_pub = rospy.Publisher('points', PointCloud2, queue_size=10)
rospy.init_node('visualizer', anonymous=True)

print("Loading Loop Closure Data")
loop_closure_data = []
with open(args.loop_closure_file) as f:
    for line in f:
        loop_closure_data.append(json.loads(line))

print("Loading Bag File")
scans, localizations, _ = get_scans_and_localizations_from_bag(bag, args.lidar_topic, args.localization_topic, TIMESTEP, TIMESTEP)

localization_timestamps = sorted(localizations.keys())
localizationTimeTree = spatial.KDTree([[t] for t in localization_timestamps])

def augmented_cloud_with_location(cloud, location):
    augmented_cloud = np.zeros(cloud.shape).astype(np.float32)
    cos, sin = np.cos(location[2]), np.sin(location[2])
    rotation = np.array([(cos, -sin), (sin, cos)])

    for i in range(cloud.shape[0]):
        augmented_cloud[i][:2] = np.matmul(rotation, cloud[i][:2]) + [location[0], location[1]]    
    return augmented_cloud

print("Visualizing Data")
for datum in loop_closure_data:
    source_timestamp = datum['source_timestamp']
    target_timestamp = datum['target_timestamp']
    source_cloud = scans[source_timestamp]
    target_cloud = scans[target_timestamp]
    _, source_loc_time_idx = localizationTimeTree.query([source_timestamp])
    _, target_loc_time_idx = localizationTimeTree.query([target_timestamp])
    source_location = localizations[localization_timestamps[source_loc_time_idx]]
    target_location = localizations[localization_timestamps[target_loc_time_idx]]

    # Transform the cloud based on its location
    augmented_source = augmented_cloud_with_location(source_cloud, source_location)
    augmented_target = augmented_cloud_with_location(target_cloud, target_location)
    
    import pdb; pdb.set_trace()
    msg = create_ros_pointcloud()
    publish_ros_pointcloud(point_pub, msg, augmented_source)

    msg = create_ros_pointcloud()
    publish_ros_pointcloud(point_pub, msg, augmented_target)
