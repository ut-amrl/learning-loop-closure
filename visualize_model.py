import rosbag
import rospy
import numpy as np
from scipy import spatial
import torch
import argparse
from learning.model import PointNetLC
from sensor_msgs.msg import PointCloud2
from helpers import get_scans_and_localizations_from_bag, embedding_for_scan, create_ros_pointcloud, publish_ros_pointcloud

TIMESTEP = 1.5

parser = argparse.ArgumentParser(
    description='Visualize key points from model')
parser.add_argument('--bag_file', type=str,
                    help='path to the bag file containing training data')
parser.add_argument('--lidar_topic', type=str,
                    help='name of topic containing lidar information')
parser.add_argument('--localization_topic', type=str,
                    help='name of topic containing localization information')
parser.add_argument('--model', type=str,
                    help='state dict of an already trained LC model to use')
parser.add_argument('--threshold', type=int,
                    help='the distance between embeddings to consider a key frame')

args = parser.parse_args()
bag = rosbag.Bag(args.bag_file)

scans = {}
localizations = {}
print("Loading scans & Localization from Bag file")
print("Bag has ", bag.get_message_count(topic_filters=[args.localization_topic]), " Localization messages")
print("Start time:", bag.get_start_time())
print("End time:", bag.get_end_time())
scans, localizations, _ = get_scans_and_localizations_from_bag(bag, args.lidar_topic, args.localization_topic, TIMESTEP)
print("Finished processing Bag file", len(scans.keys()), "scans", len(localizations.keys()), "localizations")

localization_timestamps = sorted(localizations.keys())
localizationTimeTree = spatial.KDTree([[t] for t in localization_timestamps])

point_pub = rospy.Publisher('points', PointCloud2, queue_size=10)
skipped_point_pub = rospy.Publisher('skipped_points', PointCloud2, queue_size=10)
rospy.init_node('visualizer', anonymous=True)

with torch.no_grad():
    print("Loading embedding model...")
    model = PointNetLC()
    model.load_state_dict(torch.load(args.model))
    model.eval()
    model.cuda()
    print("Finished loading embedding model")

    last_embedding = torch.tensor([np.zeros((256, 1))]).cuda()
    last_location = torch.tensor([np.zeros((2, 1))])
    for timestamp, cloud in scans.items():
        _, loc_idx = localizationTimeTree.query([timestamp])
        location = torch.tensor(localizations[localization_timestamps[loc_idx]])
        embedding, _, _ = embedding_for_scan(model, cloud)
        distance = torch.norm(embedding - last_embedding).item()
        location_distance = torch.norm(location[:2] - last_location[:2]).item()
        print("Embedding Dist: {0}; Real World Distance: {1}".format(round(distance, 3), round(location_distance, 3)))

        # Transform the cloud based on its location
        augmented_cloud = np.zeros(cloud.shape).astype(np.float32)
        loc = location.numpy()
        cos, sin = np.cos(loc[2]), np.sin(loc[2])
        rotation = np.array([(cos, -sin), (sin, cos)])

        for i in range(cloud.shape[0]):
            augmented_cloud[i][:2] = np.matmul(rotation, cloud[i][:2]) + [loc[0], loc[1]]
        
        msg = create_ros_pointcloud()
        if (distance > args.threshold):
            last_embedding = embedding
            last_location = location
            publish_ros_pointcloud(point_pub, msg, augmented_cloud)
        else:
            publish_ros_pointcloud(skipped_point_pub, msg, augmented_cloud)
