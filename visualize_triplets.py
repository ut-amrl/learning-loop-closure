import rosbag
import rospy
import numpy as np
from scipy import spatial
import torch
import argparse
from learning.model import EmbeddingNet
from learning.dataset import LCDataset
from sensor_msgs.msg import PointCloud2
from helpers import get_scans_and_localizations_from_bag, embedding_for_scan, create_ros_pointcloud, publish_ros_pointcloud

TIMESTEP = 1.5

parser = argparse.ArgumentParser(
    description='Visualize the models classification for given inputs')
parser.add_argument('--dataset', type=str, help='the dataset from which to pull the triplet')
parser.add_argument('--triplets', type=str, help='triplets file to visualize')
parser.add_argument('--model', type=str,
                    help='state dict of an already trained LC model to use')

args = parser.parse_args()

triplets = np.load(args.triplets)
dataset = LCDataset(args.dataset)

# get first triplet
batch_triplets = triplets[0]

# get first in batch
triplet = batch_triplets[0]

anchor, _, _ = dataset.get_by_timestamp(triplet[0, 0])
similar, _, _ = dataset.get_by_timestamp(triplet[1, 0])
distant, _, _ = dataset.get_by_timestamp(triplet[2, 0])
import pdb; pdb.set_trace()

point_pub = rospy.Publisher('triplets', PointCloud2, queue_size=10)
rospy.init_node('visualizer', anonymous=True)
msg = create_ros_pointcloud()
publish_ros_pointcloud(point_pub, msg, anchor)
