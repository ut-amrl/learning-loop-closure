import rosbag
import rospy
import numpy as np
from scipy import spatial
import torch
import argparse
from learning.train_helpers import create_classifier, create_embedder, get_predictions_for_model
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
parser.add_argument('--embedding_model', type=str,
                    help='state dict of an already trained LC model to use')



args = parser.parse_args()

triplets = np.load(args.triplets)
dataset = LCDataset(args.dataset)

if args.model:
    model = create_classifier('', args.model)
elif args.embedding_model:
    model = create_embedder(args.embedding_model)

model.eval()

for i in range(triplets.shape[0]):
    # get first triplet
    batch_triplets = triplets[i]
    for j in range(triplets.shape[1]):
        # get first in batch
        triplet = batch_triplets[j]

        anchor_np, _, _ = dataset.get_by_timestamp(triplet[0, 0])
        similar_np, _, _ = dataset.get_by_timestamp(triplet[1, 0])
        distant_np, _, _ = dataset.get_by_timestamp(triplet[2, 0])

        anchor = torch.tensor(anchor_np.transpose(1, 0)).unsqueeze(0).cuda()
        similar = torch.tensor(similar_np.transpose(1, 0)).unsqueeze(0).cuda()
        distant = torch.tensor(distant_np.transpose(1, 0)).unsqueeze(0).cuda()

        predictions = get_predictions_for_model(model, anchor, similar, distant)
        print("Predictions: Similar {0}, Distant {1}".format(predictions[0], predictions[1]))

        # TODO do away with matplotlib
        import matplotlib.pyplot as plt

        plt.scatter(anchor_np[:, 0], anchor_np[:, 1], c='blue', marker='.')
        plt.scatter(similar_np[:, 0], similar_np[:, 1], c='green', marker='.')
        plt.scatter(distant_np[:, 0], distant_np[:, 1], c='red', marker='.')
        plt.show()

        # point_pub = rospy.Publisher('triplets', PointCloud2, queue_size=10)
        # rospy.init_node('visualizer', anonymous=True)
        # msg = create_ros_pointcloud()
        # publish_ros_pointcloud(point_pub, msg, anchor)
