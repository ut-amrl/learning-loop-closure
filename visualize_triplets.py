import rosbag
import rospy
import numpy as np
from scipy import spatial
import torch
import argparse
from learning.train_helpers import create_classifier, create_embedder, get_predictions_for_model
from learning.dataset import LCDataset
from sensor_msgs.msg import PointCloud2
from helpers import get_scans_and_localizations_from_bag, embedding_for_scan, create_ros_pointcloud, publish_ros_pointcloud, visualize_location

TIMESTEP = 1.5

parser = argparse.ArgumentParser(
    description='Visualize the models classification for given inputs')
parser.add_argument('--dataset', type=str, help='the dataset from which to pull the triplet')
parser.add_argument('--triplets', type=str, help='triplets file to visualize')
parser.add_argument('--model', type=str,
                    help='state dict of an already trained LC model to use')
parser.add_argument('--embedding_model', type=str,
                    help='state dict of an already trained LC model to use')
parser.add_argument('--only_error', type=bool, help='If True, only show triplets where the model failed', default=False)
parser.add_argument('--threshold', type=int)
args = parser.parse_args()

triplets = np.load(args.triplets)
dataset = LCDataset(args.dataset)

if args.model:
    model = create_classifier('', args.model)
    model.eval()
elif args.embedding_model:
    model = create_embedder(args.embedding_model)
    model.eval()
else:
    model = None

# anchor_pub = rospy.Publisher('anchor', PointCloud2, queue_size=10)
# similar_pub = rospy.Publisher('similar', PointCloud2, queue_size=10)
# distant_pub = rospy.Publisher('distant', PointCloud2, queue_size=10)
# rospy.init_node('visualizer', anonymous=True)

for i in range(triplets.shape[0]):
    # get first triplet
    batch_triplets = triplets[i]
    for j in range(triplets.shape[1]):
        # get first in batch
        triplet = batch_triplets[j]

        anchor_np, anchor_loc, _ = dataset.get_by_timestamp(triplet[0, 0], include_angle=True)
        similar_np, similar_loc, _ = dataset.get_by_timestamp(triplet[1, 0], include_angle=True)
        distant_np, distant_loc, _ = dataset.get_by_timestamp(triplet[2, 0], include_angle=True)

        print("Locations", anchor_loc, similar_loc, distant_loc)

        anchor = torch.tensor(anchor_np.transpose(1, 0)).unsqueeze(0).cuda()
        similar = torch.tensor(similar_np.transpose(1, 0)).unsqueeze(0).cuda()
        distant = torch.tensor(distant_np.transpose(1, 0)).unsqueeze(0).cuda()

        if model:
            predictions = get_predictions_for_model(model, anchor, similar, distant, args.threshold)
            print("Predictions: Similar {0}, Distant {1}".format(predictions[0], predictions[1]))
            if args.only_error and predictions[0] == True and predictions[1] == False:
                continue

        # TODO do away with matplotlib
        import matplotlib.pyplot as plt
        plt.figure(1, figsize=(9, 3))
        plt.subplot(131)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.scatter(similar_np[:, 0], similar_np[:, 1], c='green', marker='.')
        plt.subplot(132)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.scatter(anchor_np[:, 0], anchor_np[:, 1], c='blue', marker='.')
        plt.subplot(133)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.scatter(distant_np[:, 0], distant_np[:, 1], c='red', marker='.')

        plt.figure(2)
        visualize_location(anchor_loc, 'blue')
        visualize_location(similar_loc, 'green')
        visualize_location(distant_loc, 'red')
        plt.gca().set_aspect('equal')
        plt.gca().autoscale()
        plt.show()

        # msg = create_ros_pointcloud()
        # publish_ros_pointcloud(anchor_pub, msg, anchor)
        # publish_ros_pointcloud(similar_pub, msg, similar)
        # publish_ros_pointcloud(distant_pub, msg, distant)
