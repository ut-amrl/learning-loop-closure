import rosbag
import rospy
import numpy as np
from scipy import spatial
import torch
import argparse

from sensor_msgs.msg import PointCloud2
from evaluation_helpers import embedding_for_scan, visualize_location, visualize_cloud, draw_map

import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
from helpers import create_classifier, create_embedder, get_predictions_for_model, create_structured_embedder
from data_processing.dataset import LCDataset, LCStructuredDataset

parser = argparse.ArgumentParser(
    description='Visualize the models classification for given inputs')
parser.add_argument('--dataset', type=str, help='the dataset from which to pull the triplet')
parser.add_argument('--triplets', type=str, help='triplets file to visualize')
parser.add_argument('--model', type=str,
                    help='state dict of an already trained LC model to use')
parser.add_argument('--embedding_model', type=str,
                    help='state dict of an already trained LC model to use')
parser.add_argument('--structured_model', type=str,
                    help='state dict of an already trained LC model to use')
parser.add_argument('--map_name', type=str, help='name of map to render', default='GDC3')
parser.add_argument('--only_error', type=bool, help='If True, only show triplets where the model failed', default=False)
parser.add_argument('--threshold', type=float)
args = parser.parse_args()

triplets = np.load(args.triplets)

if args.structured_model:
    dataset = LCStructuredDataset(args.dataset)
else: 
    dataset = LCDataset(args.dataset)

if args.model:
    model = create_classifier('', args.model)
    model.eval()
elif args.embedding_model:
    model = create_embedder(args.embedding_model)
    model.eval()
elif args.structured_model:
    model = create_structured_embedder(args.structured_model)
    model.eval()
else:
    model = None

# anchor_pub = rospy.Publisher('anchor', PointCloud2, queue_size=10)
# similar_pub = rospy.Publisher('similar', PointCloud2, queue_size=10)
# distant_pub = rospy.Publisher('distant', PointCloud2, queue_size=10)
# rospy.init_node('visualizer', anonymous=True)

def visualize_segmented_cloud(plt, partition_array):
    import random
    for partition in partition_array:
        if len(partition.nonzero()[0]):
            color = (random.random(), random.random(), random.random())
            center = np.repeat(partition[-1:], partition.shape[0] - 1, 0)
            plt.scatter(partition[:-1, 0] + center[:, 0], partition[:-1, 1] + center[:, 1], color=color)

for i in range(triplets.shape[0]):
    # get first triplet
    batch_triplets = triplets[i]
    for j in range(triplets.shape[1]):
        # get first in batch
        triplet = batch_triplets[j]

        anchor_np, anchor_loc, ts = dataset.get_by_timestamp(triplet[0, 0], include_angle=True)
        similar_np, similar_loc, _ = dataset.get_by_timestamp(triplet[1, 0], include_angle=True)
        distant_np, distant_loc, _ = dataset.get_by_timestamp(triplet[2, 0], include_angle=True)
        print("Anchor Timestamp", ts)
        print("Locations", anchor_loc, similar_loc, distant_loc)

        if args.structured_model:
            anchor = (torch.tensor(anchor_np[0]).unsqueeze(0).cuda(), torch.tensor(anchor_np[1]).unsqueeze(0).cuda())
            similar = (torch.tensor(similar_np[0]).unsqueeze(0).cuda(), torch.tensor(similar_np[1]).unsqueeze(0).cuda())
            distant = (torch.tensor(distant_np[0]).unsqueeze(0).cuda(), torch.tensor(distant_np[1]).unsqueeze(0).cuda())
        else:
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
        visualize_segmented_cloud(plt, similar_np[0])
        plt.subplot(132)
        visualize_segmented_cloud(plt, anchor_np[0])
        plt.subplot(133)
        visualize_segmented_cloud(plt, distant_np[0])
        plt.gca().set_aspect('equal', adjustable='box')

        plt.figure(2)
        visualize_location(plt, anchor_loc, 'blue')
        visualize_location(plt, similar_loc, 'green')
        visualize_location(plt, distant_loc, 'red')

        draw_map(plt, '../../../cobot/maps/{0}/{0}_vector.txt'.format(args.map_name))

        plt.show()
