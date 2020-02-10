import rosbag
import rospy
import numpy as np
from scipy import spatial
import torch
import argparse
from learning.dataset import LCDataset
from helpers import visualize_cloud

TIMESTEP = 1.5

parser = argparse.ArgumentParser(
    description='Visualize the models classification for given inputs')
parser.add_argument('--dataset', type=str, help='the dataset from which to pull the triplet')
parser.add_argument('--timestamps', type=str, help='timestamps file to visualize')
parser.add_argument('--pos_only', type=str, help='only visualize positive timestamps')
args = parser.parse_args()

timestamps = np.load(args.timestamps)
dataset = LCDataset(args.dataset)

import matplotlib.pyplot as plt

for i in range(len(timestamps)):
    timestamp = timestamps[i][0]
    label = timestamps[i][1]

    cloud, loc, ts = dataset.get_by_timestamp(timestamp)
    plt.figure(1)
    if label:
        visualize_cloud(plt, cloud, color='green')
        plt.show()
    elif not args.pos_only:
        visualize_cloud(plt, cloud, color='red')
        plt.show()