import rospy
import argparse
import numpy as np
import random
import torch
import os
from learning.train_helpers import create_embedder
from helpers import scan_to_point_cloud, LCBagDataReader, embedding_for_scan

parser = argparse.ArgumentParser(
    description='Find loop closure locations for some ROS bag')
parser.add_argument('--recall_dataset', type=str,
                    help='path to the bag file containing recakk observation data')
parser.add_argument('--model', type=str,
                    help='state dict of an already trained LC model to use')
args = parser.parse_args()

model = create_embedder(args.model)
model.eval()

# recall location x dataset x timestamp
obs_timestamps = np.load(os.path.join(args.recall_dataset, 'observation_timestamps.npy'))


# We want to assert that for every recall location, each scan in dataset 0 matches each scan in dataset 1
for loc_idx in obs_timestamps.shape[0]:
    first_dataset_timestamps = obs_timestamps[loc_idx][0]
    second_dataset_timestamps = obs_timestamps[loc_idx][1]