import rospy
import argparse
import numpy as np
import random
import torch
import os
from learning.train_helpers import create_embedder
from helpers import scan_to_point_cloud, LCBagDataReader, embedding_for_scan
import json

parser = argparse.ArgumentParser(
    description='Find loop closure locations for some ROS bag')
parser.add_argument('--recall_dataset', type=str,
                    help='path to the bag file containing recakk observation data')
parser.add_argument('--model', type=str,
                    help='state dict of an already trained LC model to use')
args = parser.parse_args()

model = create_embedder(args.model)
model.eval()

ds_file = os.path.join(args.recall_dataset, 'datasets.json')
with open(ds_file) as f:
    dataset_names = json.loads(f.read())

def load_cloud(ds, timestamp):
    dataset_name = dataset_names[ds]
    cloud_data = os.path.join(dataset_name, 'point_{0}.npy'.format(timestamp))
    return np.load(cloud_data)

# recall location x dataset x timestamp
obs_timestamps = np.load(os.path.join(args.recall_dataset, 'observation_timestamps.npy'))

# We want to assert that for every recall location, each scan in dataset 0 matches each scan in dataset 1
for loc_idx in range(obs_timestamps.shape[0]):
    first_dataset_timestamps = obs_timestamps[loc_idx, 0]
    second_dataset_timestamps = obs_timestamps[loc_idx, 1]

    for ds1_timestamp in first_dataset_timestamps:
        for ds2_timestamp in second_dataset_timestamps:
            first_cloud = load_cloud(0, ds1_timestamp[0])
            second_cloud = load_cloud(1, ds2_timestamp[0])

            emb1, _, _, _ = embedding_for_scan(model, first_cloud)
            emb2, _, _, _ = embedding_for_scan(model, second_cloud)
            
            distance = torch.norm(emb1 - emb2, p=2, dim=1)

            print('distance', distance)