import rosbag
import argparse
import numpy as np
import os
import random
import json
import torch
import time
from learning.model import EmbeddingNet
from learning.dataset import normalize_point_cloud
from helpers import scan_to_point_cloud, get_scans_and_localizations_from_bag
from scipy import spatial

parser = argparse.ArgumentParser(description='Find loop closure locations for some ROS bag')
parser.add_argument('--bag_file', type=str, help='path to the bag file containing training data')
parser.add_argument('--lidar_topic', type=str, help='name of topic containing lidar information')
parser.add_argument('--localization_topic', type=str, help='name of topic containing localization information')
parser.add_argument('--model', type=str, help='state dict of an already trained LC model to use')

args = parser.parse_args()
bag = rosbag.Bag(args.bag_file)

print ("Loading scans & Localization from Bag file")
TIMESTEP = 0.1
scans, localizations, _ = get_scans_and_localizations_from_bag(bag, args.lidar_topic, args.localization_topic, TIMESTEP, TIMESTEP)
print ("Finished processing Bag file", len(scans.keys()), "scans")

localization_timestamps = sorted(localizations.keys())
localizationTimeTree = spatial.KDTree(np.asarray([[t] for t in localization_timestamps]))
scan_timestamps = sorted(scans.keys())
scanTimeTree = spatial.KDTree(np.asarray([[t] for t in scan_timestamps]))

with torch.no_grad():
    print("Loading embedding model...")
    model = EmbeddingNet()
    model.load_state_dict(torch.load(args.model))
    model.eval()
    model.cuda()
    print("Finished loading embedding model")

    print("Creating embeddings for scans...")
    embedding_info = {}
    for timestamp, cloud in scans.items():
        normalized_cloud = normalize_point_cloud(cloud)
        point_set = torch.tensor([normalized_cloud])
        point_set = point_set.transpose(2, 1)
        point_set = point_set.cuda()
        embedding_info[timestamp] = model(point_set)
        del point_set
    print("Finished creating embeddings")
    embedding_timestamps = sorted(embedding_info.keys())
    embedding_clouds = np.asarray([embedding_info[t][0].detach().cpu().numpy() for t in embedding_timestamps]).squeeze(1)

print("Finding embedding matches...")
embeddingTree = spatial.KDTree(embedding_clouds)

MATCH_THRESHOLD = 10
loop_closures = []

last_closure_timestamp = 0

for idx in range(len(embedding_timestamps)):
    timestamp = embedding_timestamps[idx]
    _, last_loc_idx = localizationTimeTree.query([last_closure_timestamp])
    _, curr_loc_idx = localizationTimeTree.query([timestamp])
    import pdb; pdb.set_trace()
    current_location_est = localizations[localization_timestamps[curr_loc_idx]] + np.random.normal(0, 1, (1, 3))
    last_location_est = localizations[localization_timestamps[last_loc_idx]] + np.random.normal(0, 1, (1, 3))

    threshold = (timestamp - last_closure_timestamp) * 1.5

    match_dist, match_idx = embeddingTree.query(embedding_clouds[idx], k=2, distance_upper_bound=threshold)
    if(match_idx[1] < len(embedding_clouds)):
        print("MATCH", match_dist[1], match_idx[1])
        print("ORIGINAL LOCATION", current_location_est)
        print("LOOP CLOSED LOCATION", localizations[localization_timestamps[match_idx[1]]])
        loop_closures.append(embedding_clouds[match_idx[1]])
        last_closure_timestamp = embedding_timestamps[match_idx[1]]
        last_closure_location_est = localizations[localization_timestamps[match_idx[1]]]
print("Finished finding embedding matches")
print(len(loop_closures))

f = open('loop_closure_timestamps.data', 'w')
f.write('\n'.join([json.dumps(lc) for lc in loop_closures]))
f.close()