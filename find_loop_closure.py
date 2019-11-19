import rosbag
import argparse
import numpy as np
import os
import random
import json
import torch
import time
from learning.model import PointNetLC
from learning.dataset import normalize_point_cloud
from pc_helpers import scan_to_point_cloud
from scipy import spatial

parser = argparse.ArgumentParser(description='Find loop closure locations for some ROS bag')
parser.add_argument('--bag_file', type=str, help='path to the bag file containing training data')
parser.add_argument('--lidar_topic', type=str, help='name of topic containing lidar information')
parser.add_argument('--localization_topic', type=str, help='name of topic containing localization information')
parser.add_argument('--model', type=str, help='state dict of an already trained LC model to use')

args = parser.parse_args()
bag = rosbag.Bag(args.bag_file)

scans = {}
localizations = {}

print ("Loading scans & Localization from Bag file")
last_timestamp = 0.0
for topic, msg, t in bag.read_messages(topics=[args.lidar_topic, args.localization_topic]):
    timestamp = t.secs + t.nsecs * 1e-9
    if (timestamp - last_timestamp) > 0.1 and len(scans.keys()) < 5000:
        last_timestamp = timestamp
        if (topic == args.lidar_topic):
            scans[timestamp] = scan_to_point_cloud(msg)
        elif (topic == args.localization_topic):
            localizations[timestamp] = msg
bag.close()
print ("Finished processing Bag file", len(scans.keys()), "scans")

with torch.no_grad():
    print("Loading embedding model...")
    model = PointNetLC()
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
for idx in range(len(embedding_timestamps)):
    matches = embeddingTree.query_ball_point(embedding_clouds[idx], 0.1)
    filtered_matches = [m for m in matches if abs(embedding_timestamps[idx] - embedding_timestamps[m]) > 3]
    if len(filtered_matches) < MATCH_THRESHOLD and len(filtered_matches) > 1:
        # there aren't that many matches for this timestamp. Let's say its good for loop closure
        loop_closures.append((embedding_timestamps[idx], embedding_timestamps[filtered_matches[1]]))

print("Finished finding embedding matches")
print(len(loop_closures))

f = open('loop_closure_timestamps.data', 'w')
f.write('\n'.join([json.dumps(lc) for lc in loop_closures]))
f.close()