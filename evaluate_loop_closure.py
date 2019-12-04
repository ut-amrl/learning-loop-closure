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
localizationTree = spatial.KDTree(np.asarray([localizations[t][:2] for t in localization_timestamps]))
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
MATCH_THRESHOLD = 10
loop_closures = []

CLOSURE_MIN_TIME_GAP = 4
MAX_DIST_THRESHOLD = 4
MAX_EMBEDDING_THRESHOLD = 1

for idx in range(len(embedding_timestamps)):
    timestamp = embedding_timestamps[idx]
    last_closure_timestamp = 0 if len(loop_closures) == 0 else loop_closures[-1]['target_timestamp']
    _, last_loc_idx = localizationTimeTree.query([last_closure_timestamp])
    _, curr_loc_idx = localizationTimeTree.query([timestamp])
    current_location_est = localizations[localization_timestamps[curr_loc_idx]]
    current_location_est[:2] = current_location_est[:2] + np.random.normal(0, 1, (1, 2))
    last_location_est = localizations[localization_timestamps[last_loc_idx]]
    last_location_est[:2] = last_location_est[:2] + np.random.normal(0, 1, (1, 2))

    if (timestamp - last_closure_timestamp) < CLOSURE_MIN_TIME_GAP:
        continue

    log_time_dist = np.log(timestamp - last_closure_timestamp)

    dist_threshold = MAX_DIST_THRESHOLD - 1/log_time_dist
    nearby_location_indices = localizationTree.query_ball_point(current_location_est[:2], dist_threshold)
    if len(nearby_location_indices) == 0:
        continue

    _, nearby_scan_indices = scanTimeTree.query([[localization_timestamps[loc_idx]] for loc_idx in nearby_location_indices])
    max_match_timestamp = timestamp - CLOSURE_MIN_TIME_GAP
    _, max_match_idx = scanTimeTree.query([max_match_timestamp])
    filtered_scan_indices = [n for n in nearby_scan_indices if n < max_match_idx]
    if len(filtered_scan_indices) == 0:
        continue

    emb_threshold = MAX_EMBEDDING_THRESHOLD - 1/log_time_dist
    embeddingTree = spatial.KDTree(np.take(embedding_clouds, filtered_scan_indices, axis=0))
    match_dist, match_idx = embeddingTree.query(embedding_clouds[idx], distance_upper_bound=emb_threshold)

    if match_idx >= len(filtered_scan_indices):
        continue

    print("MATCH", match_dist, match_idx)
    print("ORIGINAL LOCATION, TIMESTAMP", current_location_est, timestamp)
    match_timestamp = embedding_timestamps[match_idx]
    _, localization_time_idx = localizationTimeTree.query([match_timestamp])
    localization_timestamp = localization_timestamps[localization_time_idx]
    print("LOOP CLOSED LOCATION, TIMESTAMP", localizations[localization_timestamp], match_timestamp)
    emb_info = embedding_info[embedding_timestamps[match_idx]]
    print("PREDICTION TRANSLATION, THETA", emb_info[2].detach().cpu().numpy(), emb_info[3].detach().cpu().numpy())
    loop_closures.append({ 'source_embedding': embedding_clouds[match_idx].tolist(), 'source_timestamp': match_timestamp, 'target_embedding': embedding_clouds[idx].tolist(), 'target_timestamp': timestamp })

print("Finished finding embedding matches")
print(len(loop_closures))

f = open('loop_closure_timestamps.data', 'w')
f.write('\n'.join([json.dumps(lc) for lc in loop_closures]))
f.close()