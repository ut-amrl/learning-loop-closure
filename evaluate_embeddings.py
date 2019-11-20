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
print ("Bag has ", bag.get_message_count(topic_filters=[args.localization_topic]), " Localization messages")
print ("Start time:", bag.get_start_time())
print ("End time:", bag.get_end_time())

last_loc_timestamp = 0.0
last_scan_timestamp = 0.0

for topic, msg, t in bag.read_messages(topics=[args.lidar_topic, args.localization_topic]):
    timestamp = t.secs + t.nsecs * 1e-9
    if (topic == args.lidar_topic and timestamp - last_scan_timestamp > 0.5):
        last_scan_timestamp = timestamp
        scans[timestamp] = scan_to_point_cloud(msg)
    elif (topic == args.localization_topic and timestamp - last_loc_timestamp > 0.5):
        localizations[timestamp] = np.asarray([msg.x, msg.y])
        last_loc_timestamp = timestamp
bag.close()
print ("Finished processing Bag file", len(scans.keys()), "scans", len(localizations.keys()), "localizations")


print("Finding location matches")
localization_timestamps = sorted(localizations.keys())
loc_infos = np.asarray([localizations[t] for t in localization_timestamps])
localizationTree = spatial.KDTree(loc_infos)

location_matches = localizationTree.query_pairs(.15)
filtered_location_matches = [m for m in location_matches if localization_timestamps[m[1]] - localization_timestamps[m[0]] > 10]
print(len(filtered_location_matches))
print("Finished finding location matches")

scan_timestamps = sorted(scans.keys())
scanTimeTree = spatial.KDTree(np.asarray([list([t]) for t in scan_timestamps]))


with torch.no_grad():
    print("Loading embedding model...")
    model = PointNetLC()
    model.load_state_dict(torch.load(args.model))
    model.eval()
    model.cuda()
    print("Finished loading embedding model")

    def embedding_for_scan(scan):
        normalized_cloud = normalize_point_cloud(scan)
        point_set = torch.tensor([normalized_cloud])
        point_set = point_set.transpose(2, 1)
        point_set = point_set.cuda()
        return model(point_set)
        del point_set

    print("Evaluating embeddings for matches...")
    DISTANCE_THRESHOLD = 10
    correct = 0
    avg_distance = 0
    for idx1, idx2 in filtered_location_matches:
        loc_time1 = localization_timestamps[idx1]
        loc_time2 = localization_timestamps[idx2]
        [_, [scan_idx1, scan_idx2]] = scanTimeTree.query([[loc_time1], [loc_time2]])

        scan1 = scans[scan_timestamps[scan_idx1]]
        scan2 = scans[scan_timestamps[scan_idx2]]

        embedding1, _, _ = embedding_for_scan(scan1)
        embedding2, _, _ = embedding_for_scan(scan2)
        distance = torch.norm(embedding1 - embedding2).item()
        avg_distance += distance
        if (distance < DISTANCE_THRESHOLD):
            correct += 1
    
    avg_distance /= len(filtered_location_matches)
    print("Average distance for embeddings that should have matched: ", avg_distance)
    print("{0} embeddings were 'close' out of {1} potential loop closure locations".format(correct, len(filtered_location_matches)))