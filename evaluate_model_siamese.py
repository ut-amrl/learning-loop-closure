import rosbag
import argparse
import numpy as np
import random
import torch
import statistics
from learning.model import FullNet
from helpers import scan_to_point_cloud, get_scans_and_localizations_from_bag, normalize_point_cloud
from scipy import spatial

parser = argparse.ArgumentParser(
    description='Find loop closure locations for some ROS bag')
parser.add_argument('--bag_file', type=str,
                    help='path to the bag file containing training data')
parser.add_argument('--lidar_topic', type=str,
                    help='name of topic containing lidar information')
parser.add_argument('--localization_topic', type=str,
                    help='name of topic containing localization information')
parser.add_argument('--model', type=str,
                    help='state dict of an already trained LC model to use')

args = parser.parse_args()
bag = rosbag.Bag(args.bag_file)

scans = {}
localizations = {}

print("Loading scans & Localization from Bag file")
print("Bag has ", bag.get_message_count(topic_filters=[
      args.localization_topic]), " Localization messages")
print("Start time:", bag.get_start_time())
print("End time:", bag.get_end_time())

start_time = bag.get_start_time()
SCAN_TIMESTEP = 0.1
LOC_TIMESTEP = 0.15
scans, localizations, _ = get_scans_and_localizations_from_bag(bag, args.lidar_topic, args.localization_topic, SCAN_TIMESTEP, LOC_TIMESTEP)

print("Finished processing Bag file", len(scans.keys()),
      "scans", len(localizations.keys()), "localizations")

print("Finding location matches")
localization_timestamps = sorted(localizations.keys())
loc_infos = np.asarray([localizations[t][:2] for t in localization_timestamps])
localizationTree = spatial.KDTree(loc_infos)

location_matches = localizationTree.query_pairs(.05)
# Only keep location matches that are distant in time-space, since these are the only ones that would be good for loop closure
filtered_location_matches = [
    m for m in location_matches if localization_timestamps[m[1]] - localization_timestamps[m[0]] > 15]
print(len(filtered_location_matches))
print("Finished finding location matches")

scan_timestamps = sorted(scans.keys())
scanTimeTree = spatial.KDTree(np.asarray([list([t]) for t in scan_timestamps]))

with torch.no_grad():
    print("Loading embedding model...")
    model = FullNet()
    model.load_state_dict(torch.load(args.model))
    model.eval()
    model.cuda()
    print("Finished loading embedding model")
    
    def create_input(scan):
        normalized_cloud = normalize_point_cloud(scan)
    cloud = torch.tensor([normalized_cloud])
    cloud = cloud.transpose(2, 1)
    cloud = cloud.cuda()
    return cloud

    print("Evaluating some random embeddings...")
    matches = 0
    for idx in range(2000):
        scan_ts_1 = scan_timestamps[random.randint(0, len(scan_timestamps) - 1)]
        scan_ts_2 = scan_timestamps[random.randint(0, len(scan_timestamps) - 1)]
        while abs(scan_ts_2 - scan_ts_1) < 3:
            scan_ts_2 = scan_timestamps[random.randint(0, len(scan_timestamps) - 1)]
        scan1 = scans[scan_ts_1]
        scan2 = scans[scan_ts_2]
        scores, trans, rot = model(create_input(scan1), create_input(scan2))

        match = torch.argmax(scores)
        if (match == 0):
            matches += 1
    print("Got {0} matches out of 2000 random embeddings".format(matches))

    print("Evaluating embeddings for matches...")
    matches = 0
    for idx1, idx2 in filtered_location_matches:
        loc_time1 = localization_timestamps[idx1]
        loc_time2 = localization_timestamps[idx2]
        [_, [scan_idx1, scan_idx2]] = scanTimeTree.query(
            [[loc_time1], [loc_time2]])

        scan1 = scans[scan_timestamps[scan_idx1]]
        scan2 = scans[scan_timestamps[scan_idx2]]
        scores = model(create_input(scan1), create_input(scan2))

        match = torch.argmax(scores)
        if (match == 0):
            matches += 1

    print("{0} embeddings were 'close' out of {1} potential loop closure locations".format(
        matches, len(filtered_location_matches)))
