import rosbag
import argparse
import numpy as np
import random
import torch
import statistics
from learning.model import EmbeddingNet
from helpers import scan_to_point_cloud, get_scans_and_localizations_from_bag, embedding_for_scan
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
parser.add_argument('--threshold', type=int,
                    help='distance threshold for saying things are "close"')
args = parser.parse_args()
bag = rosbag.Bag(args.bag_file)

scans = {}
localizations = {}

start_time = bag.get_start_time()
SCAN_TIMESTEP = 0.1
LOC_TIMESTEP = 0.15
scans, localizations, _ = get_scans_and_localizations_from_bag(bag, args.lidar_topic, args.localization_topic, SCAN_TIMESTEP, LOC_TIMESTEP)

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
    model = EmbeddingNet()
    model.load_state_dict(torch.load(args.model))
    model.eval()
    model.cuda()
    print("Finished loading embedding model")

    correct = 0
    random_distances = []
    print("Evaluating some random embeddings...")
    for idx in range(2000):
        scan1 = scans[scan_timestamps[random.randint(0, len(scan_timestamps) - 1)]]
        scan2 = scans[scan_timestamps[random.randint(0, len(scan_timestamps) - 1)]]

        embedding1, _, _, _ = embedding_for_scan(model, scan1)
        embedding2, _, _, _ = embedding_for_scan(model, scan2)
        distance = torch.norm(embedding1 - embedding2, p=2).item()
        random_distances.append(distance)
        if (distance < args.threshold):
            correct += 1
    print("For random embeddings (avg: {0}, med: {1}, stdev: {2})".format(
        statistics.mean(random_distances), statistics.median(random_distances), statistics.stdev(random_distances)))
    print("{0} embeddings were 'close' out of {1} random locations".format(
        correct, len(random_distances)))
    # for each location match, check if the embeddings are close
    print("Evaluating embeddings for matches...")
    match_distances = []
    correct = 0
    for idx1, idx2 in filtered_location_matches:
        loc_time1 = localization_timestamps[idx1]
        loc_time2 = localization_timestamps[idx2]
        [_, [scan_idx1, scan_idx2]] = scanTimeTree.query(
            [[loc_time1], [loc_time2]])

        scan1 = scans[scan_timestamps[scan_idx1]]
        scan2 = scans[scan_timestamps[scan_idx2]]

        embedding1, _, _, _ = embedding_for_scan(model, scan1)
        embedding2, _, _, _ = embedding_for_scan(model, scan2)
        distance = torch.norm(embedding1 - embedding2, p=2).item()
        match_distances.append(distance)
        if (distance < args.threshold):
            correct += 1

    print("For embeddings that should have matched (avg: {0}, med: {1}, stdev: {2})".format(
        statistics.mean(match_distances), statistics.median(match_distances), statistics.stdev(match_distances)))
    print("{0} embeddings were 'close' out of {1} potential loop closure locations".format(
        correct, len(match_distances)))
