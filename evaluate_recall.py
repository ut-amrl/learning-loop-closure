import rosbag
import rospy
import argparse
import numpy as np
import random
import torch
import statistics
import math
from geometry_msgs.msg import Pose

from learning.model import PointNetLC
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
    model = PointNetLC()
    model.load_state_dict(torch.load(args.model))
    model.eval()
    model.cuda()
    print("Finished loading embedding model")

def handle_location_input(data):
    print("Recieved Input from localization gui", data)
    print("Possible angle", math.atan2(data.orientation.z, data.orientation.w))
    angle = 0 # atan2(data.orientation.z, data.orientation.w)
    location = [data.position.x, data.position.y, angle]
    print("Target pose: ", location)

    _, closest_loc_indices = localizationTree.query(location[:2], k=10)
    closest_locations = [loc_infos[i] for i in closest_loc_indices]
    closest_location_times = [[localization_timestamps[i]] for i in closest_loc_indices]
    print("Using closest poses from bag file: ", '\n'.join([str(c) for c in closest_locations]))
    
    print("Evaluating recall for scans near these poses...")
    _, closest_scan_indices = scanTimeTree.query(closest_location_times)
    embeddings = [embedding_for_scan(model, scans[scan_timestamps[i]])[0] for i in closest_scan_indices]
    
    embedding_distances = []
    for i in range(0, len(embeddings) - 1):
        for j in range(i+1, len(embeddings)):
            emb1 = embeddings[i]
            emb2 = embeddings[j]
            distance = torch.norm(emb1 - emb2).item()
            embedding_distances.append(distance)
    
    print("embedding distances", '\n'.join([str(d) for d in embedding_distances]))
    print("Statistics: mean {0}, median {1}, stdev {2}".format(statistics.mean(embedding_distances), statistics.median(embedding_distances), statistics.stdev(embedding_distances)))

def location_listener():
    rospy.init_node('evaluate_recall', anonymous=True)
    rospy.Subscriber('localization_gui/nav_goal', Pose, handle_location_input)
    rospy.spin()

print("Loaded bag file. Please run localization_gui to provide evaluation locations...")

location_listener()
