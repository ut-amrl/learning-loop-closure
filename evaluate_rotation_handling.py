import rosbag
import argparse
import numpy as np
import random
import math
import torch
import statistics
from learning.model import FullNet
from helpers import get_scans_and_localizations_from_bag, normalize_point_cloud
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

def rotate_scan(cloud, theta=None):
    if not theta:
        theta = random.random() * (2 * math.pi)
    c = math.cos(theta)
    s = math.sin(theta)
    rot = torch.tensor([[c, s], [-s, c]]).cuda()
    cloud_batched = cloud.view(-1, 2, 1)
    rot_repeated = rot.unsqueeze(0).repeat(cloud_batched.shape[0], 1, 1)
    rotated = torch.bmm(rot_repeated, cloud_batched)
    return rotated, theta

def create_input(scan):
    normalized_cloud = normalize_point_cloud(scan)
    cloud = torch.tensor([normalized_cloud])
    cloud = cloud.cuda()
    return cloud

with torch.no_grad():
    print("Loading embedding model...")
    model = FullNet()
    model.load_state_dict(torch.load(args.model))
    model.eval()
    model.cuda()
    print("Finished loading embedding model")

    match_distances = []
    correct = 0
    for t in scans.keys():
        scan = scans[t]
        cloud = create_input(scan)
        rotated, theta = rotate_scan(cloud)

        cloud = cloud.view(1, 2, -1)
        rotated = rotated.view(1, 2, -1)

        scores, _, (translation_est, theta_est) = model(cloud, rotated)
