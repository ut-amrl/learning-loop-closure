import argparse

import rosbag
import numpy as np
import torch

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import helpers
from data_processing.data_processing_helpers import LCBagDataReader
from config import data_generation_config

parser = argparse.ArgumentParser()

parser.add_argument('--bag_file', type=str, required=True, help='Bag file to walk through for finding loop closures')
parser.add_argument('--alt_bag_file', type=str, required=False, help='Second file to walk through for finding loop closures. If not provided, bag_file will be checked against itself')
parser.add_argument('--time_spacing', type=float, default=1.5, help='Spacing between places to check for LC')

parser.add_argument('--match_model', type=str, required=False)
parser.add_argument('--embedding_model', type=str, required=False)
parser.add_argument('--structured_embedding_model', type=str, required=False)

opt = parser.parse_args()


convert_to_clouds = True
if opt.match_model:
  model = helpers.create_scan_matcher(opt.match_model)
  convert_to_clouds = False


bag = rosbag.Bag(opt.bag_file)

if opt.alt_bag_file:
  alt_bag = rosbag.Bag(opt.alt_bag_file)
else:
  alt_bag = rosbag.Bag(opt.bag_file)

bag_reader = LCBagDataReader(bag, data_generation_config['LIDAR_TOPIC'], data_generation_config['LOCALIZATION_TOPIC'], convert_to_clouds, opt.time_spacing, opt.time_spacing)
alt_reader = LCBagDataReader(alt_bag, data_generation_config['LIDAR_TOPIC'], data_generation_config['LOCALIZATION_TOPIC'], convert_to_clouds, opt.time_spacing, opt.time_spacing)

base_trajectory = []
for timestamp in bag_reader.get_localization_timestamps():
  loc = bag_reader.get_localizations()[timestamp]
  base_trajectory.append(loc[:2])
base_trajectory = np.array(base_trajectory)


target_trajectory = []
for timestamp in alt_reader.get_localization_timestamps():
  loc = alt_reader.get_localizations()[timestamp]
  target_trajectory.append(loc[:2])
target_trajectory = np.array(target_trajectory)

#Set up trajectory plot
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(base_trajectory[:, 0], base_trajectory[:, 1], 0, color='blue')
ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], 5, color='green')

# Now find loop closures along these trajectories
# First lets try 1 to many
base_timestamp = bag_reader.get_localization_timestamps()[20]
base_scan = torch.tensor(bag_reader.get_closest_scan_by_time(base_timestamp)[0]).cuda()
for idx, timestamp in enumerate(alt_reader.get_localization_timestamps()):
  scan = torch.tensor(alt_reader.get_closest_scan_by_time(timestamp)[0]).cuda()
  scores = model(base_scan.unsqueeze(0), scan.unsqueeze(0))
  prediction = torch.argmax(torch.nn.functional.softmax(scores))
  if (prediction):
    ax.plot([base_trajectory[20, 0], target_trajectory[idx, 0]], [base_trajectory[20, 1], target_trajectory[idx, 1]], [0, 5])

plt.show()