import argparse
import numpy as np
import rosbag
from matplotlib import pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
from data_processing.data_processing_helpers import LCBagDataReader
from evaluation.evaluation_helpers import visualize_cloud, visualize_location, draw_map

parser = argparse.ArgumentParser()
parser.add_argument(
    '--bag_file', type=str, required=True, help='bag file to walk through for evaluation scans')
parser.add_argument(
    '--localization_topic', type=str, default='/Cobot/Localization', help='localization topic')
parser.add_argument(
    '--lidar_topic', type=str, default='/Cobot/Laser', help='lidar topics')
parser.add_argument(
    '--map_name', type=str, required=True, help='map over which bag file is walking')
parser.add_argument(
    '--dataset', type=str, default='manual_evaluation', help='output evaluation dataset')
parser.add_argument(
    '--append', type=bool, default=True, help='whether or not to add to an existing evaluation dataset')

args = parser.parse_args()
# The Basic idea: Walk through all the scans in a bag file, along with their locations, and present them to the user
# Store those frames the user marks as "keyframes", along with their locations, in a dataset

map_file = '../../cobot/maps/{0}/{0}_vector.txt'.format(args.map_name)

bag = rosbag.Bag(args.bag_file)
start_timestamp = bag.get_start_time()

data_reader = LCBagDataReader(bag, args.lidar_topic, args.localization_topic)

pairs = sorted(data_reader.get_scans().items(), lambda x, y: 1 if float(x[0]) > float(y[0]) else -1)

accepted = []

for timestamp, cloud in pairs[0::30]:
  loc = data_reader.get_closest_localization_by_time(timestamp)[0]
  plt.figure(1)
  plt.clf()
  visualize_cloud(plt, cloud, color='blue')
  plt.figure(2)
  plt.clf()
  draw_map(plt, map_file)
  visualize_location(plt, loc, 'blue')

  plt.show(block=False)
  accept = str(raw_input('Is this scan a keyframe? (y/n): '))

  if accept == 'y' or accept == 'Y':
    accepted.append((timestamp, loc, cloud))
  elif accept =='break':
    break


for timestamp, loc, cloud in accepted:
    stamp = str(round(timestamp, 5))
    np.save(os.path.join(args.dataset + '_' + args.map_name, 'loc_{0}.npy'.format(stamp)), loc)
    np.save(os.path.join(args.dataset + '_' + args.map_name, 'cloud_{0}.npy'.format(stamp)), cloud)