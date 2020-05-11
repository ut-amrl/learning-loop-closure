import argparse

import rosbag
import numpy as np
import torch

import os
import sys
import evaluation_helpers

sys.path.append(os.path.join(os.getcwd(), '..'))
import helpers
from data_processing.data_processing_helpers import LCBagDataReader, scan_to_point_cloud
from config import Configuration

config = Configuration(False, True)
config.add_argument('--alt_bag_file', type=str)
config.add_argument('--time_spacing', type=float, default=1.5)
config = config.parse()

scan_conv, scan_match, scan_transform = helpers.create_laser_networks(config.model_dir, config.model_epoch)
scan_conv.eval()
scan_match.eval()
scan_transform.eval()
convert_to_clouds = False

bag = rosbag.Bag(config.bag_file)

if config.alt_bag_file:
  alt_bag = rosbag.Bag(config.alt_bag_file)
else:
  alt_bag = rosbag.Bag(config.bag_file)

bag_reader = LCBagDataReader(bag, config.lidar_topic, config.localization_topic, convert_to_clouds, config.time_spacing, config.time_spacing)
alt_reader = LCBagDataReader(alt_bag, config.lidar_topic, config.localization_topic, convert_to_clouds, config.time_spacing, config.time_spacing)

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
from matplotlib.lines import Line2D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(base_trajectory[:, 0], base_trajectory[:, 1], 0, color='blue')
ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], 5, color='green')

fp = 0
tp = 0
# Now find loop closures along these trajectories
for base_idx in np.linspace(0, len(bag_reader.get_localization_timestamps()) - 1, 50):      
  # loc = bag_reader.get_localizations()[bag_reader.get_localization_timestamps()[base_idx]]
  # map_fig = plt.figure()

  # if opt.map_name:
  #   evaluation_helpers.draw_map(map_fig, opt.map_name)
  #   evaluation_helpers.visualize_location(map_fig, loc)
  
  #   plt.show(block=False)

  # response = raw_input('Press y to see the full plot for this index')
  # plt.close()

  # if response == 'y':
  base_idx = int(base_idx)
  base_timestamp = bag_reader.get_localization_timestamps()[base_idx]
  base_scan = torch.tensor(bag_reader.get_closest_scan_by_time(base_timestamp)[0].ranges).cuda()

  for idx, timestamp in enumerate(alt_reader.get_localization_timestamps()):
    with torch.no_grad():
      scan = torch.tensor(alt_reader.get_closest_scan_by_time(timestamp)[0].ranges).cuda()
      conv = scan_conv(base_scan.unsqueeze(0), scan.unsqueeze(0))
      scores = scan_match(conv)
      prediction = torch.argmax(torch.nn.functional.softmax(scores))

    if (prediction):
      if np.linalg.norm(base_trajectory[base_idx] - target_trajectory[idx]) < 2.5:
        tp += 1
        c = 'gray'
      else:
        fp +=1 
        c = 'red'
      ax.plot([base_trajectory[base_idx, 0], target_trajectory[idx, 0]], [base_trajectory[base_idx, 1], target_trajectory[idx, 1]], [0, 5], color=c)

info_string = "True Positives: {0}\tFalse Positives: {1}".format(tp, fp)

legend_elements = [
  plt.Line2D([0], [0], color='gray', lw=4, label='Correct Match ({0})'.format(tp)),
  Line2D([0], [0], color='red', lw=4, label='Incorrect Match ({0})'.format(fp)),
]

ax.legend(handles=legend_elements)
print(info_string)
# plt.text(10, 10, info_string)
plt.show()