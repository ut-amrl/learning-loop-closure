import rosbag
import argparse
import numpy as np
import os
import random
import json
from scipy import spatial
from pc_helpers import scan_to_point_cloud

TIME_SPACING = 0.025
TRAIN_SPLIT = 0.15
DEV_SPLIT = 0.4

parser = argparse.ArgumentParser(
    description='Create some training data from ROS bags')
parser.add_argument('--bag_file', type=str,
                    help='path to the bag file containing training data')
parser.add_argument('--lidar_topic', type=str,
                    help='name of topic containing lidar information')
parser.add_argument('--localization_topic', type=str,
                    help='name of topic containing localization information')
parser.add_argument('--dataset_name', type=str,
                    help='defines the folder in which this generated data will be placed')
parser.add_argument('--info_only', type=bool,
                    help='if set, only write dataset_info.json, assuming the data has already been written')


args = parser.parse_args()
bag = rosbag.Bag(args.bag_file)

start_timestamp = bag.get_start_time()

dataset_info = {
    'name': args.dataset_name,
    'sourceBag': os.path.abspath(args.bag_file),
    'startTime': start_timestamp,
}
scans = {}
localizations = {}
last_scan_timestamp = 0

print("Loading scans & Localization from Bag file")
for topic, msg, t in bag.read_messages(topics=[args.lidar_topic, args.localization_topic]):
    timestamp = t.secs + t.nsecs * 1e-9 - start_timestamp
    if (topic == args.lidar_topic and timestamp - last_scan_timestamp > TIME_SPACING):
        last_scan_timestamp = timestamp
        scans[timestamp] = [] if args.info_only else scan_to_point_cloud(msg)
    elif (topic == args.localization_topic):
        localizations[timestamp] = np.asarray([msg.x, msg.y])
bag.close()
print("Finished processing Bag file")

dataset_info['numScans'] = len(scans.keys())

localizationTree = spatial.KDTree([list([l])
                                   for l in sorted(localizations.keys())])

base_path = './data/' + args.dataset_name + '/'
if not os.path.exists(base_path):
    os.makedirs(base_path)

if not args.info_only:
    print("Writing data to disk for {0} scans...".format(
        dataset_info['numScans']))
filenames = []
for timestamp, cloud in list(scans.items()):
    d, idx = localizationTree.query([timestamp])
    locTimestamp = sorted(localizations.keys())[idx]
    location = localizations[locTimestamp]
    cloud_file_name = 'point_' + str(timestamp) + '.data'
    if not args.info_only:
        np.savetxt(base_path + cloud_file_name, cloud)
        loc_file_name = 'location_' + str(timestamp) + '.data'
        np.savetxt(base_path + loc_file_name, location)
    filenames.append(cloud_file_name)

print("Writing dataset information...", len(filenames))

count = len(filenames)
train_indices = random.sample(
    list(range(count)), int(round(count * TRAIN_SPLIT)))
dev_indices = random.sample(list(range(count)), int(round(count * DEV_SPLIT)))

dataset_info['train_data'] = [filenames[i] for i in train_indices]
dataset_info['dev_data'] = [filenames[i] for i in dev_indices]

info_path = os.path.join(base_path, 'dataset_info.json')
with open(info_path, 'w') as f:
    f.write(json.dumps(dataset_info, indent=2))
