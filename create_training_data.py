import rosbag
import argparse
import numpy as np
import os
import random
import json
from tqdm import tqdm
from scipy import spatial
from helpers import scan_to_point_cloud, get_scans_and_localizations_from_bag

TIME_SPACING = 0.01
TRAIN_SPLIT = 0.15
DEV_SPLIT = 0.8
VAL_SPLIT = 1-DEV_SPLIT

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

print("Loading scans & Localization from Bag file")
scans, localizations, metadata = get_scans_and_localizations_from_bag(bag, args.lidar_topic, args.localization_topic, TIME_SPACING)
print("Finished processing Bag file")

dataset_info['scanMetadata'] = metadata
dataset_info['numScans'] = len(scans.keys())

localizationTree = spatial.cKDTree([list([l]) for l in sorted(localizations.keys())])

base_path = './data/' + args.dataset_name + '/'
if not os.path.exists(base_path):
    os.makedirs(base_path)

if not args.info_only:
    print("Writing data to disk for {0} scans...".format(
        dataset_info['numScans']))
filenames = []
for timestamp, cloud in tqdm(list(scans.items())):
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
val_indices = set(range(count)) - set(dev_indices)

dataset_info['train_data'] = [filenames[i] for i in train_indices]
dataset_info['dev_data'] = [filenames[i] for i in dev_indices]
dataset_info['val_data'] = [filenames[i] for i in val_indices]

info_path = os.path.join(base_path, 'dataset_info.json')
with open(info_path, 'w') as f:
    f.write(json.dumps(dataset_info, indent=2))
