import rosbag
import argparse
import numpy as np
import os
import random
import json
from scipy import spatial
from pc_helpers import scan_to_point_cloud

parser = argparse.ArgumentParser(description='Create some training data from ROS bags')
parser.add_argument('--bag_file', type=str, help='path to the bag file containing training data')
parser.add_argument('--lidar_topic', type=str, help='name of topic containing lidar information')
parser.add_argument('--localization_topic', type=str, help='name of topic containing localization information')
parser.add_argument('--dataset_name', type=str, help='defines the folder in which this generated data will be placed')
parser.add_argument('--partitions_only', type=bool, help='if set, only write partition files, assuming the data has already been written')

args = parser.parse_args()
bag = rosbag.Bag(args.bag_file)

scans = {}
localizations = {}
last_scan_timestamp = 0

print ("Loading scans & Localization from Bag file")
for topic, msg, t in bag.read_messages(topics=[args.lidar_topic, args.localization_topic]):
    timestamp = t.secs + t.nsecs * 1e-9
    if (topic == args.lidar_topic and timestamp - last_scan_timestamp > 0.05):
        last_scan_timestamp = timestamp
        scans[timestamp] = [] if args.partitions_only else scan_to_point_cloud(msg)
    elif (topic == args.localization_topic):
        localizations[timestamp] = np.asarray([msg.x, msg.y])

bag.close()
print ("Finished processing Bag file")

localizationTree = spatial.KDTree([list([l]) for l in sorted(localizations.keys())])

base_path = './data/' + args.dataset_name + '/'
if not os.path.exists(base_path):
    os.makedirs(base_path)

print ("Writing data to disk for {0} scans...".format(len(scans.keys())))
filenames = []
for timestamp, cloud in list(scans.items()):
    d, idx = localizationTree.query([timestamp])
    locTimestamp = sorted(localizations.keys())[idx]
    location = localizations[locTimestamp]
    cloud_file_name = 'point_' + str(timestamp) + '.data'
    if not args.partitions_only:
        f = open(base_path + cloud_file_name, 'w')
        cloudString = ''
        for p in cloud:
            cloudString += '{0} {1} 0\n'.format(p[0], p[1])
        f.write(cloudString)
        f.close()
        loc_file_name = 'point_' + str(timestamp) + '.data.location'
        f2 = open(base_path + loc_file_name, 'w')
        f2.write('{0} {1} 0\n'.format(location[0], location[1]))
        f2.close()
    filenames.append(cloud_file_name)

print ("Writing partition information...", len(filenames))
count = len(filenames)
train_data = random.sample(list(range(count)), int(round(count * 0.15)))
test_data = random.sample(list(range(count)), int(round(count * 0.4)))
val_data = list(range(count))

split_path = os.path.join(base_path, 'train_test_split')
if not os.path.exists(split_path):
    os.makedirs(split_path)
f = open(os.path.join(split_path, 'shuffled_train_file_list.json'), 'w')
f.write(json.dumps([filenames[t] for t in train_data]))
f.close()
f = open(os.path.join(split_path, 'shuffled_test_file_list.json'), 'w')
f.write(json.dumps([filenames[t] for t in test_data]))
f.close()
f = open(os.path.join(split_path, 'shuffled_val_file_list.json'), 'w')
f.write(json.dumps([filenames[t] for t in val_data]))
f.close()
