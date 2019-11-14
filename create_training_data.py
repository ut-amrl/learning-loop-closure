import rosbag
import argparse
import numpy as np
import os
import random
import json
from scipy import spatial

parser = argparse.ArgumentParser(description='Create some training data from ROS bags')
parser.add_argument('--bag_file', type=str, help='path to the bag file containing training data')
parser.add_argument('--lidar_topic', type=str, help='name of topic containing lidar information')
parser.add_argument('--localization_topic', type=str, help='name of topic containing localization information')
parser.add_argument('--dataset_name', type=str, help='defines the folder in which this generated data will be placed')
parser.add_argument('--max_scans', type=int, help='maximum number of scans to keep track of')

args = parser.parse_args()
bag = rosbag.Bag(args.bag_file)

scans = {}
localizations = {}


def scan_to_point_cloud(scan):
    angle_offset = 0.0
    cloud = list()
    for r in scan.ranges:
        if r >= scan.range_min and r <= scan.range_max:
            point = np.transpose(np.array([[r, 0]]))
            cos, sin = np.cos(angle_offset), np.sin(angle_offset)
            rotation = np.array([(cos, -sin), (sin, cos)])
            point = np.matmul(rotation, point)
            cloud.append(point)
        angle_offset += scan.angle_increment
    return cloud

print ("Loading scans & Localization from Bag file")
for topic, msg, t in bag.read_messages(topics=[args.lidar_topic, args.localization_topic]):
    if len(scans.keys()) < args.max_scans:
        timestamp = t.secs + t.nsecs * 1e-9
        if (topic == args.lidar_topic):
            scans[timestamp] = scan_to_point_cloud(msg)
        elif (topic == args.localization_topic):
            localizations[timestamp] = msg

print ("Finished processing Bag file")

localizationTree = spatial.KDTree([list([l]) for l in sorted(localizations.keys())])

base_path = './data/' + args.dataset_name + '/'
os.makedirs(base_path, exist_ok=True)

print ("Writing data to disk...")
filenames = []
for timestamp, cloud in list(scans.items())[:args.max_scans]:
    d, idx = localizationTree.query([timestamp])
    locTimestamp = sorted(localizations.keys())[idx]
    location = localizations[locTimestamp]
    cloud_file_name = 'point_' + str(timestamp) + '.data'
    f = open(base_path + cloud_file_name, 'w')
    cloudString = ''
    for p in cloud:
        cloudString += '{0} {1} 0\n'.format(p[0][0], p[1][0])
    f.write(cloudString)
    f.close()
    loc_file_name = 'point_' + str(timestamp) + '.data.location'
    f2 = open(base_path + loc_file_name, 'w')
    f2.write('{0} {1} 0\n'.format(location.x, location.y))
    f2.close()
    filenames.append(cloud_file_name)

print ("Writing partition information...", len(filenames))
count = len(filenames)
train_data = random.sample(list(range(count)), round(count * 0.15))
test_data = random.sample(list(range(count)), round(count * 0.4))
val_data = list(range(count))

os.makedirs(base_path + 'train_test_split', exist_ok=True)
f = open(base_path + 'train_test_split/shuffled_train_file_list.json', 'w')
f.write(json.dumps([filenames[t] for t in train_data]))
f.close()
f = open(base_path + 'train_test_split/shuffled_test_file_list.json', 'w')
f.write(json.dumps([filenames[t] for t in test_data]))
f.close()
f = open(base_path + 'train_test_split/shuffled_val_file_list.json', 'w')
f.write(json.dumps([filenames[t] for t in val_data]))
f.close()
bag.close()