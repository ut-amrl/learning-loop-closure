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

args = parser.parse_args()
bag = rosbag.Bag(args.bag_file)

scans = {}
localizations = {}

for topic, msg, t in bag.read_messages(topics=[args.lidar_topic, args.localization_topic]):
    timestamp = t.secs + t.nsecs * 1e-9
    if (topic == args.lidar_topic):
        scans[timestamp] = msg
    elif (topic == args.localization_topic):
        localizations[timestamp] = msg

localizationTree = spatial.KDTree([list([l]) for l in sorted(localizations.keys())])

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

base_path = './data/' + args.dataset_name + '/'
os.makedirs(base_path, exist_ok=True)

data = []
for timestamp, scan in scans.items():
    d, idx = localizationTree.query([timestamp])
    locTimestamp = sorted(localizations.keys())[idx]
    location = localizations[locTimestamp]
    cloud = scan_to_point_cloud(scan)
    cloud_file_name = 'point_' + str(timestamp) + '.data'
    f = open(base_path + cloud_file_name, 'w')
    cloudString = ''
    for p in cloud:
        cloudString += '{0} {1}\n'.format(p[0], p[1])
    f.write(cloudString)
    f.close()
    loc_file_name = 'point_' + str(timestamp) + '.data.location'
    f2 = open(base_path + loc_file_name, 'w')
    f2.write('{0} {1}\n'.format(location.x, location.y))
    f2.close()
    data.append((cloud, location, cloud_file_name))

count = len(data)
train_data = random.sample(list(range(count)), round(count * 0.15))
test_data = random.sample(list(range(count)), round(count * 0.4))
val_data = list(range(count))

os.makedirs(base_path + 'train_test_split', exist_ok=True)
f = open(base_path + 'train_test_split/shuffled_train_file_list.json', 'w')
f.write(json.dumps([data[t][2] for t in train_data]))
f.close()
f = open(base_path + 'train_test_split/shuffled_test_file_list.json', 'w')
f.write(json.dumps([data[t][2] for t in test_data]))
f.close()
f = open(base_path + 'train_test_split/shuffled_val_file_list.json', 'w')
f.write(json.dumps([data[t][2] for t in val_data]))
f.close()
bag.close()