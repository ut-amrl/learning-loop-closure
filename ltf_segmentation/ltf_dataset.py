import rosbag
from tqdm import tqdm
from ltf_helpers import discretize_point_cloud

import sys
import os

sys.path.append(os.path.join(os.getcwd(), '..'))
from data_processing.data_processing_helpers import scan_to_point_cloud

class LTFDataset:
  def __init__(self, bag_file, base_topic, filtered_topic, dimensions=200, lidar_range=20):
    self.bag_file = bag_file
    self.base_topic = base_topic
    self.filtered_topic = filtered_topic
    self.dimensions = dimensions
    self.lidar_range = lidar_range
    self.load_data()

  def load_data(self):
    bag = rosbag.Bag(self.bag_file)
    timestamps = set()
    self.data = {}
    start_time = bag.get_start_time()
    for topic, msg, t in tqdm(bag.read_messages(topics=[self.base_topic, self.filtered_topic])):
        timestamp = t.secs + t.nsecs * 1e-9 - start_time
        timestamps.add(timestamp)
        if timestamp not in self.data:
          self.data[timestamp] = [{}, {}]
        if (topic == self.base_topic):
          self.data[timestamp][0] = discretize_point_cloud(scan_to_point_cloud(msg), self.lidar_range, self.dimensions)
        elif (topic == self.filtered_topic):
          
          self.data[timestamp][1] = discretize_point_cloud(scan_to_point_cloud(msg), self.lidar_range, self.dimensions)
    
    self.timestamps = sorted(list(timestamps))

  def __getitem__(self, index):
    return self.data[self.timestamps[index]]
