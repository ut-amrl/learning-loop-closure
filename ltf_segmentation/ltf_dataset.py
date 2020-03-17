class LTFDataset:
  def __init__(self, bag_file, base_topic, filtered_topic, dimensions=200):
    self.bag_file = bag_file
    self.base_topic = base_topic
    self.filtered_topic = filtered_topic
    self.load_data()

  def load_data(self):
    
    for topic, msg, t in tqdm(bag.read_messages(topics=[self.base_topic, self.filtered_topic])):
        timestamp = t.secs + t.nsecs * 1e-9 - start_time
        if (topic == lidar_topic and timestamp - last_scan_time > scan_timestep):
            if not 'range_max' in metadata:
                metadata['range_max'] = msg.range_max
                metadata['range_min'] = msg.range_min
                metadata['angle_min'] = msg.angle_min
                metadata['angle_max'] = msg.angle_max
                metadata['angle_increment'] = msg.angle_increment

            last_scan_time = timestamp
            scans[timestamp] = scan_to_point_cloud(msg)
        elif (topic == localization_topic and timestamp - last_localization_time > loc_timestep):
            last_localization_time = timestamp
            localizations[timestamp] = np.asarray([msg.x, msg.y, msg.angle])
