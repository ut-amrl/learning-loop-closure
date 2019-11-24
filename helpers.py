import numpy as np
import torch
from learning.dataset import normalize_point_cloud
from sensor_msgs.msg import PointCloud2, PointField
from rospy import rostime

def scan_to_point_cloud(scan, trim_edges=True):
    angle_offset = 0.0
    if trim_edges:
        scan.ranges = scan.ranges[4:-4]

    cloud = np.zeros((len(scan.ranges), 3)).astype(np.float32)

    for idx,r in enumerate(scan.ranges):
        if r >= scan.range_min and r <= scan.range_max:
            point = np.transpose(np.array([[r, 0]]))
            cos, sin = np.cos(angle_offset), np.sin(angle_offset)
            rotation = np.array([(cos, -sin), (sin, cos)])
            point = np.matmul(rotation, point)
            cloud[idx][0] = point[0]
            cloud[idx][1] = point[1]
        angle_offset += scan.angle_increment

    return cloud

def embedding_for_scan(model, scan):
    normalized_cloud = normalize_point_cloud(scan)
    cloud = torch.tensor([normalized_cloud])
    cloud = cloud.transpose(2, 1)
    cloud = cloud.cuda()
    result = model(cloud)
    del cloud
    return result

def get_scans_and_localizations_from_bag(bag, lidar_topic, localization_topic, scan_timestep=0, loc_timestep=0):
    localizations = {}
    scans = {}
    start_time = bag.get_start_time()
    last_scan_time = -10
    last_localization_time = -10
    for topic, msg, t in bag.read_messages(topics=[lidar_topic, localization_topic]):
        timestamp = t.secs + t.nsecs * 1e-9 - start_time
        if (topic == lidar_topic and timestamp - last_scan_time > scan_timestep):
            last_scan_time = timestamp
            scans[timestamp] = scan_to_point_cloud(msg)
        elif (topic == localization_topic and timestamp - last_localization_time > loc_timestep):
            last_localization_time = timestamp
            localizations[timestamp] = np.asarray([msg.x, msg.y, msg.angle])

    return scans, localizations


def create_ros_pointcloud():
    msg = PointCloud2()
    msg.header.frame_id="map"
    msg.header.stamp = rostime.Time.now()
    msg.header.seq = 1
    msg.fields = [
        PointField('x', 0, 7, 1),
        PointField('y', 4, 7, 1),
        PointField('z', 8, 7, 1)
    ]
    msg.height = 1
    msg.width = 0
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 0
    msg.is_dense = True
    return msg

def publish_ros_pointcloud(publisher, msg, clouds):
    for cloud in clouds:
        msg.data += cloud.tobytes()
        msg.width += len(cloud)
    publisher.publish(msg)