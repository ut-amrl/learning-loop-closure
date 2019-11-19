import numpy as np
def scan_to_point_cloud(scan):
    angle_offset = 0.0
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