import numpy as np
import sys
import os
import math
cloud = np.load('../data/gdc1_cobot/point_125.12109.npy')

sys.path.append(os.path.join(os.getcwd(), '..'))
from helpers import normalize_point_cloud

# Takes an input point cloud and discretizes it into an occupancy grid of size (dimensions x dimensions)
# Assumes the input point cloud is normalized to [-1, 1]
def discretize_point_cloud(cloud, dimensions):
  discretized = np.zeros((dimensions, dimensions))

  for p in cloud:
    cell = (int(math.floor(dimensions/2 + p[0] * dimensions/2)), int(math.floor(dimensions/2 + p[1] * dimensions/2)))
    discretized[cell] = 1
  
  return discretized


discretized = discretize_point_cloud(cloud, 200)
from matplotlib import pyplot as plt
plt.imshow(discretized, interpolation='nearest')
plt.show()