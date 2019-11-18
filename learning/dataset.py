import torch.utils.data as data
from scipy.spatial import KDTree
import os
import os.path
import torch
import numpy as np
import sys
import random
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement

CLOSE_DISTANCE_THRESHOLD = 0.25
FAR_DISTANCE_THRESHOLD = 0.75

def get_point_cloud_from_file(filename):
    point_set = np.loadtxt(filename).astype(np.float32)
    return normalize_point_cloud(point_set)

def normalize_point_cloud(point_set):
    point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
    point_set = np.delete(point_set, 2, axis=1)
    # normalize 
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    point_set = point_set / dist #scale
    return point_set

class LCDataset(data.Dataset):
    def __init__(self,
                 root,
                 num_points,
                 split='train',
                 data_augmentation=True):
        self.root = root
        self.num_points = num_points
        self.data_augmentation = data_augmentation

        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))

        self.datapath = []
        idx = 0
        for file in filelist:
            data_file = os.path.join(self.root, file)
            location_file = os.path.join(self.root, file + '.location')
            location = np.loadtxt(location_file).astype(np.float32)
            self.datapath.append((data_file, location))
        self.location_tree = KDTree(np.asarray([d[1] for d in self.datapath]))

    def __getitem__(self, index):
        fn = self.datapath[index]
        loc = fn[1]
        point_set = get_point_cloud_from_file(fn[0])
        # random perturbations, because why not
        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,1]] = point_set[:,[0,1]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        out_points = np.zeros((self.num_points, 2)).astype(np.float32)
        out_points[:point_set.shape[0]] = point_set

        out_points = torch.from_numpy(out_points)
        loc = torch.from_numpy(loc)

        return out_points, loc, fn[0]

    def get_similar_points(self, locations):
        neighbor_matrix = self.location_tree.query_ball_point(locations, CLOSE_DISTANCE_THRESHOLD)
        similar_points = torch.tensor([])
        similar_locs = torch.tensor([])
        for i, neighbors in enumerate(neighbor_matrix):
            idx = random.randint(0, len(neighbors) - 1)
            points, loc, _ = self[idx]
            similar_points = torch.cat((similar_points, points.unsqueeze(0)), 0)
            similar_locs = torch.cat((similar_locs, loc.unsqueeze(0)), 0)

        return similar_points, similar_locs

    def get_distant_points(self, locations):
        neighbor_matrix = self.location_tree.query_ball_point(locations, FAR_DISTANCE_THRESHOLD)
        distant_points = torch.tensor([])
        distant_locs = torch.tensor([])

        for i in range(locations.shape[0]):
            idx = random.randint(0, len(self.datapath) - 1)
            while idx in neighbor_matrix[i]:
                idx = random.randint(0, len(self.datapath) - 1)

            points, loc, _ = self[idx]

            distant_points = torch.cat((distant_points, points.unsqueeze(0)), 0)
            distant_locs = torch.cat((distant_locs, loc.unsqueeze(0)), 0)

        return distant_points, distant_locs

    def get_random_points(self, batch_size):
        distant_points = torch.tensor([])
        distant_locs = torch.tensor([])

        for i in range(batch_size):
            idx = random.randint(0, len(self.datapath) - 1)
            points, loc = self[idx]
            distant_points = torch.cat((distant_points, points.unsqueeze(0)), 0)
            distant_locs = torch.cat((distant_locs, loc.unsqueeze(0)), 0)

        return distant_points, distant_locs

    def __len__(self):
        return len(self.datapath)