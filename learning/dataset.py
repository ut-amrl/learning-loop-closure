import torch.utils.data as data
from scipy.spatial import KDTree
import os
import os.path
import glob
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
    point_set = point_set - \
        np.expand_dims(np.mean(point_set, axis=0), 0)  # center
    point_set = np.delete(point_set, 2, axis=1)
    # normalize
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set = point_set / dist  # scale
    return point_set


class LCDataset(data.Dataset):
    def __init__(self,
                 root,
                 split='dev'):
        self.root = root
        self.split = split

        info_file = os.path.join(self.root, 'dataset_info.json')
        #from IPython import embed; embed()
        self.dataset_info = json.load(open(info_file, 'r'))
        self.file_list = self.dataset_info[self.split + '_data']

    def __getitem__(self, index):
        fname = self.file_list[index]
        timestamp = fname[fname.find('_')+1:fname.find('.data')]
        location_file = os.path.join(
            self.root, 'location_{0}.data'.format(timestamp))
        location = np.loadtxt(location_file).astype(np.float32)
        cloud = get_point_cloud_from_file(os.path.join(self.root, fname))

        return cloud, location, timestamp

    def __len__(self):
        return len(self.file_list)


class LCTripletDataset(data.Dataset):
    def __init__(self,
                 root,
                 split='train',
                 data_augmentation=False):
        self.root = root
        self.data_augmentation = data_augmentation
        self.split = split

        info_file = os.path.join(self.root, 'dataset_info.json')
        #from IPython import embed; embed()
        self.dataset_info = json.load(open(info_file, 'r'))
        self.data_loaded = False
        self.triplets_loaded = False
        self.data = []
        self.triplets = []

    def load_data(self):
        # Use dataset_info to load data files
        filelist = self.dataset_info[self.split + '_data']
        for fname in filelist:
            timestamp = fname[fname.find('_')+1:fname.find('.data')]
            location_file = os.path.join(
                self.root, 'location_{0}.data'.format(timestamp))
            location = np.loadtxt(location_file).astype(np.float32)
            cloud = get_point_cloud_from_file(os.path.join(self.root, fname))
            # random perturbations, because why not
            if self.data_augmentation:
                theta = np.random.uniform(0, np.pi*2)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])
                # random rotation
                cloud[:, :] = cloud[:, :].dot(rotation_matrix)
                # random jitter
                cloud += np.random.normal(0, 0.02, size=cloud.shape)
            self.data.append((cloud, location, timestamp))

        self.location_tree = KDTree(np.asarray([d[1] for d in self.data]))
        self.data_loaded = True

    def load_triplets(self):
        if not self.data_loaded:
            raise Exception('Call load_data before attempting to load triplets')
        del self.triplets[:]
        # Compute triplets
        for cloud, location, timestamp in self.data:
            neighbors = self.location_tree.query_ball_point(
                location, CLOSE_DISTANCE_THRESHOLD)
            idx = random.randint(0, len(neighbors) - 1)
            similar_cloud, similar_loc, similar_timestamp = self.data[idx]

            idx = random.randint(0, len(self.data) - 1)
            while idx in neighbors:
                idx = random.randint(0, len(self.data) - 1)
            distant_cloud, distant_loc, distant_timestamp = self.data[idx]
            self.triplets.append((
                (cloud, location, timestamp),
                (similar_cloud, similar_loc, similar_timestamp),
                (distant_cloud, distant_loc, distant_timestamp)
            ))

        self.triplets_loaded = True

    def __getitem__(self, index):
        if not self.triplets_loaded:
            raise Exception('Call load_triplets before attempting to access elements')

        return self.triplets[index]

    def __len__(self):
        return len(self.triplets)
