import torch.utils.data as data
from scipy.spatial import cKDTree
import os
import os.path
import glob
import torch
import numpy as np
import sys
import random
import math
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement
import pickle

sys.path.append(os.path.join(os.getcwd(), '..'))
from helpers import compute_overlap, normalize_point_cloud

CLOSE_DISTANCE_THRESHOLD = 1
FAR_DISTANCE_THRESHOLD = 0.5
OVERLAP_THRESHOLD = 0.75

def get_point_cloud_from_file(filename):
    point_set = np.loadtxt(filename).astype(np.float32)
    return normalize_point_cloud(point_set)

class LCDataset(data.Dataset):
    def __init__(self,
                 root,
                 split='dev'):
        self.root = root
        self.split = split

        info_file = os.path.join(self.root, 'dataset_info.json')

        self.dataset_info = json.load(open(info_file, 'r'))
        if self.split == 'val':
            self.file_list = glob.glob(os.path.join(self.root, 'point_*.data'))
        else:
            self.file_list = self.dataset_info[self.split + '_data']

    def __getitem__(self, index):
        fname = self.file_list[index]
        timestamp = fname[fname.find('_')+1:fname.find('.data')]
        location_file = os.path.join(
            self.root, 'location_{0}.data'.format(timestamp))
        location = np.loadtxt(location_file).astype(np.float32)
        cloud = get_point_cloud_from_file(os.path.join(self.root, fname))

        return cloud, location[:2], timestamp

    def __len__(self):
        return len(self.file_list)

class LCTripletDataset(data.Dataset):
    def __init__(self,
                 root,
                 split='train',
                 jitter_augmentation=False,
                 person_augmentation=False):
        self.root = root
        self.jitter_augmentation = jitter_augmentation
        self.person_augmentation = person_augmentation
        self.split = split

        info_file = os.path.join(self.root, 'dataset_info.json')
        #from IPython import embed; embed()
        self.dataset_info = json.load(open(info_file, 'r'))
        # self.overlap_radius = self.dataset_info['scanMetadata']['range_max'] * 0.4 if 'scanMetadata' in self.dataset_info else 4
        self.data_loaded = False
        self.triplets_loaded = False
        self.data = []
        self.overlaps = {}
        self.triplets = []

    def load_data(self):
        # Use dataset_info to load data files
        if self.split == 'val':
            filelist = glob.glob(os.path.join(self.root, 'point_*.data'))
            filelist = [os.path.relpath(f, self.root) for f in filelist]
        else:
            filelist = self.dataset_info[self.split + '_data']
            
        for fname in filelist:
            timestamp = fname[fname.find('_')+1:fname.find('.data')]
            location_file = os.path.join(
                self.root, 'location_{0}.data'.format(timestamp))
            location = np.loadtxt(location_file).astype(np.float32)
            cloud = get_point_cloud_from_file(os.path.join(self.root, fname))
            # random perturbations, because why not
            if self.jitter_augmentation:
                theta = np.random.uniform(-np.pi / 4, np.pi / 4)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])
                # random rotation
                cloud[:, :] = cloud[:, :].dot(rotation_matrix)
                # random jitter
                cloud += np.random.normal(0, 0.02, size=cloud.shape)
            self.data.append((cloud, location, timestamp))
        self.location_tree = cKDTree(np.asarray([d[1][:2] for d in self.data]))
        self.data_loaded = True

    def load_triplets(self):
        if not self.data_loaded:
            raise Exception('Call load_data before attempting to load triplets')
        del self.triplets[:]
        # Compute triplets
        for cloud, location, timestamp in self.data:
            neighbors = self.location_tree.query_ball_point(location[:2], CLOSE_DISTANCE_THRESHOLD)
            filtered_neighbors = self.filter_scan_matches(timestamp, location, neighbors[1:])
            idx = random.randint(0, len(filtered_neighbors) - 1)
            similar_cloud, similar_loc, similar_timestamp = self.data[idx]

            idx = random.randint(0, len(self.data) - 1)
            while idx in filtered_neighbors:
                idx = random.randint(0, len(self.data) - 1)
            distant_cloud, distant_loc, distant_timestamp = self.data[idx]
            self.triplets.append((
                (cloud, location, timestamp),
                (similar_cloud, similar_loc, similar_timestamp),
                (distant_cloud, distant_loc, distant_timestamp)
            ))

        self.triplets_loaded = True

    def filter_scan_matches(self, timestamp, location, neighbors):
        return np.asarray(list(filter(self.check_overlap(location, timestamp), neighbors)))

    def check_overlap(self, location, timestamp):
        def overlap_checker(alt_idx):
            alt_loc = self.data[alt_idx][1]
            alt_timestamp = self.data[alt_idx][2]
            key = (timestamp, alt_timestamp) if timestamp < alt_timestamp else (alt_timestamp, timestamp)
            if key in self.overlaps:
                return self.overlaps[key] > OVERLAP_THRESHOLD
            else:
                overlap = compute_overlap(location, alt_loc)
                self.overlaps[key] = overlap
                return self.overlaps[key] > OVERLAP_THRESHOLD

        return overlap_checker

    def _get_distance_cache(self):
        return self.dataset_info['name'] + '_' + self.split + '_distances.pkl'

    def load_distances(self):
        print("Loading overlap information from cache...")
        with open(self._get_distance_cache(), 'rb') as f:
            self.overlaps = pickle.load(f)

    def cache_distances(self):
        print("Saving overlap information to cache...")
        with open(self._get_distance_cache(), 'wb') as f:
            pickle.dump(self.overlaps, f)

    def __getitem__(self, index):
        if not self.triplets_loaded:
            raise Exception('Call load_triplets before attempting to access elements')

        return self.triplets[index]

    def __len__(self):
        return len(self.triplets)
