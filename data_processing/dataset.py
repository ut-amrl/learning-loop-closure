import torch.utils.data as data
from scipy.spatial import cKDTree
import multiprocessing as mp
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
import pickle
from data_processing_helpers import compute_overlap

CLOSE_DISTANCE_THRESHOLD = .75
FAR_DISTANCE_THRESHOLD = 3
OVERLAP_THRESHOLD = 0.8
TIME_IGNORE_THRESHOLD = 0.5
class LCDataset(data.Dataset):
    def __init__(self,
                 root,
                 split=None):
        self.root = root
        self.split = split
        self.timestamp_tree = None

        info_file = os.path.join(self.root, 'dataset_info.json')

        self.dataset_info = json.load(open(info_file, 'r'))
        if self.split:
            self.file_list = self.dataset_info[self.split + '_data']
        else:
            self.file_list = [f[:f.rfind('.npy')] for f in glob.glob(os.path.join(self.root, 'point_*.npy'))]

        self.timestamps = [[float(f[f.find('point_') + len('point_'):])] for f in self.file_list]

    def __getitem__(self, index):
        fname = self.file_list[index]
        timestamp = fname[fname.rfind('_')+1:]

        return self.get_by_timestamp(timestamp)

    def get_by_timestamp(self, timestamp, include_angle=False):
        location_file = os.path.join(
            self.root, 'location_{0}.npy'.format(timestamp))
        location = np.load(location_file).astype(np.float32)
        cloud_file = os.path.join(self.root, 'point_{0}.npy'.format(timestamp))
        cloud = np.load(cloud_file).astype(np.float32)
        if not include_angle:
            location = location[:2]
        return cloud, location, timestamp

    def get_by_nearest_timestamp(self, target_timestamp, include_angle=False):
        if not self.timestamp_tree:
            self.timestamp_tree = cKDTree(self.timestamps)
        
        _, timestamp_idx = self.timestamp_tree.query([target_timestamp])
        timestamp =  self.timestamps[timestamp_idx][0]
        return self.get_by_timestamp(timestamp, include_angle)

    def __len__(self):
        return len(self.file_list)

class LCTripletDataset(data.Dataset):
    def __init__(self,
                 root,
                 split='train',
                 jitter_augmentation=True,
                 person_augmentation=False,
                 order_augmentation=False):
        self.root = root
        self.jitter_augmentation = jitter_augmentation
        self.person_augmentation = person_augmentation
        self.order_augmentation = order_augmentation
        self.split = split

        info_file = os.path.join(self.root, 'dataset_info.json')
        #from IPython import embed; embed()
        self.dataset_info = json.load(open(info_file, 'r'))
        # self.overlap_radius = self.dataset_info['scanMetadata']['range_max'] * 0.4 if 'scanMetadata' in self.dataset_info else 4
        self.data_loaded = False
        self.triplets_loaded = False
        self.computed_new_distances = False
        self.data = []
        self.overlaps = {}
        self.triplets = []

    def load_data(self):
        # Use dataset_info to load data files
        filelist = self.dataset_info[self.split + '_data']
            
        for fname in tqdm(filelist):
            timestamp = fname[fname.rfind('_')+1:]
            location_file = os.path.join(
                self.root, 'location_{0}.npy'.format(timestamp))
            location = np.load(location_file).astype(np.float32)
            cloud = np.load(os.path.join(self.root, fname + '.npy')).astype(np.float32)
            self.data.append((cloud, location, timestamp))
        self.location_tree = cKDTree(np.asarray([d[1][:2] for d in self.data]))
        self.data_loaded = True

    def _create_triplet(self, cloud, location, timestamp, similar_cloud, similar_loc, similar_timestamp):
        idx = random.randint(0, len(self.data) - 1)
        # We don't want anything that's even remotely nearby to count as "distant"
        dist_neighbors = self.location_tree.query_ball_point(location[:2], FAR_DISTANCE_THRESHOLD)
        while idx in dist_neighbors:
            idx = random.randint(0, len(self.data) - 1)
        distant_cloud, distant_loc, distant_timestamp = self.data[idx]
        return (
            (cloud, location, timestamp),
            (similar_cloud, similar_loc, similar_timestamp),
            (distant_cloud, distant_loc, distant_timestamp)
        )

    def _generate_augmented_triplet(self, cloud, location, timestamp):
        augmented_neighbors = self.generate_augmented_neighbors(cloud)
        idx = random.randint(0,  len(augmented_neighbors) - 1)
        similar_cloud = augmented_neighbors[idx]
        similar_loc = location
        similar_timestamp = timestamp

        return self._create_triplet(cloud, location, timestamp, similar_cloud, similar_loc, similar_timestamp)

    def _generate_triplet(self, cloud, location, timestamp):
        neighbors = self.location_tree.query_ball_point(location[:2], CLOSE_DISTANCE_THRESHOLD)
        filtered_neighbors = self.filter_scan_matches(timestamp, location, neighbors[1:])
        if len(filtered_neighbors) > 0:
            idx = np.random.choice(filtered_neighbors, 1)[0]
            similar_cloud, similar_loc, similar_timestamp = self.data[idx]
        else:
            return None

        return self._create_triplet(cloud, location, timestamp, similar_cloud, similar_loc, similar_timestamp)


    def load_triplets(self):
        if not self.data_loaded:
            raise Exception('Call load_data before attempting to load triplets')
        del self.triplets[:]

        self.triplets = filter(None, [self._generate_triplet(cloud, location, timestamp) for cloud, location, timestamp in tqdm(self.data)])
        self.triplets.extend([self._generate_augmented_triplet(cloud, location, timestamp) for cloud, location, timestamp in tqdm(self.data)])

        self.triplets_loaded = True

    def generate_augmented_neighbors(self, cloud):
        neighbors = []
        # random perturbations, because why not
        if self.jitter_augmentation:
            theta = np.random.uniform(-np.pi / 3, np.pi / 3)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            augmented = np.zeros(cloud.shape).astype(np.float32)
            # random rotation
            augmented[:, :] = cloud[:, :].dot(rotation_matrix)
            neighbors.append(augmented)
        
        if self.person_augmentation:
            pass
            # neighbors.append(augmented)
        
        #We will roll instead of randomly permuting, so sequential local features are maintained
        if self.order_augmentation:
            neighbors.append(np.roll(cloud, np.random.randint(0, cloud.shape[0] / 10), 0))

        return neighbors

    def filter_scan_matches(self, timestamp, location, neighbors):
        filtered = list(filter(self.time_filter(timestamp), neighbors))
        filtered = list(filter(self.check_overlap(location, timestamp), filtered))
        return np.array(filtered)

    def time_filter(self, timestamp):
        def filter_checker(alt_idx):
            alt_timestamp = self.data[alt_idx][2]
            return abs(float(timestamp) - float(alt_timestamp)) > TIME_IGNORE_THRESHOLD
        
        return filter_checker

    def check_overlap(self, location, timestamp):
        def overlap_checker(alt_idx):
            alt_loc = self.data[alt_idx][1]
            alt_timestamp = self.data[alt_idx][2]
            key = (timestamp, alt_timestamp) if timestamp < alt_timestamp else (alt_timestamp, timestamp)
            if key in self.overlaps:
                return self.overlaps[key] > OVERLAP_THRESHOLD
            else:
                overlap = compute_overlap(location, alt_loc)
                self.computed_new_distances = True
                self.overlaps[key] = overlap
                return self.overlaps[key] > OVERLAP_THRESHOLD

        return overlap_checker

    def _get_distance_cache(self):
        return self.dataset_info['name'] + '_' + self.split + '_distances.pkl'

    def load_distances(self, distance_cache):
        if not distance_cache:
            distance_cache = self._get_distance_cache()
        if os.path.exists(distance_cache):
            print("Loading overlap information from cache...")
            with open(distance_cache, 'rb') as f:
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


class LCCDataset(LCDataset):
    def __init__(self,
                 root,
                 timestamps,
                 split='dev'):
        self.labeled_timestamps = np.load(timestamps)
        super(LCCDataset, self).__init__(root, split)

    def __getitem__(self, index):
        timestamp = self.labeled_timestamps[index][0]
        label = int(self.labeled_timestamps[index][1])
        condition = float(self.labeled_timestamps[index][2])
        scale = float(self.labeled_timestamps[index][3])
        cloud, _, timestamp = self.get_by_nearest_timestamp(timestamp)

        return label, condition, scale, cloud, timestamp

    def __len__(self):
        return len(self.labeled_timestamps)
    
