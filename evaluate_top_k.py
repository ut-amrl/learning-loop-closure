import rospy
import argparse
import numpy as np
import random
import torch
import os
from learning.train_helpers import create_embedder
from helpers import scan_to_point_cloud, LCBagDataReader, embedding_for_scan
import json

parser = argparse.ArgumentParser(
    description='Find loop closure locations for some ROS bag')
parser.add_argument('--recall_dataset', type=str,
                    help='path to the bag file containing recakk observation data')
parser.add_argument('--model', type=str,
                    help='state dict of an already trained LC model to use')
parser.add_argument('--k', type=int, default=3, help='Number of nearby scans to select')
parser.add_argument('--threshold', type=int, default=30, help='Upper bound on distance after which we dont even consider matches')
args = parser.parse_args()

model = create_embedder(args.model)
model.eval()

ds_file = os.path.join(args.recall_dataset, 'datasets.json')
with open(ds_file) as f:
    dataset_names = json.loads(f.read())

def load_cloud(ds, timestamp):
    dataset_name = dataset_names[ds]
    cloud_data = os.path.join(dataset_name, 'point_{0}.npy'.format(timestamp))
    return np.load(cloud_data)

# recall location x dataset x timestamp
obs_timestamps = np.load(os.path.join(args.recall_dataset, 'observation_timestamps.npy'))

# We want to assert that for every recall location, each scan in dataset 0 matches each scan in dataset 1
total = 0.0
match = 0.0
for loc_idx in range(obs_timestamps.shape[0]):
    choice_ds = random.randint(0, 1)
    choice_idx = random.randint(0, obs_timestamps.shape[2] - 1)
    choice_timestamp = obs_timestamps[loc_idx, choice_ds, choice_idx].squeeze()
    
    print("Loc {0}, choice: Dataset {1}, timestamp {2}".format(loc_idx, choice_ds, choice_timestamp))

    # all our candidates come from the "other" dataset
    candidate_timestamps = np.array([[np.random.choice(obs_timestamps[loc, 1 - choice_ds].squeeze(), args.k), loc] for loc in range(obs_timestamps.shape[0])])
    candidate_timestamps = np.array([[(c[0][i], c[1]) for i in range(len(c[0]))] for c in candidate_timestamps]).reshape(-1, 2)
    # candidates should be the 'rest' of the clouds
    candidates = np.array([load_cloud(1 - choice_ds, t[0]) for t in candidate_timestamps])
    choice = load_cloud(choice_ds, choice_timestamp)
    with torch.no_grad():
        embeddings, _, _ = embedding_for_scan(model, candidates, batched=True)
        anchor, _, _ = embedding_for_scan(model, choice)

        distances = torch.norm(anchor.repeat(len(embeddings), 1, 1) - embeddings, dim=1).squeeze()
        min_dist, min_indices = torch.topk(distances, args.k, largest=False)
        min_indices = min_indices.cpu().numpy()
        min_dist = min_dist.cpu().numpy()

        print("MIN DISTANCES", min_dist)

        min_candidates = candidate_timestamps[min_indices]
        min_locs = [c[1] for c in min_candidates]
        
        for i in range(len(min_candidates)):
            loc = min_candidates[i][1]
            dist = min_dist[i]
            if loc == loc_idx:
                match += 1
            else:
                print("Incorrect match. Loc {0} dist {1}".format(loc, dist))
            total += 1


print("Results: {0} / {1} = {2}".format(match, total, match / total))