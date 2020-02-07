import argparse
import os
import select
import sys
import torch
import torch.utils.data
import numpy as np
import pickle
import time
import random
from scipy.spatial import cKDTree
from tqdm import tqdm
from learning import train_helpers
from learning.train_helpers import print_output
from learning.dataset import LCDataset



parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=2, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--model', type=str, default='', help='model to evaluate');
parser.add_argument('--threshold', type=float, default=0.3, help='Threshold of uniquness score')
opt = parser.parse_args()
start_time = str(int(time.time()))
train_helpers.initialize_logging(start_time, 'evaluate_')
print_output(opt)

num_workers = int(opt.workers)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

embedder = train_helpers.create_embedder(opt.model)
embedder.eval()
dataset = LCDataset(opt.dataset)
batch_count = len(dataset) // opt.batch_size
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)

embeddings = np.ndarray((batch_count * opt.batch_size, 16))
locations = np.ndarray((batch_count * opt.batch_size, 2))
timestamps = np.ndarray((batch_count * opt.batch_size))
with torch.no_grad():
    for i, data in tqdm(enumerate(dataloader, 0)):
        clouds, locs, ts = data
        clouds = clouds.transpose(2, 1).cuda()

        emb, _, _ = embedder(clouds)
        embeddings[i * opt.batch_size:i*opt.batch_size + opt.batch_size] = emb.cpu().numpy()
        locations[i * opt.batch_size:i*opt.batch_size + opt.batch_size] = locs
        timestamps[i * opt.batch_size:i*opt.batch_size + opt.batch_size] = np.array(ts)

emb_tree = cKDTree(embeddings)
loc_tree = cKDTree(locations)

nearby_embs = [emb_tree.query_ball_point(e, 0.5) for e in embeddings]

uniqueness_scores = []
unique_timestamps = []
for i in range(len(nearby_embs)):
    base_loc = locations[i]
    emb_list = nearby_embs[i]
    nearby_loc_matches = loc_tree.query_ball_point(base_loc, 1.5)
    distant_emb_matches = set(emb_list) - set(nearby_loc_matches)
    uniqueness_score = len(distant_emb_matches) / len(emb_list)
    uniqueness_scores.append(uniqueness_score)
    if (uniqueness_score < opt.threshold):
        unique_timestamps.append(round(timestamps[i], 5))

print(uniqueness_scores)

print("GOOD TIMESTAMPS")
print(unique_timestamps)
print(len(unique_timestamps))