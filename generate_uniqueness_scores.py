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
parser.add_argument('--threshold', type=int, default=2, help='Threshold of distance for which 2 scans are "similar"')
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

with torch.no_grad():
    for i, data in tqdm(enumerate(dataloader, 0)):
        clouds, locations, timestamps = data
        clouds = clouds.transpose(2, 1).cuda()

        emb, _, _ = embedder(clouds)
        embeddings[i * opt.batch_size:i*opt.batch_size + opt.batch_size] = emb.cpu().numpy()

emb_tree = cKDTree(embeddings)

uniqueness_scores = np.array([len(emb_tree.query_ball_point(e, 0.5)) for e in embeddings])
print(uniqueness_scores)