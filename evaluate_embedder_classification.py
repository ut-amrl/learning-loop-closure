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
from tqdm import tqdm
from learning import train_helpers
from learning.train_helpers import print_output

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=2, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--evaluation_set', type=str, default='dev', help='subset of the data to train on. One of [val, dev, train].')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--model', type=str, default='', help='model to evaluate');
parser.add_argument('--distance_cache', type=str, default=None, help='cached overlap info to start with')

opt = parser.parse_args()
train_helpers.initialize_logging(str(int(time.time())), 'evaluate_')
print_output(opt)

num_workers = int(opt.workers)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

embedder = train_helpers.create_embedder(opt.model)
embedder.eval()
dataset = train_helpers.load_dataset(opt.dataset, opt.evaluation_set, opt.distance_cache, num_workers)
batch_count = len(dataset) // opt.batch_size
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)

pos_labels = torch.tensor(np.ones((opt.batch_size, 1)).astype(np.long)).squeeze(1)
neg_labels = torch.tensor(np.zeros((opt.batch_size, 1)).astype(np.long)).squeeze(1)
labels = torch.cat([pos_labels, neg_labels], dim=0).cuda()

metrics = [0.0, 0.0, 0.0, 0.0] # True Positive, True Negative, False Positive, False Negative

for i, data in tqdm(enumerate(dataloader, 0)):
    ((clouds, locations, _), (similar_clouds, similar_locs, _), (distant_clouds, distant_locs, _)) = data
    clouds = clouds.transpose(2, 1)
    similar_clouds = similar_clouds.transpose(2, 1)
    distant_clouds = distant_clouds.transpose(2, 1)
    
    clouds = clouds.cuda()
    similar_clouds = similar_clouds.cuda()
    distant_clouds = distant_clouds.cuda()

    
    anchor_embeddings, trans_feat, trans, theta = embedder(clouds)
    similar_embeddings, sim_feat, sim_trans, sim_theta = embedder(similar_clouds)
    distant_embeddings, dist_feat, dist_trans, dist_theta = embedder(distant_clouds)

    distance_pos = torch.norm(anchor_embeddings - similar_embeddings, p=2, dim=1) * 1e-1
    distance_neg = torch.norm(anchor_embeddings - distant_embeddings, p=2, dim=1) * 1e-1

    predictions_pos = (distance_pos < 2).int()
    predictions_neg = (distance_neg < 2).int()

    train_helpers.update_metrics(metrics, torch.cat([predictions_pos, predictions_neg]), labels)

acc = (metrics[0] + metrics[1]) / sum(metrics)
prec = (metrics[0]) / (metrics[0] + metrics[2])
rec = (metrics[0]) / (metrics[0] + metrics[3])
print_output('(Acc: %f, Precision: %f, Recall: %f)' % (acc, prec, rec))
