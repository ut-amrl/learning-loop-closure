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
parser.add_argument('--publish_triplets', type=bool, default=False, help="if included, publish evaluated triplets, as well as classification result.")

opt = parser.parse_args()
train_helpers.initialize_logging(str(int(time.time())), 'evaluate_')
print_output(opt)

num_workers = int(opt.workers)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

classifier = train_helpers.create_classifier('', opt.model)
classifier.eval()
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

triplets = np.zeros((batch_count, opt.batch_size, 3, 2))

for i, data in tqdm(enumerate(dataloader, 0)):
    ((clouds, locations, _), (similar_clouds, similar_locs, _), (distant_clouds, distant_locs, _)) = data
    clouds = clouds.transpose(2, 1)
    similar_clouds = similar_clouds.transpose(2, 1)
    distant_clouds = distant_clouds.transpose(2, 1)
    
    clouds = clouds.cuda()
    similar_clouds = similar_clouds.cuda()
    distant_clouds = distant_clouds.cuda()

    scores, (x_trans_feat, y_trans_feat), (translation, theta) = classifier(torch.cat([clouds, clouds], dim=0), torch.cat([similar_clouds, distant_clouds], dim=0))
    predictions = torch.argmax(scores, dim=1).cpu()
    
    train_helpers.update_metrics(metrics, predictions, labels)

    if opt.publish_triplets:
        triplets[i, :, 0, 0] = timestamp
        triplets[i, :, 1, 0] = similar_timestamp
        triplets[i, :, 2, 0] = distant_timestamp
        triplets[i, :, 1, 1] = (predictions_pos == 1).cpu()
        triplets[i, :, 2, 1] = (predictions_neg == 0).cpu()

acc = (metrics[0] + metrics[1]) / sum(metrics)
prec = (metrics[0]) / (metrics[0] + metrics[2])
rec = (metrics[0]) / (metrics[0] + metrics[3])
print_output('(Acc: %f, Precision: %f, Recall: %f)' % (acc, prec, rec))
