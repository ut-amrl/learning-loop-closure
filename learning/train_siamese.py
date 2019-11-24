import argparse
import os
import select
import sys
import random
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
from model import PointNetSiamese
from pointnet.model import feature_transform_regularizer
from dataset import LCDataset, LCTripletDataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=2, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=1200, help='number of points in each point cloud')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument(
    '--train_set', type=str, default='train', help='subset of the data to train on. One of [val, dev, train].')
parser.add_argument('--outf', type=str, default='cls_siamese', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--model', type=str, default='', help='pretrained model to start with');

opt = parser.parse_args()

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = LCTripletDataset(
    root=opt.dataset,
    split=opt.train_set)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

embedder = PointNetSiamese()
if opt.model != '':
    embedder.load_state_dict(torch.load(opt.model))
embedder.cuda()

optimizer = optim.Adam(embedder.parameters(), lr=1e-3, weight_decay=1e-5)

print("Loading training data into memory...", )
dataset.load_data()
print("Finished loading training data.")

lossFunc = torch.nn.BCEWithLogitsLoss().cuda()

pos_labels = torch.tensor(np.zeros((opt.batch_size, 1))).cuda()
neg_labels = torch.tensor(np.ones((opt.batch_size, 1))).cuda()

print("Press 'return' at any time to finish training after the current epoch.")
for epoch in range(opt.nepoch):
    total_loss = 0

    # We want to reload the triplets every 5 epochs to get new matches
    if epoch % 15 == 0:
        dataset.load_triplets()
        batch_count = len(dataset) // opt.batch_size
        print("Loaded new training triplets: {0} batches of size {1}".format(batch_count, opt.batch_size))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.workers),
            drop_last=True)

    for i, data in enumerate(dataloader, 0):
        ((clouds, locations, _), (similar_clouds, similar_locs, _), (distant_clouds, distant_locs, _)) = data
        clouds = clouds.transpose(2, 1)
        similar_clouds = similar_clouds.transpose(2, 1)
        distant_clouds = distant_clouds.transpose(2, 1)
        
        clouds = clouds.cuda()
        similar_clouds = similar_clouds.cuda()
        distant_clouds = distant_clouds.cuda()

        optimizer.zero_grad()
        embedder.zero_grad()
        embedder.train()

        pos_scores = embedder(clouds, similar_clouds)
        neg_scores = embedder(clouds, distant_clouds)
        loss = lossFunc(pos_scores, pos_labels)
        loss += lossFunc(neg_scores, neg_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('[Epoch %d] Total loss: %f' % (epoch, total_loss))
    torch.save(embedder.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print("Completed training for {0} epochs".format(epoch + 1))