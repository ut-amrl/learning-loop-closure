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
from model import PointNetLC
from pointnet.model import feature_transform_regularizer
from dataset import LCDataset, LCTripletDataset
from tqdm import tqdm


class TripletLoss(torch.nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, p=2, dim=1) * 1e-1
        distance_negative = torch.norm(anchor - negative, p=2, dim=1) * 1e-1
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.sum()

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
parser.add_argument(
    '--generate_embeddings', type=bool, default=False, help='if true, generate embeddings for test set in embeddings/*timestamp*')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
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

embedder = PointNetLC()
if opt.model != '':
    embedder.load_state_dict(torch.load(opt.model))
embedder.cuda()

optimizer = optim.Adam(embedder.parameters(), lr=1e-3, weight_decay=1e-5)

print("Loading training data into memory...", )
dataset.load_data()
print("Finished loading training data.")
lossFn = TripletLoss(5)

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

        anchor_embeddings, trans, trans_feat = embedder(clouds)
        similar_embeddings, sim_trans, sim_feat = embedder(similar_clouds)
        distant_embeddings, dist_trans, dist_feat = embedder(distant_clouds)

        # Compute loss here
        loss = lossFn.forward(anchor_embeddings, similar_embeddings, distant_embeddings)
        loss += feature_transform_regularizer(trans_feat) * 1e-3
        loss += feature_transform_regularizer(sim_feat) * 1e-3
        loss += feature_transform_regularizer(dist_feat) * 1e-3

        (U, S, V) = torch.svd(trans)
        loss += torch.mean(torch.norm(trans - torch.bmm(U, V), dim=(1,2))) * 1e-3;
        (U, S, V) = torch.svd(sim_trans)
        loss += torch.mean(torch.norm(sim_trans - torch.bmm(U, V), dim=(1,2))) * 1e-3;
        (U, S, V) = torch.svd(dist_trans)
        loss += torch.mean(torch.norm(dist_trans - torch.bmm(U, V), dim=(1,2))) * 1e-3;

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('[Epoch %d] Total loss: %f' % (epoch, total_loss))
    torch.save(embedder.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print("Completed training for {0} epochs".format(epoch + 1))

if opt.generate_embeddings:
    test_dataset = LCDataset(
        root=opt.dataset,
        split='dev')

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        drop_last=True)
    print("Generating output for test set...")
    OUTPUT_DIR = 'embeddings'
    embedder.eval()
    with torch.no_grad():
        for i, data in enumerate(testdataloader, 0):
            clouds, locations, timestamps = data
            clouds = clouds.transpose(2, 1)
            clouds = clouds.cuda()

            embeddings, _, _ = embedder.forward(clouds)

            for i in range(len(embeddings)):
                np.savetxt(os.path.join(OUTPUT_DIR, os.path.basename(timestamps[i])), embeddings[i].cpu().numpy())
