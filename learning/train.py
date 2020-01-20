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
from model import EmbeddingNet
from pointnet.model import feature_transform_regularizer
from dataset import LCDataset, LCTripletDataset
import time
from tqdm import tqdm
import train_helpers
from train_helpers import print_output


class TripletLoss(torch.nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.sum()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=2, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument(
    '--train_set', type=str, default='train', help='subset of the data to train on. One of [val, dev, train].')
parser.add_argument(
    '--generate_embeddings', type=bool, default=False, help='if true, generate embeddings for test set in embeddings/*timestamp*')
parser.add_argument('--feature_transform', type=bool, default=False, help='Whether or not to additionally use feature transforms')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--embedding_model', type=str, default='', help='pretrained embedding model to start with')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--distance_cache', type=str, default=None, help='cached overlap info to start with')

opt = parser.parse_args()
start_time = str(int(time.time()))
train_helpers.initialize_logging(start_time)

print_output(opt)

num_workers = int(opt.workers)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = train_helpers.load_dataset(opt.dataset, opt.train_set, opt.distance_cache)

out_dir = opt.outf + '_' + dataset.dataset_info['name'] + '_' + dataset.split

if opt.feature_transform:
    out_dir += '_feat'

try:
    os.makedirs(out_dir)
except OSError:
    pass

embedder = train_helpers.create_embedder(opt.embedding_model, opt.feature_transform)
embedder.train()

optimizer = optim.Adam(embedder.parameters(), lr=1e-3, weight_decay=1e-5)
lossFunc = TripletLoss(10)

pos_labels = torch.tensor(np.ones((opt.batch_size, 1)).astype(np.long)).squeeze(1).cuda()
neg_labels = torch.tensor(np.zeros((opt.batch_size, 1)).astype(np.long)).squeeze(1).cuda()
labels = torch.cat([pos_labels, neg_labels], dim=0)

print_output("Press 'return' at any time to finish training after the current epoch.")
for epoch in range(opt.nepoch):
    total_loss = 0

    # We want to reload the triplets every 5 epochs to get new matches
    dataset.load_triplets()
    batch_count = len(dataset) // opt.batch_size
    print_output("Loaded new training triplets: {0} batches of size {1}".format(batch_count, opt.batch_size))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)

    for i, data in tqdm(enumerate(dataloader, 0)):
        ((clouds, locations, _), (similar_clouds, similar_locs, _), (distant_clouds, distant_locs, _)) = data
        clouds = clouds.transpose(2, 1)
        similar_clouds = similar_clouds.transpose(2, 1)
        distant_clouds = distant_clouds.transpose(2, 1)
        
        clouds = clouds.cuda()
        similar_clouds = similar_clouds.cuda()
        distant_clouds = distant_clouds.cuda()

        optimizer.zero_grad()
        embedder.zero_grad()

        anchor_embeddings, trans_feat, trans, theta = embedder(clouds)
        similar_embeddings, sim_feat, sim_trans, sim_theta = embedder(similar_clouds)
        distant_embeddings, dist_feat, dist_trans, dist_theta = embedder(distant_clouds)

        # Compute loss here
        loss = lossFunc.forward(anchor_embeddings, similar_embeddings, distant_embeddings)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 1e-2
            loss += feature_transform_regularizer(sim_feat) * 1e-2
            loss += feature_transform_regularizer(dist_feat) * 1e-2

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('[Epoch %d] Total loss: %f' % (epoch, total_loss))
    train_helpers.save_model(embedder, out_dir, epoch)
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
