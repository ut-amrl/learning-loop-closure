import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
from model import PointNetLC
from dataset import LCDataset
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
        distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.sum()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=2, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=1200, help='number of points in each point cloud')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--model', type=str, default='', help='pretrained model to evaluate');

opt = parser.parse_args()

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = LCDataset(
    root=opt.dataset,
    num_points=opt.num_points)

test_dataset = LCDataset(
    root=opt.dataset,
    num_points=opt.num_points,
    split='test')

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    drop_last=True)

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    drop_last=True)

print(len(dataset), len(test_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

embedder = PointNetLC()

if opt.model != '':
    embedder.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(embedder.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
embedder.cuda()
lossFn = TripletLoss(1)
num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        point_clouds, locations = data
        similar_point_clouds, sLoc = dataset.get_similar_points(locations)
        distant_point_clouds, dLoc = dataset.get_distant_points(locations)
        point_clouds = point_clouds.transpose(2, 1)
        similar_point_clouds = similar_point_clouds.transpose(2, 1)
        distant_point_clouds = distant_point_clouds.transpose(2, 1)
        
        point_clouds = point_clouds.cuda()
        similar_point_clouds = similar_point_clouds.cuda()
        distant_point_clouds = distant_point_clouds.cuda()
        # print("DATA:")
        # print(point_clouds.shape)
        # print(similar_point_clouds.shape)
        # print(distant_point_clouds.shape)

        optimizer.zero_grad()
        embedder.train()

        anchor_embeddings, trans, trans_feat = embedder(point_clouds)
        similar_embeddings, _, _ = embedder(similar_point_clouds)
        distant_embeddings, _, _ = embedder(distant_point_clouds)

        # print("EMBEDDINGS:")
        # print(anchor_embeddings.shape)
        # print(similar_embeddings.shape)
        # print(distant_embeddings.shape)

        # Compute loss here
        loss = lossFn.forward(anchor_embeddings, similar_embeddings, distant_embeddings)

        # loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))
    
    scheduler.step()
    torch.save(embedder.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

