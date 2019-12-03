import argparse
import os
import select
import sys
import random
import math
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
from model import FullNet
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
parser.add_argument('--outf', type=str, default='cls_transform', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--model', type=str, default='', help='pretrained model to start with');

opt = parser.parse_args()

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = LCDataset(
    root=opt.dataset,
    split=opt.train_set)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

embedder = EmbeddingNet()
if opt.embedder_model != '':
    embedder.load_state_dict(torch.load(opt.model))
embedder.cuda()

classifier = FullNet(embedder)
classifier.cuda()

optimizer = optim.Adam(embedder.parameters(), lr=1e-3, weight_decay=1e-5)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.workers),
    drop_last=True)

def transform_scans(clouds):
    theta = random.random() * (2 * math.pi)
    trans = torch.tensor([random.random() * 5, random.random() * 5]).cuda()
    c = math.cos(theta)
    s = math.sin(theta)
    batch_size = clouds.shape[0]
    rot = torch.tensor([[c, s], [-s, c]]).cuda()
    clouds_batched = clouds.view(batch_size, -1, 2, 1)
    rot_repeated = rot.unsqueeze(0).repeat(clouds_batched.shape[1], 1, 1)
    trans_repeated = trans.unsqueeze(0).repeat(clouds_batched.shape[1], 1, 1).view(-1, 2, 1)
    rotated = torch.zeros(clouds_batched.shape).cuda()
    for i in range(batch_size):
        rotated[i] = torch.bmm(rot_repeated, clouds_batched[i]) +trans_repeated
    return rotated.squeeze(), theta, trans

lossFunc = torch.nn.NLLLoss().cuda()
labels = torch.tensor([0] * opt.batch_size).cuda()
print("Press 'return' at any time to finish training after the current epoch.")
for epoch in range(opt.nepoch):
    total_loss = 0

    total_predictions = [0, 0]
    correct_predictions = 0

    for i, data in enumerate(dataloader, 0):
        clouds, _, _ = data
        
        clouds = clouds.cuda()
        transformed, theta, translation = transform_scans(clouds)

        optimizer.zero_grad()
        embedder.zero_grad()
        embedder.train()

        clouds =  clouds.transpose(2, 1)
        transformed = transformed.transpose(2, 1)

        scores, (x_trans_feat, y_trans_feat), (translation_est, theta_est) = classifier(clouds, transformed)

        # Compute loss here        
        predictions = torch.argmax(scores, dim=1).cpu()
        loss = lossFunc(scores, labels)
        loss += torch.mean(torch.abs(translation - translation_est))
        loss += torch.mean(torch.abs(theta - theta_est))
        loss += feature_transform_regularizer(x_trans_feat) * 1e-3
        loss += feature_transform_regularizer(y_trans_feat) * 1e-3

        for i in range(len(predictions)):
            if predictions[i].item() == labels[i].item():
                correct_predictions += 1

            if predictions[i].item():
                total_predictions[0] += 1
            else:
                total_predictions[1] += 1

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('[Epoch %d] Total loss: %f' % (epoch, total_loss))
    torch.save(embedder.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print("Completed training for {0} epochs".format(epoch + 1))