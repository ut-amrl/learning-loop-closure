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
parser.add_argument('--outf', type=str, default='cls_full', help='output folder')
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

embedder = EmbeddingNet()
if opt.embedding_model != '':
    embedder.load_state_dict(torch.load(opt.embedding_model))
embedder.cuda()

classifier = FullNet(embedder)
classifier.cuda()

optimizer = optim.Adam(embedder.parameters(), lr=1e-3, weight_decay=1e-5)

print("Loading training data into memory...", )
dataset.load_data()
print("Finished loading training data.")

lossFunc = torch.nn.NLLLoss().cuda()

pos_labels = torch.tensor(np.ones((opt.batch_size, 1)).astype(np.long)).squeeze(1).cuda()
neg_labels = torch.tensor(np.zeros((opt.batch_size, 1)).astype(np.long)).squeeze(1).cuda()
labels = torch.cat([pos_labels, neg_labels], 0)

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

    total_predictions = [0, 0]
    correct_predictions = 0

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

        scores, (x_trans_feat, y_trans_feat), (translation, theta)  = classifier(torch.cat([clouds, clouds], 0), torch.cat([similar_clouds, distant_clouds]))
        predictions = torch.argmax(scores, dim=1).cpu()
        loss = lossFunc(scores, labels)

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
    print('[Epoch %d] Total loss: %f, predictions: (Y: %d, N: %d, Correct: %d)' % (epoch, total_loss, total_predictions[0], total_predictions[1], correct_predictions))
    torch.save(embedder.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print("Completed training for {0} epochs".format(epoch + 1))