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
import train_helpers
import time
from train_helpers import print_output
from pointnet.model import feature_transform_regularizer

start_time = str(round(time.time(), 0))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=2, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument(
    '--train_set', type=str, default='train', help='subset of the data to train on. One of [val, dev, train].')
parser.add_argument('--feature_regularization', type=bool, default=True, help='Whether or not to additionally use feature regularization loss')
parser.add_argument('--outf', type=str, default='cls_full_' + start_time, help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--embedding_model', type=str, default='', help='pretrained embedding model to start with')
parser.add_argument('--model', type=str, default='', help='pretrained full model to start with')
parser.add_argument('--distance_cache', type=str, default=None, help='cached overlap info to start with')


opt = parser.parse_args()
train_helpers.initialize_logging(start_time)
print_output(opt)

num_workers = int(opt.workers)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = train_helpers.create_classifier(opt.embedding_model, opt.model)
dataset = train_helpers.load_dataset(opt.dataset, opt.train_set, opt.distance_cache, opt.workers)
classifier.train()

optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-5)

lossFunc = torch.nn.NLLLoss().cuda()

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

    metrics = [0.0, 0.0, 0.0, 0.0] # True Positive, True Negative, False Positive, False Negative

    for i, data in enumerate(dataloader, 0):
        ((clouds, locations, _), (similar_clouds, similar_locs, _), (distant_clouds, distant_locs, _)) = data
        clouds = clouds.transpose(2, 1)
        similar_clouds = similar_clouds.transpose(2, 1)
        distant_clouds = distant_clouds.transpose(2, 1)
        
        clouds = clouds.cuda()
        similar_clouds = similar_clouds.cuda()
        distant_clouds = distant_clouds.cuda()

        optimizer.zero_grad()
        classifier.zero_grad()
        scores, (x_trans_feat, y_trans_feat), (translation, theta) = classifier(torch.cat([clouds, clouds], dim=0), torch.cat([similar_clouds, distant_clouds], dim=0))
        predictions = torch.argmax(scores, dim=1).cpu()
        loss = lossFunc(scores, labels)

        if opt.feature_regularization:
            loss += feature_transform_regularizer(x_trans_feat) * 1e-3
            loss += feature_transform_regularizer(y_trans_feat) * 1e-3

        train_helpers.update_metrics(metrics, predictions, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    acc = (metrics[0] + metrics[1]) / sum(metrics)
    prec = (metrics[0]) / (metrics[0] + metrics[2])
    rec = (metrics[0]) / (metrics[0] + metrics[3])
    print_output('[Epoch %d] Total loss: %f, (Acc: %f, Precision: %f, Recall: %f)' % (epoch, total_loss, acc, prec, rec))
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print_output("Completed training for {0} epochs".format(epoch + 1))

train_helpers.close_logging()