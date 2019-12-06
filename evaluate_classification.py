import argparse
import os
import select
import sys
import torch
import torch.utils.data
import numpy as np
import pickle
from learning.model import FullNet, EmbeddingNet
from learning.dataset import LCDataset, LCTripletDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=2, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--evaluation_set', type=str, default='dev', help='subset of the data to train on. One of [val, dev, train].')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--model', type=str, default='', help='model to evaluate');
parser.add_argument('--cached_dataset', type=str, default='', help='cached LCTripletDataset to start with')

opt = parser.parse_args()

with torch.no_grad():
    classifier = FullNet()
    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))
    classifier.cuda()
    print("Loading evaluation data into memory...", )
    if opt.cached_dataset != '':
        with open(opt.cached_dataset, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = LCTripletDataset(
            root=opt.dataset,
            split=opt.evaluation_set)
        dataset.load_data()
        dataset.load_triplets()
        with open('evaluation_full_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
    batch_count = len(dataset) // opt.batch_size
    print("Loaded evaluation triplets: {0} batches of size {1}".format(batch_count, opt.batch_size))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        drop_last=True)

    pos_labels = torch.tensor(np.ones((opt.batch_size, 1)).astype(np.long)).squeeze(1)
    neg_labels = torch.tensor(np.zeros((opt.batch_size, 1)).astype(np.long)).squeeze(1)
    labels = torch.cat([pos_labels, neg_labels], 0).cuda()

    metrics = [0.0, 0.0, 0.0, 0.0] # True Positive, True Negative, False Positive, False Negative

    for i, data in enumerate(dataloader, 0):
        ((clouds, locations, _), (similar_clouds, similar_locs, _), (distant_clouds, distant_locs, _)) = data
        clouds = clouds.transpose(2, 1)
        similar_clouds = similar_clouds.transpose(2, 1)
        distant_clouds = distant_clouds.transpose(2, 1)
        
        clouds = clouds.cuda()
        similar_clouds = similar_clouds.cuda()
        distant_clouds = distant_clouds.cuda()

        scores, (x_trans_feat, y_trans_feat), (translation, theta)  = classifier(torch.cat([clouds, clouds], 0), torch.cat([similar_clouds, distant_clouds]))
        predictions = torch.argmax(scores, dim=1).cpu()

        for i in range(len(predictions)):
            label = labels[i].item()
            prediction = predictions[i].item()
            if label and prediction:
                metrics[0] += 1 # True Positive
            elif not label and not prediction:
                metrics[1] += 1 # True Negative
            elif not label and prediction:
                metrics[2] += 1 # False Positive
            elif label and not prediction:
                metrics[3] += 1 # False Negative

    acc = (metrics[0] + metrics[1]) / sum(metrics)
    prec = (metrics[0]) / (metrics[0] + metrics[2])
    rec = (metrics[0]) / (metrics[0] + metrics[3])
    print('(Acc: %f, Precision: %f, Recall: %f)' % (acc, prec, rec))