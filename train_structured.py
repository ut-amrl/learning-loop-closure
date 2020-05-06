import argparse
import os
import select
import sys
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import EmbeddingNet
from data_processing.dataset import LCDataset, LCTripletDataset, MergedDataset
import time
from tqdm import tqdm
import helpers
from helpers import print_output, initialize_logging, update_metrics

from config import training_config, execution_config, data_config

class TripletLoss(torch.nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, similar_thresh):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.threshold = similar_thresh

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        losses = F.relu(distance_positive - distance_negative + self.margin) + F.relu(distance_positive - self.threshold)
        return losses.sum()

parser = argparse.ArgumentParser()
parser.add_argument('--outf', type=str, default='cls_structured', help='output folder')
parser.add_argument('--embedding_model', type=str, default='', help='pretrained embedding model to start with')
parser.add_argument('--datasets', nargs='+', required=True, help="dataset paths")
parser.add_argument('--distance_cache', type=str, default=None, help='cached overlap info to start with')
parser.add_argument('--exhaustive', type=bool, default=False, help='Whether or not to check the exhaustive list of all triplets')

opt = parser.parse_args()
start_time = str(int(time.time()))
initialize_logging(start_time)

print_output(opt)

num_workers = execution_config['NUM_WORKERS']

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

datasets = []
val_datasets = []
name = ''
for dataset in opt.datasets:
    ds = helpers.load_structured_dataset(dataset, training_config['TRAIN_SET'], opt.distance_cache, opt.exhaustive)
    vs = helpers.load_structured_dataset(dataset, training_config['VALIDATION_SET'], opt.distance_cache, opt.exhaustive)
    name += ds.dataset_info['name'] + '_' + ds.split + '_'
    datasets.append(ds)
    val_datasets.append(vs)

merged_dataset = MergedDataset(datasets, name)
val_merged_dataset = MergedDataset(val_datasets, name)

out_dir = opt.outf + '_' + merged_dataset.name

try:
    os.makedirs(out_dir)
except OSError:
    pass

embedder = helpers.create_structured_embedder(opt.embedding_model)
embedder.train()

optimizer = optim.Adam(embedder.parameters(), lr=1e-3, weight_decay=1e-6)
lossFunc = TripletLoss(training_config['MARGIN'], training_config['SIMILAR_THRESHOLD'])

print_output("Press 'return' at any time to finish training after the current epoch.")

pos_labels = torch.tensor(np.ones((execution_config['BATCH_SIZE'] * data_config['MATCH_REPEAT_FACTOR'], 1)).astype(np.long)).squeeze(1).cuda()
neg_labels = torch.tensor(np.zeros((execution_config['BATCH_SIZE'] * data_config['MATCH_REPEAT_FACTOR'], 1)).astype(np.long)).squeeze(1).cuda()
labels = torch.cat([pos_labels, neg_labels], dim=0)

for epoch in range(training_config['NUM_EPOCH']):
    total_loss = 0

    # We want to reload the triplets every 5 epochs to get new matches
    if opt.exhaustive:
        merged_dataset.load_all_triplets()
    else:
        merged_dataset.load_triplets()
    batch_count = len(merged_dataset) // execution_config['BATCH_SIZE']
    print_output("Loaded new training triplets: {0} batches of size {1}".format(batch_count, execution_config['BATCH_SIZE']))
    dataloader = torch.utils.data.DataLoader(
        merged_dataset,
        batch_size=execution_config['BATCH_SIZE'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)

    total_similar_dist = 0.0
    total_distant_dist = 0.0
    metrics = torch.zeros(4)

    for i, data in tqdm(enumerate(dataloader, 0)):
        (((clouds, length), locations, _), ((similar_clouds, similar_length), similar_locs, _), ((distant_clouds, distant_length), distant_locs, _)) = data
        
        clouds = clouds.cuda()
        similar_clouds = similar_clouds.cuda()
        distant_clouds = distant_clouds.cuda()
        length = length.cuda()
        similar_length = similar_length.cuda()
        distant_length = distant_length.cuda()

        optimizer.zero_grad()
        embedder.zero_grad()

        anchor_embeddings = embedder(clouds, length)
        similar_embeddings = embedder(similar_clouds, similar_length)
        distant_embeddings = embedder(distant_clouds, distant_length)

        distance_pos = torch.norm(anchor_embeddings - similar_embeddings, p=2, dim=1)
        distance_neg = torch.norm(anchor_embeddings - distant_embeddings, p=2, dim=1)
 
        anchor_matches = (distance_pos < training_config['SIMILAR_THRESHOLD']).int()
        distant_matches = (distance_neg < training_config['SIMILAR_THRESHOLD']).int()

        update_metrics(metrics, torch.cat((anchor_matches, distant_matches)), labels)

        total_similar_dist += torch.sum(distance_pos) 
        total_distant_dist += torch.sum(distance_neg) 

        # Compute loss here
        loss = lossFunc.forward(anchor_embeddings, similar_embeddings, distant_embeddings)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_similar_dist = total_similar_dist / (batch_count * execution_config['BATCH_SIZE'])
    avg_distant_dist = total_distant_dist / (batch_count * execution_config['BATCH_SIZE'])

    print_output('[Epoch %d] Total loss: %f' % (epoch, total_loss))
    print_output('Average distance, Similar: {0} Distant: {1}'.format(avg_similar_dist, avg_distant_dist))
    
    acc = (metrics[0] + metrics[1]) / sum(metrics)
    precision = (metrics[0]) / (metrics[0] + metrics[2])
    recall = (metrics[0]) / (metrics[0] + metrics[3])
    f1 = 2 * precision * recall / (precision + recall)
    print_output('Metrics: Acc {0}, Prec {1}, Recall {2}, F1 {3}'.format(acc, precision, recall, f1))

    if epoch % 5 == 0:
        with torch.no_grad():
            val_metrics = np.zeros(4)
            val_dataloader = torch.utils.data.DataLoader(
                val_merged_dataset,
                batch_size=execution_config['BATCH_SIZE'],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True)

            for i, data in tqdm(enumerate(val_dataloader, 0)):
                ((clouds, locations, timestamp), (similar_clouds, similar_locs, similar_timestamp), (distant_clouds, distant_locs, distant_timestamp)) = data

                clouds[0] = clouds[0].cuda()
                similar_clouds[0] = similar_clouds[0].cuda()
                distant_clouds[0] = distant_clouds[0].cuda()

                clouds[1] = clouds[1].cuda()
                similar_clouds[1] = similar_clouds[1].cuda()
                distant_clouds[1] = distant_clouds[1].cuda()
                
                distances = helpers.get_distances_for_model(embedder, clouds, similar_clouds, distant_clouds)
                predictions = (distances < training_config['SIMILAR_THRESHOLD']).int()
                helpers.update_metrics(val_metrics, predictions, labels)
            val_acc = (val_metrics[0] + val_metrics[1]) / sum(val_metrics)
            val_precision = (val_metrics[0]) / (val_metrics[0] + val_metrics[2])
            val_recall = (val_metrics[0]) / (val_metrics[0] + val_metrics[3])
            val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall)
            print_output('Validation Metrics: val_acc {0}, Prec {1}, val_recall {2}, val_f1 {3}'.format(val_acc, val_precision, val_recall, val_f1))

    helpers.save_model(embedder, out_dir, epoch)
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print_output("Completed training for {0} epochs".format(epoch + 1))
