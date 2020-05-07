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
from data_processing.dataset import LCLaserDataset
import time
from tqdm import tqdm
import helpers
from helpers import print_output, initialize_logging

from config import training_config, execution_config

parser = argparse.ArgumentParser()
parser.add_argument('--outf', type=str, default='matcher', help='output folder')
parser.add_argument('--model', type=str, default='', help='pretrained scan matcher model to start with')
parser.add_argument('--bag_file', type=str, required=True, help="bag file")
parser.add_argument('--name', type=str, default='laser_dataset', help="bag file")
parser.add_argument('--distance_cache', type=str, default=None, help='cached overlap info to start with')

opt = parser.parse_args()
start_time = str(int(time.time()))
initialize_logging(start_time)

print_output(opt)

num_workers = execution_config['NUM_WORKERS']

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = helpers.load_laser_dataset(opt.bag_file, opt.name)

out_dir = opt.outf + '_' + dataset.name

try:
    os.makedirs(out_dir)
except OSError:
    pass

matcher = helpers.create_scan_matcher(opt.model)
matcher.train()

optimizer = optim.Adam(matcher.parameters(), lr=1e-3, weight_decay=1e-6)
lossFunc = torch.nn.CrossEntropyLoss()

pos_labels = torch.tensor(np.ones((execution_config['BATCH_SIZE'], 1)).astype(np.long)).squeeze(1).cuda()
neg_labels = torch.tensor(np.zeros((execution_config['BATCH_SIZE'], 1)).astype(np.long)).squeeze(1).cuda()
labels = torch.cat([pos_labels, neg_labels], dim=0)

print_output("Press 'return' at any time to finish training after the current epoch.")
for epoch in range(training_config['NUM_EPOCH']):
    total_loss = 0

    batch_count = len(dataset) // execution_config['BATCH_SIZE']
    print_output("Loaded new training data: {0} batches of size {1}".format(batch_count, execution_config['BATCH_SIZE']))
    # dataset.load_data()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=execution_config['BATCH_SIZE'],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)

    total = 0
    correct = 0

    for i, data in tqdm(enumerate(dataloader, 0)):
        ((clouds, locations, _), (alt_clouds, alt_locs, _), labels) = data
        clouds = clouds.cuda()
        alt_clouds = alt_clouds.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        matcher.zero_grad()

        scores = matcher(clouds, alt_clouds)
        predictions = torch.argmax(F.softmax(scores), dim=1)
        loss = lossFunc.forward(scores, labels)

        correct += torch.sum(predictions == labels)
        total += len(labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print_output('[Epoch %d] Total loss: %f' % (epoch, total_loss))
    print_output('Correct: {0} / {1} = {2}'.format(correct, total, float(correct) / total))
    helpers.save_model(matcher, out_dir, epoch)
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print_output("Completed training for {0} epochs".format(epoch + 1))