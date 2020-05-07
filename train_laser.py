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
parser.add_argument('--conv_model', type=str, default='', help='pretrained scan conv model to start with')
parser.add_argument('--match_model', type=str, default='', help='pretrained scan matcher model to start with')
parser.add_argument('--transform_model', type=str, default='', help='pretrained scan transform model to start with')
parser.add_argument('--bag_file', type=str, required=True, help="bag file")
parser.add_argument('--name', type=str, default='laser_dataset', help="bag file")
parser.add_argument('--distance_cache', type=str, default=None, help='cached overlap info to start with')

parser.add_argument('--train_transform', action='store_true')
parser.add_argument('--lock_conv', action='store_true')

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

scan_conv, scan_match, scan_transform = helpers.create_laser_networks(opt.conv_model, opt.match_model, opt.transform_model)
scan_conv.train()
scan_match.train()
scan_transform.train()

conv_optimizer = optim.Adam(scan_conv.parameters(), lr=1e-3, weight_decay=1e-6)
match_optimizer = optim.Adam(scan_match.parameters(), lr=1e-3, weight_decay=1e-6)
transform_optimizer = optim.Adam(scan_transform.parameters(), lr=1e-3, weight_decay=1e-6)
matchLossFunc = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([40.0, 1.0]).cuda())
transLossFunc = torch.nn.MSELoss()

print_output("Press 'return' at any time to finish training after the current epoch.")
for epoch in range(training_config['NUM_EPOCH']):
    total_loss = 0

    batch_count = len(dataset) // execution_config['BATCH_SIZE']
    print_output("Loaded new training data: {0} batches of size {1}".format(batch_count, execution_config['BATCH_SIZE']))
    dataset.load_data()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=execution_config['BATCH_SIZE'],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)

    total = 0
    correct = 0
    fp = 0
    fn = 0

    for i, data in tqdm(enumerate(dataloader, 0)):
        ((clouds, locations, _), (alt_clouds, alt_locs, _), labels) = data
        clouds = clouds.cuda()
        alt_clouds = alt_clouds.cuda()
        labels = labels.cuda()

        conv_optimizer.zero_grad()
        match_optimizer.zero_grad()
        transform_optimizer.zero_grad()
        scan_conv.zero_grad()
        scan_match.zero_grad()
        scan_transform.zero_grad()

        conv = scan_conv(clouds, alt_clouds)

        # import pdb; pdb.set_trace()
        #Compute match prediction
        scores = scan_match(conv)
        predictions = torch.argmax(F.softmax(scores), dim=1)
        loss = training_config['LASER_MATCH_WEIGHT'] * matchLossFunc.forward(scores, labels)

        if opt.train_transform:
            # Compute transforms, but only for things that *should* match
            match_indices = labels.nonzero()
            filtered = conv[match_indices]
            transforms = scan_transform(filtered)
            true_transforms = (locations - alt_locs)[match_indices]
            # Clamp angle between -pi and pi
            true_transforms[:, :, 2] = torch.fmod(2 * np.pi + true_transforms[:, :, 2], 2 * np.pi) - np.pi
            loss += training_config['LASER_TRANS_WEIGHT'] * transLossFunc.forward(transforms, true_transforms.squeeze(1).cuda())

        correct += torch.sum(predictions == labels)
        fp += torch.sum(predictions > labels)
        fn += torch.sum(predictions < labels)
        total += len(labels)

        loss.backward()
        if not opt.lock_conv:
            conv_optimizer.step()
        match_optimizer.step()
        transform_optimizer.step()
        total_loss += loss.item()
    
    print_output('[Epoch %d] Total loss: %f' % (epoch, total_loss))
    print_output('Correct: {0} / {1} = {2}; FP: {3}; FN: {4}'.format(correct, total, float(correct) / total, fp, fn))
    helpers.save_model(scan_conv, out_dir, epoch, 'conv')
    helpers.save_model(scan_match, out_dir, epoch, 'match')
    helpers.save_model(scan_transform, out_dir, epoch, 'transform')
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print_output("Completed training for {0} epochs".format(epoch + 1))