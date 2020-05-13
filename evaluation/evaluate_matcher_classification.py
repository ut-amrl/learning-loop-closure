import argparse
import os
import select
import sys
import torch
import torch.utils.data
import numpy as np
import pickle
import time
import random
from tqdm import tqdm
sys.path.append(os.path.join(os.getcwd(), '..'))
import helpers
from helpers import initialize_logging, print_output
from config import Configuration, execution_config, evaluation_config

config = Configuration(False, True).parse()

start_time = str(int(time.time()))
initialize_logging(start_time, 'evaluate_')
print_output(config)

num_workers = int(execution_config['NUM_WORKERS'])

config.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", config.manualSeed)
random.seed(config.manualSeed)
torch.manual_seed(config.manualSeed)

scan_conv, scan_match, scan_transform = helpers.create_laser_networks(config.model_dir, config.model_epoch)
scan_conv.eval()
scan_match.eval()
dataset = helpers.load_laser_dataset(config.bag_file, '', config.dist_close_ratio)
batch_count = len(dataset) // config.batch_size
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)

metrics = np.zeros((len(thresholds), 4)) # True Positive, True Negative, False Positive, False Negative

print("Evaluation over {0} batches of size {1}".format(batch_count, config.batch_size))

for i, data in tqdm(enumerate(dataloader, 0)):
    ((clouds, locations, _), (alt_clouds, alt_locs, _), labels) = data
    clouds = clouds.cuda()
    alt_clouds = alt_clouds.cuda()
    labels = labels.cuda()
    
    conv = scan_conv(clouds, alt_clouds)

    # import pdb; pdb.set_trace()
    #Compute match prediction
    scores = scan_match(conv)
    predictions = torch.argmax(F.softmax(scores), dim=1)
    pos_predictions = predictions.nonzero()
    import pdb; pdb.set_trace()
    metrics[0] += (pos_predictions == labels)
    metrics[2] += (pos_predictions != labels)

    metrics[1] += (pos_predictions == labels)
    metrics[3] += (pos_predictions != labels)


confusion = np.zeros((len(thresholds), 2, 2))
for i in range(len(thresholds)):
    
    confusions[i] = [[threshold_metrics[0], threshold_metrics[2]], [threshold_metrics[3], threshold_metrics[1]]]

import matplotlib as mpl
if opt.no_vis:
    mpl.use('Agg')
from matplotlib import pyplot as plt

plt.plot(roc[:, 2], roc[:, 1], color='r', label="Threshold")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

if not opt.no_vis:
    plt.show()
else:
    plt.savefig('precision_recall_curve.png')

if opt.publish_triplets:
    name = os.path.basename(opt.dataset)
    print("Writing triplets_{0}.npy".format(name))
    np.save('triplets_{0}'.format(name), triplets)
    np.save('confusions_{0}'.format(name), confusions)
