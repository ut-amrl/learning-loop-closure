from model import FullNet, EmbeddingNet
from dataset import LCTripletDataset
import time
import torch

log_file = None
def initialize_logging(start_time, file_prefix='train_'):
    global log_file
    log_file = open('./logs/' + file_prefix + start_time + '.log', 'w+')

def print_output(*args):
    print(args)
    if log_file:
        log_file.write(' '.join([str(a) for a in args]) + '\n')
        log_file.flush()
    else:
        print("warning: log file not initialized. Please call initialize_logging")

def close_logging():
    global log_file
    log_file.close()
    log_file = None

def load_dataset(root, split, distance_cache, num_workers):
    print_output("Loading data into memory...", )
    dataset = LCTripletDataset(
        root=root,
        split=split,
        num_workers=num_workers)
    dataset.load_data()
    dataset.load_distances(distance_cache)
    dataset.load_triplets()
    dataset.cache_distances()
    print_output("Finished loading data.")
    return dataset

def create_classifier(embedding_model='', model=''):
    embedder = EmbeddingNet()
    if embedding_model != '':
        embedder.load_state_dict(torch.load(embedding_model))
    classifier = FullNet(embedder)

    if model != '':
        classifier.load_state_dict(torch.load(model))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        classifier = torch.nn.DataParallel(classifier)

    classifier.cuda()
    return classifier

def save_classifier(classifier, outf, epoch):
    to_save = classifier
    if torch.cuda.device_count() > 1:
        to_save = classifier.module
    torch.save(to_save.state_dict(), '%s/cls_model_%d.pth' % (outf, epoch))

def update_metrics(metrics, predictions, labels):
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
