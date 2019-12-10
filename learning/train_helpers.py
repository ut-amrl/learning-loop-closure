from model import FullNet, EmbeddingNet
from dataset import LCTripletDataset
import time

log_file = None
def initialize_logging(start_time):
    log_file = open('./logs/train_' + start_time + '.log', 'w+')

def print_output(string):
    print(string)
    if log_file:
        log_file.write(str(string) + '\n')
        log_file.flush()
    else:
        print("warning: log file not initialized. Please call initialize_logging")

def load_dataset(root, split, distance_cache):
    print_output("Loading training data into memory...", )
    dataset = LCTripletDataset(
        root=root,
        split=split,
        num_workers=num_workers)
    dataset.load_data()
    dataset.load_distances(distance_cache)
    dataset.load_triplets()
    dataset.cache_distances()
    print_output("Finished loading training data.")
    return dataset

def create_classifier(embedding_model='', model=''):
    embedder = EmbeddingNet()
    if opt.embedding_model != '':
        embedder.load_state_dict(torch.load(opt.embedding_model))
    classifier = FullNet(embedder)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        classifier = torch.nn.DataParallel(classifier)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))
    classifier.cuda()
    return classifier

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