from model import FullNet, EmbeddingNet, LCCNet
from dataset import LCTripletDataset, LCCDataset
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

def load_dataset(root, split, distance_cache, augment_prob=0.1):
    print_output("Loading data into memory...", )
    dataset = LCTripletDataset(
        root=root,
        split=split,
        augmentation_prob=augment_prob)
    dataset.load_data()
    dataset.load_distances(distance_cache)
    dataset.load_triplets()
    if dataset.computed_new_distances:
        dataset.cache_distances()
    print_output("Finished loading data.")
    return dataset

def load_lcc_dataset(root):
    print_output("Loading data into memory...")
    dataset = LCCDataset(root=root)
    dataset.load_data()
    print_output("Finished loading data.")
    return dataset

def create_embedder(embedding_model=''):
    embedder = EmbeddingNet()
    if embedding_model != '':
        embedder.load_state_dict(torch.load(embedding_model))
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        embedder = torch.nn.DataParallel(embedder)

    embedder.cuda()
    return embedder

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

def create_lcc(model=''):
    lcc = LCCNet()
    if model != '':
        lcc.load_state_dict(torch.load(model))
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        lcc = torch.nn.DataParallel(lcc)
    
    lcc.cuda()
    return lcc

def save_model(model, outf, epoch):
    to_save = model
    if torch.cuda.device_count() > 1:
        to_save = model.module
    torch.save(to_save.state_dict(), '%s/model_%d.pth' % (outf, epoch))

def get_predictions_for_model(model, clouds, similar, distant, threshold=2):
    model_type = None
    model_to_check = model
    if isinstance(model, torch.nn.DataParallel):
        model_to_check = model.module
    
    if isinstance(model_to_check, EmbeddingNet):
        model_type = "embedder"
    elif isinstance(model_to_check, FullNet):
        model_type = "full"
    else:
        raise Exception('Unexpected model', model_to_check)


    if model_type == 'embedder':
        anchor_embeddings, _, _ = model(clouds)
        similar_embeddings, _, _ = model(similar)
        distant_embeddings, _, _ = model(distant)
        
        distance_pos = torch.norm(anchor_embeddings - similar_embeddings, p=2, dim=1)
        distance_neg = torch.norm(anchor_embeddings - distant_embeddings, p=2, dim=1)

        predictions_pos = (distance_pos < threshold).int()
        predictions_neg = (distance_neg < threshold).int()

        predictions = torch.cat([predictions_pos, predictions_neg])
        return predictions
    elif model_type == 'full':
        scores, _, _ = model(torch.cat([clouds, clouds], dim=0), torch.cat([similar, distant], dim=0))
        predictions = torch.argmax(scores, dim=1).cpu()
        
        return predictions

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
