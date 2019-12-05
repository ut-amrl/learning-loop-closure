import argparse
import torch
import json
import numpy as np
import os
from model import EmbeddingNet
from dataset import get_point_cloud_from_file

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', type=str, help='path to pretrained model')
parser.add_argument(
    '--data_path', type=str, help='path to file to embed')
parser.add_argument(
    '--out_path', type=str,  help='path to output embedding file')

opt = parser.parse_args()

model = EmbeddingNet()
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
with torch.no_grad():
    f = opt.data_path
    point_set = torch.tensor([get_point_cloud_from_file(f)]).cuda()
    point_set = point_set.transpose(2, 1)
    embedding, _, trans, theta = model.forward(point_set)
    np.savetxt(opt.out_path, embedding.detach().cpu().numpy())
