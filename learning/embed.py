import argparse
import torch
import json
import numpy as np
import os
from model import PointNetLC

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', type=str, help='path to pretrained model')
parser.add_argument(
    '--data_path', type=str, help='path to json file containing file paths to embed')
parser.add_argument(
    '--out_path', type=str,  help='path to output embedding files')

opt = parser.parse_args()

model = PointNetLC()
model.load_state_dict(torch.load(opt.model))
model.eval()

with open(opt.data_path) as f:
    files = json.loads(f.read())

for f in files:
    fname = os.path.basename(f)
    point_set = np.loadtxt(f).astype(np.float32)
    point_set = torch.tensor([point_set])
    point_set = point_set.transpose(2, 1)
    embedding, trans, _ = model.forward(point_set)
    embedding_out_path = os.path.join(opt.out_path, fname) + '.embedding'
    np.savetxt(embedding_out_path, embedding.detach().numpy())