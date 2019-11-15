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
    '--data_path', type=str, help='path to file to embed')
parser.add_argument(
    '--out_path', type=str,  help='path to output embedding file')

opt = parser.parse_args()

model = PointNetLC()
model.load_state_dict(torch.load(opt.model))
model.eval()

f = opt.data_path
fname = os.path.basename(f)
point_set = np.loadtxt(fname).astype(np.float32)
point_set = torch.tensor([point_set])
point_set = point_set.transpose(2, 1)
embedding, trans, _ = model.forward(point_set)
np.savetxt(opt.out_path, embedding.detach().numpy())
