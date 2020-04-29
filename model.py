import torch
import math
from torch import nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from config import data_generation_config

# Predicts a 3-length vector [0:2] are x,y translation
# [2] is theta
class TransformPredictionNetwork(nn.Module):
    def __init__(self):
        super(TransformPredictionNetwork, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 16, 1)
        self.conv2 = torch.nn.Conv1d(16, 32, 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(24)
        self.fc1 = nn.Linear(32, 24)
        self.fc2 = nn.Linear(24, 3)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 32)
        x = self.bn3(self.fc1(x))
        x = self.fc2(x)
        trans = x[:,0:2]
        theta = x[:,2]
        theta = torch.clamp(theta, min=0, max=2 * math.pi)
        return trans, theta

class TransformNet(nn.Module):
    def __init__(self):
        super(TransformNet, self).__init__()
        self.transform_pred = TransformPredictionNetwork()

    def forward(self, x):
        translation, theta = self.transform_pred(x)

        rotations = torch.zeros(x.shape[0], 2, 2).cuda()
        
        c = torch.cos(theta)
        s = torch.sin(theta)

        rotations[:, 0, 0] = c.squeeze()
        rotations[:, 1, 0] = s.squeeze()
        rotations[:, 0, 1] = -s.squeeze()
        rotations[:, 1, 1] = c.squeeze()

        rotated = torch.bmm(rotations, x)
        translations = translation.unsqueeze(2).expand(x.shape)

        transformed = rotated + translations
        return transformed, translation, theta

EMBEDDING_SIZE = 32
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, EMBEDDING_SIZE, 5, 1)
        self.conv2 = torch.nn.Conv1d(EMBEDDING_SIZE, EMBEDDING_SIZE*2, 3, 2)
        self.conv3 = torch.nn.Conv1d(EMBEDDING_SIZE*2, EMBEDDING_SIZE, 3, 1)
        self.conv4 = torch.nn.Conv1d(EMBEDDING_SIZE, EMBEDDING_SIZE, 3)
        self.conv5 = torch.nn.Conv1d(EMBEDDING_SIZE, EMBEDDING_SIZE, 1)
        self.dropout = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm1d(EMBEDDING_SIZE)
        self.bn2 = nn.BatchNorm1d(EMBEDDING_SIZE)
        self.bn3 = nn.BatchNorm1d(EMBEDDING_SIZE)
        self.bn4 = nn.BatchNorm1d(EMBEDDING_SIZE)  
        self.bn5 = nn.BatchNorm1d(EMBEDDING_SIZE)  

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        x = F.max_pool1d(x, x.shape[2])
        x = self.dropout(x)
        return x, None, None

class DistanceNet(nn.Module):
    def __init__(self, embedding=EmbeddingNet()):
        super(DistanceNet, self).__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(0.2)
        self.ff = nn.Linear(32, 1)
        nn.init.xavier_uniform_(self.ff.weight)

    def forward(self, x, y):
        x_emb, x_translation, x_theta = self.embedding(x)
        y_emb, y_translation, y_theta = self.embedding(y)
        dist = F.relu(self.ff(self.dropout(torch.cat([x_emb, y_emb], dim=1))))

        translation = y_translation - x_translation
        theta = y_theta - x_theta

        return dist, translation, theta

class FullNet(nn.Module):
    def __init__(self, embedding=EmbeddingNet()):
        super(FullNet, self).__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(0.4)
        self.ff = nn.Linear(64, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        nn.init.xavier_uniform_(self.ff.weight)

    def forward(self, x, y):
        x_emb, x_translation, x_theta = self.embedding(x)
        y_emb, y_translation, y_theta = self.embedding(y)

        scores = self.ff(self.dropout(torch.cat([x_emb, y_emb], dim=1)))

        out = self.softmax(scores)

        translation = y_translation - x_translation
        theta = y_theta - x_theta

        return out, (translation, theta)

class StructuredEmbeddingNet(nn.Module):
    def __init__(self, threshold=0.75, embedding=EmbeddingNet()):
        super(StructuredEmbeddingNet, self).__init__()
        self.embedding = embedding
        # self.conv = torch.nn.Conv1d(32, 32, 1)
        # self.lstm = torch.nn.LSTM(EMBEDDING_SIZE, 32, batch_first=True)
        self.att_weights = torch.nn.Parameter(torch.Tensor(1, EMBEDDING_SIZE),
                                     requires_grad=True)

        torch.nn.init.xavier_uniform_(self.att_weights.data)

    def forward(self, x, l):
        batch_size, partitions, partition_and_center_size, dims = x.shape
        partition_size = partition_and_center_size - 1
        centers = x[:, :, partition_size:, :]
        c_in = (x[:batch_size, :partitions, :partition_size, :dims] + centers.repeat(1, 1, partition_size, 1)).view(batch_size * partitions, dims, partition_size) 
        c_out = self.embedding(c_in)[0]
        r_in = c_out.view(batch_size, partitions, EMBEDDING_SIZE)

        weights = torch.bmm(r_in,
            self.att_weights  # (1, hidden_size)
            .permute(1, 0)  # (hidden_size, 1)
            .unsqueeze(0)  # (1, hidden_size, 1)
            .repeat(batch_size, 1, 1) # (batch_size, hidden_size, 1)
            )

        attentions = F.softmax(F.relu(weights.squeeze(-1)))
        # create mask based on the sentence lengths
        mask = torch.autograd.Variable(torch.ones(attentions.size())).cuda()
        for i, l in enumerate(l):
            if l < partition_and_center_size - 2:
                mask[i, l:] = 0

         # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(1).expand_as(masked)  # sums per row
        attentions = masked.div(_sums)
            
        # apply weights
        weighted = torch.mul(r_in, attentions.unsqueeze(-1).expand_as(r_in))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze(-1)
        return representations
        # BATCH_SIZE X PARTITION_COUNT X EMBEDDING_SIZE + 2 (last 2 are the "center")
        # spatial = torch.cat((r_in, centers), dim=2)
        # self.lstm.flatten_parameters()
        # _, (h_out, c_out) = self.lstm(r_in)
        # return mp_out

class LCCNet(nn.Module):
    def __init__(self, embedding=EmbeddingNet()):
        super(LCCNet, self).__init__()
        self.embedding = embedding
        # self.conv1 = torch.nn.Conv1d(16, 12, 1)
        # self.conv2 = torch.nn.Conv1d(12, 8, 1)
        # self.bn1 = nn.BatchNorm1d(12)
        # self.bn2 = nn.BatchNorm1d(8)
        self.ff = nn.Linear(16, 2)
        nn.init.xavier_uniform_(self.ff.weight)

    def forward(self, x):
        emb, _, _ = self.embedding(x)
        
        # x = self.bn1(self.conv1(emb.unsqueeze(2)))
        # x = self.bn2(self.conv2(x))
        out = self.ff(emb)

        return out
