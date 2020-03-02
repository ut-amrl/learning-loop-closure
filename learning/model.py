import torch
import math
from torch import nn
import torch.nn.functional as F

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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 32)
        x = F.relu(self.bn3(self.fc1(x)))
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

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.transform = TransformNet()
        self.conv1 = torch.nn.Conv1d(2, 8, 1)
        self.conv2 = torch.nn.Conv1d(8, 16, 1)
        self.conv3 = torch.nn.Conv1d(16, 32, 1)
        self.conv4 = torch.nn.Conv1d(32, 32, 1)
        self.dropout = nn.Dropout(0.5)
        self.ff = nn.Linear(32, 16)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        nn.init.xavier_uniform_(self.ff.weight)

    def forward(self, x):
        x, translation, theta = self.transform(x)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        x = F.max_pool1d(x, x.shape[2])
        x = F.relu(x)
        x = self.dropout(x)
        x = self.ff(x.transpose(2, 1)).transpose(2, 1).squeeze(-1)
        return x, translation, theta

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

class LCCNet(nn.Module):
    def __init__(self, embedding=EmbeddingNet()):
        super(LCCNet, self).__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 12)
        self.fc3 = nn.Linear(12, 2)
        self.softmax = nn.LogSoftmax(dim=1)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        emb, translation, theta = self.embedding(x)
        
        scores = self.fc3(F.relu(self.fc2(F.relu(self.fc1(self.dropout(emb))))))
        out = self.softmax(scores)

        return out, translation, theta
