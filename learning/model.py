from pointnet.model import STNkd
import torch
from torch import nn
import torch.nn.functional as F

class S2Net(nn.Module):
    def __init__(self):
        super(S2Net, self).__init__()
        self.translate = torch.zeros((2, 1))
        self.theta = Variable(0)

    # TODO handle batches
    def forward(self, x):
        # construct rotation
        s = torch.sin(self.theta)
        c = torch.cos(self.theta)
        rotation = torch.tensor([
            [c, s],
            [-s, c]
        ])
        return torch.mm(rotation, x) + self.translate

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.transform = S2Net()
        self.conv1 = torch.nn.Conv1d(2, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 128, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(128)
        self.fstn = STNkd(k=32)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = self.bn2(self.conv2(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)
        return x, trans, trans_feat

class FullNet(nn.Module):
    def __init__(self):
        super(FullNet, self).__init__()
        self.embedding = EmbeddingNet()
        self.ff = nn.Linear(256, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, y):
        x_emb, x_trans, _ = self.embedding(x)
        y_emb, y_trans, _ = self.embedding(y)

        scores = self.ff(torch.cat(x_emb, y_emb))

        out = self.LogSoftmax(scores)

        out, x_trans, y_trans