from pointnet.model import STNkd
import torch
from torch import nn
import torch.nn.functional as F

class PointNetLC(nn.Module):
    def __init__(self):
        super(PointNetLC, self).__init__();
        self.feat = PointNetfeat2d(global_feat=True, feature_transform=True)
        self.ff = nn.Linear(1024, 1024, True)
        nn.init.xavier_uniform_(self.ff.weight)

    def forward(self, x):
        global_vec, trans, trans_feat = self.feat(x)

        #feed forward layer just to adjust the global vec.
        out_vec = self.ff(global_vec)

        return out_vec, trans, trans_feat


class PointNetfeat2d(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat2d, self).__init__()
        self.stn = STNkd(2)
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
