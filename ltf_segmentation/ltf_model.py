import torch
from torch import nn

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.transform = TransformNet()
        self.conv1 = torch.nn.Conv1d(2, 8, 1)
        self.conv2 = torch.nn.Conv1d(8, 16, 1)
        self.conv3 = torch.nn.Conv1d(16, 32, 1)
        self.conv4 = torch.nn.Conv1d(32, 32, 1)
        self.dropout = nn.Dropout(0.35)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        x, translation, theta = self.transform(x)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        x = F.max_pool1d(x, x.shape[2])
        x = F.relu(x)
        x = self.dropout(x)
        return x, translation, theta