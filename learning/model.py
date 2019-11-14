from pointnet.model import PointNetfeat
from torch import nn

class PointNetLC(nn.Module):
    def __init__(self):
        super(PointNetLC, self).__init__();
        self.feat = PointNetfeat(global_feat=True, feature_transform=True)
        self.ff = nn.Linear(1024, 1024, True)
        nn.init.xavier_uniform_(self.ff.weight)

    def forward(self, x):
        global_vec, trans, trans_feat = self.feat(x)

        #feed forward layer just to adjust the global vec.
        out_vec = self.ff(global_vec)

        return out_vec, trans, trans_feat
