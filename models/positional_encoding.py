import torch.nn as nn
import torch.nn.functional as fn

# sequential relative

# tree-based


class TreeStructuralEncoding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TreeStructuralEncoding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        b, l, c = x.shape
        assert c == self.in_channels
        return fn.relu(x)
