import torch
from data.dataset import SkeletonDataset
from torch_geometric.data import DataLoader
from models.layers import HGAConv
from third_party.performer import SelfAttention
from einops import rearrange
from models.net import DualGraphTransformer


ds = SkeletonDataset(root='dataset',
                     name='ntu')
loader = DataLoader(ds, batch_size=4)
b = next(iter(loader))
# ly = HGAConv(in_channels=7,
#              out_channels=16,
#              heads=8)
# t = ly(b.x, adj=ds.skeleton_)
#
# t = rearrange(t, 'b n c -> n b c')
# h = 4  # num_heads
# b, n, c = t.shape
# lt = SelfAttention(dim=c,
#                    heads=h,
#                    causal=True)

lt = DualGraphTransformer(in_channels=7,
                          hidden_channels=16,
                          out_channels=16,
                          num_layers=3)

t = lt(b.x, ds.skeleton_)
print(t.shape)
