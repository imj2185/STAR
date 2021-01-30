import os
import os.path as osp
from argparse import ArgumentParser

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torchviz import make_dot
from tqdm import tqdm

from data.dataset3 import SkeletonDataset
from models.net import DualGraphTransformer

parser = ArgumentParser()
parser.add_argument('--root', dest='root',
                    default=osp.join(os.getcwd(), 'dataset', 'ntu_60'),
                    type=str, help='Dataset')
parser.add_argument('--dataset', dest='dataset', default='ntu_60',
                    type=str, help='Dataset')
args = parser.parse_args()

ds = SkeletonDataset(root=args.root,
                     name=args.dataset,
                     benchmark='xsub',
                     sample='val')

loader = DataLoader(ds, batch_size=8, shuffle=True)
c = ds[0].x.shape[-1]
# b = next(iter(loader))
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

lt = DualGraphTransformer(in_channels=c,
                          hidden_channels=16,
                          out_channels=16,
                          sequential=False,
                          num_layers=3)

b = next(iter(loader))

make_dot(lt(b.x, ds.skeleton_, b.batch), params=dict(lt.named_parameters()))

loss_fn = CrossEntropyLoss()
optimizer = Adam(lt.parameters(),
                 lr=1e-3,
                 weight_decay=1e-2,
                 betas=(0.9, 0.98))
running_loss = 0.0
show_parameter = False
total = 0
correct = 0

print('start training')
pbar = tqdm(enumerate([next(iter(loader))] * 200))
with torch.autograd.set_detect_anomaly(True):
    for bi, b in pbar:
        pbar.set_description("processing %d" % (bi + 1))
        lt.train()
        t, y = lt(b.x, ds.skeleton_, b.batch), b.y
        loss = loss_fn(t, y)
        optimizer.zero_grad()
        loss.backward()
        # check out whether parameter is updated?
        if show_parameter:
            for p in lt.parameters():
                print(p.grad.data.sum())

        optimizer.step()
        running_loss += loss.item()
        if bi % 10 == 9:
            print('[%5d] loss: %.3f' %
                  (bi + 1, running_loss / 10))
            running_loss = 0.0
            lt.eval()
            with torch.no_grad():
                _, predicted = torch.max(t.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            print('\n\t accuracy: %d %%' % (100 * correct / total))

# print(lt.context_attention.weights)
