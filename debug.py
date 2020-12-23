from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.data import DataLoader

from data.dataset2 import SkeletonDataset
from models.net import DualGraphTransformer

ds = SkeletonDataset(root='dataset/ntu_60',
                     name='ntu')
loader = DataLoader(ds, batch_size=4)
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

lt = DualGraphTransformer(in_channels=7,
                          hidden_channels=16,
                          out_channels=16,
                          sequential=False,
                          num_layers=3)
loss_fn = CrossEntropyLoss()
optimizer = Adam(lt.parameters(),
                 lr=1e-3,
                 weight_decay=1e-2,
                 betas=(0.9, 0.98))
running_loss = 0.0
show_parameter = False
for bi, b in enumerate([next(iter(loader))] * 50):
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

print(lt.context_attention.weights)
