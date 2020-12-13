from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as fn
from .layers import HGAConv
from third_party.performer import SelfAttention
from einops import rearrange


class DualGraphTransformer(nn.Module, ABC):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 num_heads=8,
                 num_joints=25,
                 classes=60,
                 drop_rate=0.5,
                 sequential=True,
                 trainable_factor=True):
        super(DualGraphTransformer, self).__init__()
        self.spatial_factor = nn.Parameter(torch.ones(num_layers)) * 0.5
        self.sequential = sequential
        self.num_layers = num_layers
        self.trainable_factor = trainable_factor
        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.spatial_layers = nn.ModuleList([
            HGAConv(in_channels=channels[i],
                    heads=num_heads,
                    dropout=drop_rate,
                    out_channels=channels[i + 1]) for i in range(num_layers)])
        channels_ = channels[1:] + [out_channels]
        self.temporal_layers = nn.ModuleList([
            # necessary parameters are: dim
            SelfAttention(dim=channels_[i],  # TODO ??? potential dimension problem
                          heads=num_heads,
                          dropout=drop_rate,
                          causal=True) for i in range(num_layers)])
        if not self.sequential:
            self.temporal_lls = nn.ModuleList([nn.Linear(in_features=channels_[i],
                                                         out_features=channels_[i + 1])
                                               for i in range(num_layers)])
        self.bottle_neck = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.final_layer = nn.Linear(in_features=out_channels * num_joints, out_features=classes)

    def forward(self, t, adj):  # t: tensor, adj: dataset.skeleton_
        if self.sequential:     # sequential architecture
            for i in range(self.num_layers):
                t = rearrange(fn.relu(self.spatial_layers[i](t, adj)),
                              'b n c -> n b c')
                t = rearrange(fn.relu(fn.layer_norm(self.temporal_layers[i](t),
                                                    t.shape[1:]) + t),
                              'n b c -> b n c')
        else:  # parallel architecture
            for i in range(self.num_layers):
                s = t
                t = self.temporal_lls[i](rearrange(t, 'b n c -> n b c'))
                s = fn.relu(self.spatial_layers[i](s, adj))
                t = fn.relu(fn.layer_norm(self.temporal_layers[i](t),
                                          t.shape[1:]) + t)
                if self.trainable_factor:
                    factor = fn.sigmoid(self.spatial_factor).to(t.device)
                    t = factor[i] * s + (1. - factor[i]) * rearrange(t, 'n b c -> b n c')
                else:
                    t = (s + rearrange(t, 'n b c -> b n c')) * 0.5
        t = rearrange(self.bottle_neck(t), 'b n c -> b (n c)')
        t = self.final_layer(t)
        # return fn.sigmoid(t)  # dimension (b, n, oc)
        return t
