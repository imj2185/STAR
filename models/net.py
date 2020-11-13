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
        self.spatial_factor = nn.Parameter(torch.ones(1)) * 0.5
        self.sequential = sequential
        self.num_layers = num_layers
        self.trainable_factor = trainable_factor
        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.spatial_layers = nn.ModuleList([
            HGAConv(in_channels=channels[i],
                    heads=num_heads,
                    dropout=drop_rate,
                    out_channels=channels[i + 1]) for i in range(num_layers)
        ])
        channels_ = channels[1:] + [out_channels]
        self.temporal_layers = nn.ModuleList([
            # necessary parameters are: dim
            SelfAttention(dim=channels_[i],  # TODO ??? potential dimension problem
                          heads=num_heads,
                          dropout=drop_rate,
                          causal=True) for i in range(num_layers)
        ])
        if not self.sequential:
            self.temporal_in = nn.Linear(in_features=in_channels, out_features=hidden_channels)
        self.bottle_neck = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.final_layer = nn.Linear(in_features=out_channels * num_joints, out_features=classes)

    def forward(self, t, adj):  # adj=dataset.skeleton_
        if self.sequential:  # sequential architecture
            for i in range(self.num_layers):
                t = rearrange(fn.relu(self.spatial_layers[i](t, adj)),
                              'b n c -> n b c')
                t = rearrange(fn.relu(fn.layer_norm(self.temporal_layers[i](t),
                                                    t.shape[1:]) + t),
                              'n b c -> b n c')
        else:  # parallel architecture
            s = t
            t = self.temporal_in(rearrange(t, 'b n c -> n b c'))

            for i in range(self.num_layers):
                s = fn.relu(self.spatial_layers[i](s, adj))
                t = fn.relu(fn.layer_norm(self.temporal_layers[i](t),
                                          t.shape[1:]) + t)
            if self.trainable_factor:
                factor = fn.sigmoid(self.spatial_factor).cuda()
                t = factor * s + (1. - factor) * rearrange(t, 'n b c -> b n c')
            else:
                t = (s + rearrange(t, 'n b c -> b n c')) * 0.5
        t = rearrange(self.bottle_neck(t), 'b n c -> b (n c)')
        t = self.final_layer(t)
        # return fn.sigmoid(t)  # dimension (b, n, oc)
        return t
