from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as fn
# from third_party.performer import SelfAttention
from einops import rearrange

from .layers import HGAConv, GlobalContextAttention, TemporalSelfAttention


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
                 linear_temporal=True,
                 trainable_factor=False):
        super(DualGraphTransformer, self).__init__()
        self.spatial_factor = nn.Parameter(torch.ones(num_layers)) * 0.5
        self.sequential = sequential
        self.num_layers = num_layers
        # self.num_joints = num_joints
        # self.num_classes = classes
        self.dropout = drop_rate
        self.trainable_factor = trainable_factor
        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.spatial_norms = nn.ModuleList([
            # nn.BatchNorm1d(channels[i + 1]) for i in range(num_layers)
            nn.LayerNorm(channels[i + 1]) for i in range(num_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            HGAConv(in_channels=channels[i],
                    heads=num_heads,
                    dropout=drop_rate,
                    out_channels=channels[i + 1]) for i in range(num_layers)])

        if not self.sequential:
            self.temporal_lls = nn.ModuleList([nn.Linear(in_features=channels[i],
                                                         out_features=channels[i + 1])
                                               for i in range(num_layers)])

        channels_ = channels[1:] + [out_channels]
        self.temporal_layers = nn.ModuleList([
            # necessary parameters are: dim
            TemporalSelfAttention(in_channels=channels_[i],  # TODO ??? potential dimension problem
                                  hid_channels=channels_[i + 1],
                                  is_linear=linear_temporal,
                                  heads=num_heads,
                                  dropout=drop_rate) for i in range(num_layers)])

        self.context_attention = GlobalContextAttention(in_channels=out_channels)
        self.final_layer = nn.Linear(in_features=out_channels * num_joints,
                                     out_features=classes)

    def forward(self, t, adj, bi):  # t: tensor, adj: dataset.skeleton_
        """

        :param t: tensor
        :param adj: adjacency matrix (sparse)
        :param bi: batch index
        :return: tensor
        """
        if self.sequential:  # sequential architecture
            for i in range(self.num_layers):
                t = rearrange(fn.relu(self.spatial_layers[i](t, adj)),
                              'b n c -> n b c')
                t = rearrange(fn.relu(fn.layer_norm(fn.dropout(self.temporal_layers[i](t),
                                                               self.dropout),
                                                    t.shape[1:]) + t),  # residual and add_norm
                              'n b c -> b n c')
        else:  # parallel architecture
            for i in range(self.num_layers):
                s = t
                t = fn.relu(self.temporal_lls[i](t))
                t = fn.relu(self.temporal_layers[i](t, bi))
                s = self.spatial_norms[i](fn.relu(self.spatial_layers[i](s, adj)))
                if self.trainable_factor:
                    factor = torch.sigmoid(self.spatial_factor).to(t.device)
                    t = factor[i] * s + (1. - factor[i]) * t
                else:
                    t = (s + t) * 0.5

        t = rearrange(self.context_attention(rearrange(t,
                                                       'f n c -> n f c'),
                                             batch_index=bi),
                      'n f c -> f (n c)')  # bi is the shrunk along the batch index
        t = self.final_layer(fn.relu(t))
        # return fn.sigmoid(t)  # dimension (b, n, oc)
        return t
