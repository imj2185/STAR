from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as fn
# from third_party.performer import SelfAttention
from einops import rearrange
from .attentions import EncoderLayer, PositionalEncoding
from .layers import HGAConv, GlobalContextAttention, TemporalSelfAttention, PositionalEncoding, TransformerEncoder
from torch_scatter import scatter_mean

class DualGraphEncoder(nn.Module, ABC):
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
        super(DualGraphEncoder, self).__init__()
        self.spatial_factor = nn.Parameter(torch.ones(num_layers)) * 0.5
        self.sequential = sequential
        self.num_layers = num_layers
        # self.num_joints = num_joints
        # self.num_classes = classes
        self.dropout = drop_rate
        self.trainable_factor = trainable_factor
        self.bn = nn.BatchNorm1d(in_channels * 25)
        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        
        #self.spatial_norms = nn.ModuleList([
            # nn.BatchNorm1d(channels[i + 1]) for i in range(num_layers)
        #    nn.LayerNorm(channels[i + 1]) for i in range(num_layers)
        #])

        channels_ = channels[1:] + [out_channels]
        self.positional_encoding = PositionalEncoding(model_dim=in_channels)

        self.lls = nn.ModuleList([nn.Linear(in_features=channels[i],
                                            out_features=channels[i + 1]) for i in range(num_layers)])

        self.spatial_layers = nn.ModuleList([
            EncoderLayer(in_channels=channels_[i],
                    mdl_channels=channels_[i + 1],
                    spatial=True,
                    heads=num_heads,
                    dropout=drop_rate) for i in range(num_layers)])

        """self.temporal_layers = nn.ModuleList([
            # necessary parameters are: dim
            EncoderLayer(in_channels=channels_[i],  # TODO ??? potential dimension problem
                                  mdl_channels=channels_[i + 1],
                                  spatial=False,
                                  heads=num_heads,
                                  dropout=drop_rate) for i in range(num_layers)])"""

        self.context_attention = GlobalContextAttention(in_channels=out_channels)
        #self.final_layer = nn.Linear(in_features=out_channels * num_joints,
        #                             out_features=classes)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(out_channels * num_joints),
            nn.Linear(out_channels * num_joints, classes)
        )

    def forward(self, t, adj, bi):  # t: tensor, adj: dataset.skeleton_
        """

        :param t: tensor
        :param adj: adjacency matrix (sparse)
        :param bi: batch index
        :return: tensor
        """
        """ if self.sequential:  # sequential architecture
            for i in range(self.num_layers):
                t = rearrange(fn.relu(self.spatial_layers[i](t, adj)),
                              'b n c -> n b c')
                t = rearrange(fn.relu(fn.layer_norm(fn.dropout(self.temporal_layers[i](t),
                                                               self.dropout),
                                                    t.shape[1:]) + t),  # residual and add_norm
                              'n b c -> b n c')
        else:  # parallel architecture"""
        c = t.shape[-1]
        t = self.bn(rearrange(t, 'b n c -> b (n c)'))
        t = rearrange(t, 'b (n c) -> b n c', c=c)
        t = self.positional_encoding(t)
        for i in range(self.num_layers):
            #Batch X Frames, 25, 6
            t = fn.relu(self.lls[i](t))
            #t = self.lls[i](t)
            #t = self.temporal_layers[i](t, bi)
            t = self.spatial_layers[i](t, adj)
            #t = fn.relu(self.temporal_layers[i](t, bi))
                #s = self.spatial_norms[i](fn.relu(self.spatial_layers[i](s, adj)))
                #if self.trainable_factor:
                #    factor = torch.sigmoid(self.spatial_factor).to(t.device)
                #    t = factor[i] * s + (1. - factor[i]) * t
                #else:
                #    t = (s + t) * 0.5
        #t = scatter_mean(rearrange(t, 'n f c -> f n c'), bi, dim=0)
        t = rearrange(t, 'f n c -> n f c')
        t = rearrange(self.context_attention(t, batch_index=bi),
                      'n f c -> f (n c)')  # bi is the shrunk along the batch index
        t = self.mlp_head(t)
        # return fn.sigmoid(t)  # dimension (b, n, oc)
        return t
