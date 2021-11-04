from abc import ABC

import torch
import torch.nn as nn
from einops import rearrange

from .layers import SpatialEncoderLayer, TemporalEncoderLayer
from .attentions import GlobalContextAttention


class DualGraphEncoder(nn.Module, ABC):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 mlp_head_hidden,
                 num_layers,
                 num_heads=8,
                 num_joints=25,
                 classes=60,
                 spatial_pos_enc=None,
                 temporal_pos_enc=None,
                 drop_rate=None,
                 sequential=True,
                 trainable_factor=False,
                 num_conv_layers=3):
        super(DualGraphEncoder, self).__init__()
        if drop_rate is None:
            self.drop_rate = [0.5, 0.5, 0.5, 0.5]  # temp_conv, sparse_attention, add_norm, ffn
        else:
            self.drop_rate = drop_rate
        self.spatial_factor = nn.Parameter(torch.ones(num_layers)) * 0.5
        self.sequential = sequential
        self.num_layers = num_layers
        self.num_conv_layers = num_conv_layers
        self.num_joints = num_joints
        self.num_classes = classes
        self.dropout = drop_rate
        self.trainable_factor = trainable_factor
        # self.bn = nn.BatchNorm1d(hidden_channels * 25, affine=False)
        self.dn = nn.BatchNorm1d(in_channels * 25, affine=True)
        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        channels_ = channels[1:] + [out_channels]

        self.tree_encoding = None
        # tree_encoding_from_traversal(onehot_length=3, max_padding=hidden_channels)
        self.temporal_pos_enc = temporal_pos_enc
        self.spatial_pos_enc = spatial_pos_enc

        self.lls = nn.Linear(in_features=channels[0], out_features=channels[1])

        self.spatial_layers = nn.ModuleList([
            SpatialEncoderLayer(in_channels=channels_[i],
                                mdl_channels=channels_[i + 1],
                                heads=num_heads,
                                dropout=self.drop_rate) for i in range(num_layers)])

        self.temporal_layers = nn.ModuleList([
            TemporalEncoderLayer(in_channels=channels_[i],
                                 mdl_channels=channels_[i + 1],
                                 heads=num_heads,
                                 dropout=self.drop_rate) for i in range(num_layers)])

        self.context_attention = GlobalContextAttention(in_channels=out_channels)

        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(out_channels * num_joints),
            nn.Linear(out_channels * num_joints, mlp_head_hidden),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(mlp_head_hidden, classes)
        )

    def forward(self, t, adj, bi):  # t: tensor, adj: dataset.skeleton_
        """

        :param t: tensor
        :param adj: adjacency matrix (sparse)
        :param bi: batch index
        :return: tensor
        """
        c = t.shape[-1]
        t = self.dn(rearrange(t, 'b n c -> b (n c)'))
        t = self.lls(rearrange(t, 'b (n c) -> b n c', c=c))
        # c = t.shape[-1]
        # t = self.bn(rearrange(t, 'b n c -> b (n c)'))
        # t = rearrange(t, 'b (n c) -> b n c', c=c)
        t = rearrange(t, 'b n c -> n b c')

        t = self.temporal_pos_enc(t, bi)
        t = rearrange(t, 'n b c -> b n c')
        att = None

        # Core pipeline
        for i in range(self.num_layers):
            u = t  # branch
            # t = self.spatial_layers[i](t, FullMask(25, 25, device=t.device))
            t = self.spatial_layers[i](t, adj)
            u = rearrange(u, 'f n c -> n f c')
            u = self.temporal_layers[i](u, bi)
            u = rearrange(u, 'n f c -> f n c')
            t = u + t

        t = rearrange(t, 'f n c -> n f c')
        # bi_ = bi[:bi.shape[0]:2**self.num_layers]
        t = rearrange(self.context_attention(t, batch_index=bi),
                      'n f c -> f (n c)')  # bi is the shrunk along the batch index
        # t = rearrange(global_mean_pool(t, bi), 'f n c -> f (n c)')
        t = self.mlp_head(t)
        # return fn.sigmoid(t)  # dimension (b, n, oc)
        return t
