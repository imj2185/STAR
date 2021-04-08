from abc import ABC

import torch
import torch.nn as nn
# from third_party.performer import SelfAttention
from einops import rearrange
from models.layers import Swish

from models.positional_encoding import SeqPosEncoding
from .attentions import SpatialEncoderLayer, TemporalEncoderLayer, GlobalContextAttention


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
                 drop_rate=None,
                 mean_bottleneck=False,
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
        self.positional_encoding = SeqPosEncoding(model_dim=hidden_channels)

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

        self.mean_bottleneck = mean_bottleneck

        self.mlp_head = nn.Sequential(
            nn.Linear(out_channels if mean_bottleneck else out_channels * num_joints,
                      mlp_head_hidden),
            # nn.Linear(out_channels, mlp_head_hidden),
            # nn.Tanh(),
            # nn.LeakyReLU(),  # nn.SiLU(),
            Swish(),
            # nn.Dropout(p=0.3),
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

        t = self.positional_encoding(t, bi)
        t = rearrange(t, 'n b c -> b n c')

        # Core pipeline
        for i in range(self.num_layers):
            # u = t  # branch
            # t = self.spatial_layers[i](t, FullMask(25, 25, device=t.device))
            t = self.spatial_layers[i](t, adj) + self.temporal_layers[i](t, bi)

        t = rearrange(t, 'f n c -> n f c')
        # @context-aware attention shrinks the frames dimension: f -> m,
        # where m is the actual number of video clip in a batch
        if self.mean_bottleneck:
            t = self.context_attention(t, batch_index=bi)
            t = rearrange(t, 'n m c -> m n c').mean(1)
        else:
            t = rearrange(self.context_attention(t, batch_index=bi),
                          'n m c -> m (n c)')  # bi is the shrunk along the batch index
        t = self.mlp_head(t)
        return t
