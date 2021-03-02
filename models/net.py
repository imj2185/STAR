from abc import ABC

import torch
import torch.nn as nn
# from third_party.performer import SelfAttention
from einops import rearrange


from .attentions import SpatialEncoderLayer, TemporalEncoderLayer
from models.positional_encoding import SeqPosEncoding
from .layers import GlobalContextAttention
from utility.tree import tree_encoding_from_traversal


class DualGraphEncoder(nn.Module, ABC):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 num_heads=8,
                 num_joints=25,
                 classes=60,
                 drop_rate=None,
                 sequential=True,
                 trainable_factor=False,
                 num_conv_layers=3):
        super(DualGraphEncoder, self).__init__()
        if drop_rate is None:
            self.drop_rate = [0.3, 0.3, 0.3, 0.3]
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
        self.bn = nn.BatchNorm1d(hidden_channels * 25)
        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        # channels = [hidden_channels * (i + 1)^2 for i in range(num_layers - 1)]
        # self.spatial_norms = nn.ModuleList([
        # nn.BatchNorm1d(channels[i + 1]) for i in range(num_layers)
        #    nn.LayerNorm(channels[i + 1]) for i in range(num_layers)
        # ])

        channels_ = channels[1:] + [out_channels]

        
        self.tree_encoding = tree_encoding_from_traversal(onehot_length=3, max_padding=hidden_channels)
        #self.embedding_weight = nn.Parameter(torch.randn(num_joints, hidden_channels))
        self.positional_encoding = SeqPosEncoding(model_dim=hidden_channels)

        # self.lls = nn.ModuleList([nn.Linear(in_features=channels[i],
        #                                     out_features=channels[i + 1]) for i in range(num_layers)])
        self.lls = nn.Linear(in_features=channels[0], out_features=channels[1])

        self.spatial_layers = nn.ModuleList([
            SpatialEncoderLayer(in_channels=channels_[i],
                         mdl_channels=channels_[i + 1],
                         heads=num_heads,
                         dropout=drop_rate) for i in range(num_layers)])

        self.temporal_layers = nn.ModuleList([
            TemporalEncoderLayer(in_channels=channels_[i],
                         mdl_channels=channels_[i + 1],
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

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(out_channels * num_joints),
            nn.Linear(out_channels * num_joints, out_channels * num_joints),
            # nn.Tanh(),
            nn.LeakyReLU(),
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
        t = self.lls(t)
        c = t.shape[-1]
        t = self.bn(rearrange(t, 'b n c -> b (n c)'))
        #t = rearrange(t, 'b (n c) -> b n c', c=c)
        t = rearrange(t, 'b (n c) -> n b c', c=c)

        t = self.positional_encoding(t, bi)
        t = rearrange(t, 'n b c -> b n c')
        
        # Core pipeline
        for i in range(self.num_layers):
            t = self.spatial_layers[i](t, adj, tree_encoding=self.tree_encoding)
            t = rearrange(t, 'f n c -> n f c')
            t = self.temporal_layers[i](t, bi)
            t = rearrange(t, 'n f c -> f n c')

        t = rearrange(t, 'f n c -> n f c')
        # bi_ = bi[:bi.shape[0]:2**self.num_layers]
        t = rearrange(self.context_attention(t, batch_index=bi),
                      'n f c -> f (n c)')  # bi is the shrunk along the batch index
        t = self.mlp_head(t)
        # return fn.sigmoid(t)  # dimension (b, n, oc)
        return t
