import copy
import math
from inspect import Parameter as Pr
from typing import Union, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as fn
from einops import rearrange, reduce
from fast_transformers.attention.full_attention import FullAttention
from fast_transformers.attention.linear_attention import LinearAttention
from fast_transformers.masking import FullMask, LengthMask
from torch import Tensor
from torch.nn import Parameter, Linear, Dropout
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_scatter import scatter_mean

from utility.linalg import batched_spmm, batched_transpose, BatchedMask, softmax_, spmm_

class SparseAttention(nn.Module):
    """Implement the sparse scaled dot product attention with softmax.
    Inspired by:
    https://tinyurl.com/yxq4ry64 and https://tinyurl.com/yy6l47p4
    """
    def __init__(self,
                 heads,
                 in_channels,
                 mdl_channels,
                 softmax_temp=None,
                 attention_dropout=0.1):
        """
        :param heads (int):
        :param in_channels (int):
        :param out_channels (int):
        :param softmax_temp (torch.Tensor): The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        :param attention_dropout (float): The dropout rate to apply to the attention
                           (default: 0.1)
        """
        super(SparseAttention, self).__init__()
        assert mdl_channels % heads == 0
        self.heads = heads
        self.in_channels = in_channels
        self.mdl_channels = mdl_channels
        self.softmax_temp = softmax_temp
        self.dropout = attention_dropout
        self.ln_q = Linear(in_channels, mdl_channels)
        self.ln_k = Linear(in_channels, mdl_channels)
        self.ln_v = Linear(in_channels, mdl_channels)
        self.ln_o = Linear(mdl_channels, mdl_channels)

    def split_head(self, x):
        # x: (batch, time_in, embed_dim)
        b, l, c = x.shape
        depth = c // self.heads
        x = x.reshape(b, l, self.heads, depth)  # (batch, length, n_heads, depth)
        x = x.transpose(2, 1)                   # (batch, n_heads, length, depth)
        return x

    def forward(self, queries, keys, values, adj, edge_pos_enc):
        """Implements the multi-head softmax attention.
        Arguments
        ---------
            :param queries: torch.Tensor (N, L, E) The tensor containing the queries
            :param keys: torch.Tensor (N, S, E) The tensor containing the keys
            :param values: torch.Tensor (N, S, D) The tensor containing the values
            :param adj: An implementation of BaseMask that encodes where each query can attend to
            :param edge_pos_enc: torch.Tensor,

        """
        lq, lk, lv = self.ln_q(queries), self.ln_k(keys), self.ln_v(values)

        # Extract some shapes and compute the temperature
        q, k, v = self.split_head(lq), self.split_head(lk), self.split_head(lv)
        n, h, l, e = q.shape  # batch, n_heads, length, depth
        _, _, s, d = v.shape

        softmax_temp = self.softmax_temp or 1. / math.sqrt(e)

        # Compute the un-normalized sparse attention according to adjacency matrix indices
        qk = torch.sum(q[..., adj[0], :] * k[..., adj[1], :], dim=-1)  # .to(queries.device),

        # Compute the attention and the weighted average, adj[0] is cols idx in the same row
        alpha = fn.dropout(softmax_(softmax_temp * (qk + edge_pos_enc), adj[0]),
                           training=self.training)
        v = spmm_(adj, alpha, l, d, v)   # sparse matmul, adj as indices and qk as nonzero
        v = torch.reshape(v.transpose(2, 1), (n, s, h * d))   # concatenate the multi-heads attention
        # Make sure that what we return is contiguous
        return self.ln_o(v.contiguous())
    
class FullAttention(nn.Module):

    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, attention_head_size, max_position_embeddings=128,
                 softmax_temp=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.max_position_embeddings = max_position_embeddings
        self.distance_embedding = nn.Embedding(
            2 * max_position_embeddings + 1, attention_head_size)

    def forward(self, queries, keys, values, attn_mask):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape 
        softmax_temp = self.softmax_temp or 1. / math.sqrt(E)

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys)

        position_ids_l = torch.arange(
            L, dtype=torch.long, device=queries.device).view(-1, 1)
        position_ids_r = torch.arange(
            L, dtype=torch.long, device=queries.device).view(1, -1)
        
        distance = (position_ids_l - position_ids_r).clip(-self.max_position_embeddings, self.max_position_embeddings)
        positional_embedding = self.distance_embedding(distance + self.max_position_embeddings)

        relative_position_scores_query = torch.einsum(
            "blhd,lrd->bhlr", queries, positional_embedding)
        relative_position_scores_key = torch.einsum(
            "brhd,lrd->bhlr", keys, positional_embedding)
        QK = QK + relative_position_scores_query + relative_position_scores_key

        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        #QK = QK + key_lengths.additive_matrix[:, None, None]

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        # Make sure that what we return is contiguous
        return V.contiguous()

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)

class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hid_channels,
                 num_layers=2,
                 bias=True):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        channels = [in_channels] + \
                   [hid_channels] * (num_layers - 1) + \
                   [out_channels]  # [64, 64, 64]

        self.layers = nn.ModuleList([
            nn.Linear(in_features=channels[i],
                      out_features=channels[i + 1],
                      bias=bias) for i in range(num_layers)
        ])  # weight initialization is done in Linear()

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = fn.relu(self.layers[i](x))
        return self.layers[-1](x)

class EncoderLayer(nn.Module):
    def __init__(self,
                 in_channels=6,
                 mdl_channels=64,
                 heads=8,
                 activation="relu",
                 # add bool parameter: spatial or not
                 dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.mdl_channels = mdl_channels
        self.heads = heads
        self.dropout = dropout

        #self.bn = nn.BatchNorm1d(in_channels * 25)
        self.lin_q = Linear(in_channels, mdl_channels)
        self.lin_k = Linear(in_channels, mdl_channels)
        self.lin_v = Linear(in_channels, mdl_channels)
        self.spatial = False
        self.multi_head_attn = FullAttention(attention_head_size = heads, 
                                                max_position_embeddings=128,
                                                softmax_temp=None, 
                                                attention_dropout=dropout)
        self.add_norm_att = AddNorm(self.mdl_channels, self.dropout)
        self.add_norm_mlp = AddNorm(self.mdl_channels, self.dropout)
        self.mlp = MLP(self.mdl_channels, self.mdl_channels, self.mdl_channels)

    def forward(self, x, bi=None):
        #x = self.bn()
        #batch norm (x)
        f, n, c = x.shape
        q, k, v = x, x, x

        query = self.lin_q(q)
        key = self.lin_k(k)
        value = self.lin_v(v)

        attn_mask = BatchedMask(bi) if not self.spatial else None

        query = rearrange(query, 'n f (h c) -> n f h c', h=self.heads)
        key = rearrange(key, 'n f (h c) -> n f h c', h=self.heads)
        value = rearrange(value, 'n f (h c) -> n f h c', h=self.heads)

        t = self.multi_head_attn(query, key, value, attn_mask)
        t = rearrange(t, 'n f h c -> n f (h c)', h=self.heads)
        x = self.add_norm_att(x, t)
        x = self.add_norm_mlp(x, self.mlp(x))
        x = rearrange(x, 'n f c -> f n c')
        #batch norm(x)
        return x

