import math

import torch
import torch.nn as nn
import torch.nn.functional as fn
from einops import rearrange
from torch import Tensor
from torch.nn import Linear

from utility.linalg import BatchedMask, softmax_, spmm_


class SparseAttention(nn.Module):
    """Implement the sparse scaled dot product attention with softmax.
    Inspired by:
    https://tinyurl.com/yxq4ry64 and https://tinyurl.com/yy6l47p4
    """

    def __init__(self,
                 in_channels,
                 max_position_embeddings=128,
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
        self.in_channels = in_channels
        self.softmax_temp = softmax_temp
        self.dropout = attention_dropout

        # self.ln_o = Linear(mdl_channels, mdl_channels)

    def forward(self, queries, keys, values, adj):
        """Implements the multi-head softmax attention.
        Arguments
        ---------
            :param queries: torch.Tensor (N, L, E) The tensor containing the queries
            :param keys: torch.Tensor (N, S, E) The tensor containing the keys
            :param values: torch.Tensor (N, S, D) The tensor containing the values
            :param adj: the adjacency matrix plays role of mask that encodes where each query can attend to
        """
        # lq, lk, lv = self.ln_q(queries), self.ln_k(keys), self.ln_v(values)

        # Extract some shapes and compute the temperature
        # q, k, v = self.split_head(lq), self.split_head(lk), self.split_head(lv)

        n, l, h, e = queries.shape  # batch, n_heads, length, depth
        _, _, s, d = values.shape

        softmax_temp = self.softmax_temp or 1. / math.sqrt(e)

        # queries = rearrange(queries, 'n l h e -> n l (h e)')
        # keys = rearrange(keys, 'n l h e -> n l (h e)')
        # Compute the un-normalized sparse attention according to adjacency matrix indices
        # .to(queries.device),
        qk = torch.sum(queries[..., adj[0], :, :] *
                       keys[..., adj[1], :, :], dim=-1)

        # qk = rearrange(qk, 'n h l e -> n h (l e)')
        # Compute the attention and the weighted average, adj[0] is cols idx in the same row
        # alpha = fn.dropout(softmax_(softmax_temp * (qk + edge_pos_enc), adj[0]),
        #                   training=self.training)
        alpha = fn.dropout(softmax_(softmax_temp * qk, adj[0]),
                           training=self.training)
        # sparse matmul, adj as indices and qk as nonzero
        v = spmm_(adj, alpha, l, s, values)
        # v = torch.reshape(v, (n, l, h * d))   # concatenate the multi-heads attention
        # Make sure that what we return is contiguous
        return v.contiguous()


class FullAttention(nn.Module): #B * T X V X C
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

    def __init__(self, in_channels, max_position_embeddings=128,
                 softmax_temp=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.max_position_embeddings = max_position_embeddings
        self.embedding_weight = nn.Parameter(torch.randn(
            2 * max_position_embeddings + 1, in_channels))
        self.distance_embedding = nn.Embedding(
            2 * max_position_embeddings + 1, in_channels, _weight=self.embedding_weight)

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
        n, l, h, e = queries.shape
        _, s, _, d = values.shape
        softmax_temp = self.softmax_temp or 1. / math.sqrt(e)

        # Compute the unnormalized attention and apply the masks
        qk = torch.einsum("nlhe, nshe -> nhls", queries, keys)

        position_ids_l = torch.arange(
            l, dtype=torch.long, device=queries.device).view(-1, 1)
        position_ids_r = torch.arange(
            l, dtype=torch.long, device=queries.device).view(1, -1)

        distance = (position_ids_l - position_ids_r).clip(-self.max_position_embeddings,
                                                          self.max_position_embeddings)
        positional_embedding = self.distance_embedding(
            distance + self.max_position_embeddings)

        relative_position_scores_query = torch.einsum(
            "blhd, lrd -> bhlr", queries, positional_embedding)
        relative_position_scores_key = torch.einsum(
            "brhd, lrd -> bhlr", keys, positional_embedding)
        qk = qk + relative_position_scores_query + relative_position_scores_key

        if not attn_mask.all_ones:
            qk = qk + attn_mask.additive_matrix
        # QK = QK + key_lengths.additive_matrix[:, None, None]

        # Compute the attention and the weighted average
        att = self.dropout(torch.softmax(softmax_temp * qk, dim=-1))
        v = torch.einsum("nhls, nshd -> nlhd", att, values)

        # Make sure that what we return is contiguous
        return v.contiguous()


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, beta, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
        self.beta = beta
        if self.beta:
            self.lin_beta = Linear(3 * normalized_shape, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.ln.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x, y):
        if self.beta:
            b = self.lin_beta(torch.cat([y, x, y - x], dim=-1))
            b = b.sigmoid()
            return self.ln(b * x + (1 - b) * self.dropout(y))

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


class PositionalEncoding(nn.Module):
    def __init__(self,
                 model_dim: int):
        """ Positional Encoding
            This kind of encoding uses the trigonometric functions to
            incorporate the relative position information into the input
            sequence
        :param model_dim (int): the dimension of the token (feature channel length)
        """
        super(PositionalEncoding, self).__init__()
        self.model_dim = model_dim

    def forward(self, x) -> Tensor:
        sequence_length = x.shape[-2]
        pos = torch.arange(sequence_length, dtype=torch.float,
                           device=x.device).reshape(1, -1, 1)
        dim = torch.arange(self.model_dim, dtype=torch.float,
                           device=x.device).reshape(1, 1, -1)
        phase = (pos / 1e4) ** (dim // self.model_dim)
        assert x.shape[-2] == sequence_length and x.shape[-1] == self.model_dim
        return x + torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels),
            nn.Dropout(dropout)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        return self.net(x)


class TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 activation = False,
                 dropout=0,
                 bias=True):
        super(TemporalConv, self).__init__()
        pad = int((kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            stride=stride,
            bias=bias)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.activation = activation

    def forward(self, x):
        
        x = self.dropout(x)
        x = self.bn(self.conv(x))  # B * M, C, T, V
        return self.relu(x) if self.activation else x


class EncoderLayer(nn.Module):
    def __init__(self,
                 in_channels=6,
                 mdl_channels=64,
                 heads=8,
                 spatial=False,
                 beta=True,
                 dropout=0.1,
                 temp_conv_knl=9,
                 temp_conv_stride=1,
                 num_conv_layers=3):
        super(EncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.mdl_channels = mdl_channels
        self.heads = heads
        self.dropout = dropout
        self.spatial = spatial
        self.beta = beta
        self.num_conv_layers = num_conv_layers

        # self.bn = nn.BatchNorm1d(in_channels * 25)
        self.temp_conv = nn.ModuleList([TemporalConv(in_channels=mdl_channels,
                                                     out_channels=mdl_channels,
                                                     kernel_size=temp_conv_knl // 2 + 1 if i == (num_conv_layers - 1) else temp_conv_knl,
                                                     stride=temp_conv_stride * 2 if i == (num_conv_layers - 1) else temp_conv_stride) for i in range(num_conv_layers)])

        self.lin_q = Linear(in_channels, mdl_channels)
        self.lin_k = Linear(in_channels, mdl_channels)
        self.lin_v = Linear(in_channels, mdl_channels)

        if spatial:
            self.multi_head_attn = SparseAttention(in_channels=mdl_channels // heads,
                                                   max_position_embeddings=128,
                                                   attention_dropout=dropout)

        else:
            self.multi_head_attn = FullAttention(in_channels=mdl_channels // heads,
                                                 max_position_embeddings=128,
                                                 attention_dropout=dropout)

        self.add_norm_att = AddNorm(self.mdl_channels, self.beta, self.dropout)
        self.add_norm_ffn = AddNorm(self.mdl_channels, False, self.dropout)
        self.ffn = FeedForward(
            self.mdl_channels, self.mdl_channels, self.dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_k.reset_parameters()
        self.lin_q.reset_parameters()
        self.lin_v.reset_parameters()
        self.add_norm_att.reset_parameters()
        self.add_norm_ffn.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, bi=None):
        # x = self.bn()
        # batch norm (x)
        f, n, c = x.shape
        q, k, v = x, x, x

        query = self.lin_q(q)
        key = self.lin_k(k)
        value = self.lin_v(v)
        if self.spatial:
            attn_mask = bi
        else:
            attn_mask = BatchedMask(bi) if not self.spatial else None

        if self.spatial:
            query = rearrange(query, 'f n (h c) -> f n h c', h=self.heads)
            key = rearrange(key, 'f n(h c) -> f n h c', h=self.heads)
            value = rearrange(value, 'f n (h c) -> f n h c', h=self.heads)
        else:
            query = rearrange(query, 'f n (h c) -> n f h c', h=self.heads)
            key = rearrange(key, 'f n (h c) -> n f h c', h=self.heads)
            value = rearrange(value, 'f n (h c) -> n f h c', h=self.heads)

        t = self.multi_head_attn(query, key, value, attn_mask)
        if self.spatial:
            t = rearrange(t, 'f n h c -> f n (h c)', h=self.heads)
        else:
            t = rearrange(t, 'n f h c -> f n (h c)', h=self.heads)

        x = self.add_norm_att(x, t)
        x = self.add_norm_ffn(x, self.ffn(x))

        x = rearrange(x, 'f n c -> n c f')
        for i in range(self.num_conv_layers):
            x = self.temp_conv[i](x)

        # x = rearrange(x, 'n f c -> f n c')
        # batch norm(x)
        x = rearrange(x, 'n c f -> f n c')
        return x
