import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as fn
from einops import rearrange
from fast_transformers.feature_maps import elu_feature_map
from torch.nn import Linear
# from torch_geometric.nn.norm import LayerNorm
from .layers import LayerNorm
from .kernels import generalized_kernel, softmax_kernel, gaussian_orthogonal_random_matrix
from torch_scatter import scatter_sum, scatter_mean

from utility.linalg import softmax_, spmm_


class SparseAttention(nn.Module):
    """Implement the sparse scaled dot product attention with softmax.
    Inspired by:
    https://tinyurl.com/yxq4ry64 and https://tinyurl.com/yy6l47p4
    """

    def __init__(self,
                 in_channels,
                 softmax_temp=None,
                 # num_adj=1,
                 attention_dropout=0.1):
        """
        :param heads (int):
        :param in_channels (int):
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

    def forward(self, queries, keys, values, adj):
        """Implements the multi-head softmax attention.
        Arguments
        ---------
            :param queries: torch.Tensor (N, L, E) The tensor containing the queries
            :param keys: torch.Tensor (N, S, E) The tensor containing the keys
            :param values: torch.Tensor (N, S, D) The tensor containing the values
            :param adj: the adjacency matrix plays role of mask that encodes where each query can attend to
        """
        # Extract some shapes and compute the temperature
        n, l, h, e = queries.shape  # batch, n_heads, length, depth
        _, _, s, d = values.shape

        softmax_temp = self.softmax_temp or 1. / math.sqrt(e)

        # Compute the un-normalized sparse attention according to adjacency matrix indices
        if isinstance(adj, torch.Tensor):
            adj_ = adj
            qk = torch.sum(queries.index_select(dim=-3, index=adj[0]) * keys.index_select(dim=-3, index=adj[1]), dim=-1)

        else:
            qk = adj_ = None
            """qk = torch.cat([self.beta[k] * torch.sum(
                queries[..., adj[k][0], :, :] *
                keys[..., adj[k][1], :, :], dim=-1) for k in range(len(adj))], dim=0)
            adj_ = torch.cat(adj, dim=1)
            _, idx = adj_[0].sort()
            adj_ = adj_[:, idx]
            qk = qk[idx]"""

        # Compute the attention and the weighted average, adj[0] is cols idx in the same row
        '''alpha = fn.dropout(softmax_(softmax_temp * qk, adj[0]),
                           p=self.dropout,
                           training=self.training)'''
        alpha = softmax_(softmax_temp * qk, adj[0])
        # sparse matmul, adj as indices and qk as nonzero
        v = spmm_(adj_, alpha, l, l, values)
        v = fn.dropout(v, p=self.dropout)
        # Make sure that what we return is contiguous
        return v.contiguous()


class LinearAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 softmax_temp=None,
                 use_generalized_kernel=False,
                 use_gaussian_feature=False,
                 eps=1e-6,
                 attention_dropout=0.1):
        super(LinearAttention, self).__init__()
        self.in_channels = in_channels
        self.softmax_temp = softmax_temp
        self.dropout = attention_dropout
        self.eps = eps
        self.gaussian_feature = partial(gaussian_orthogonal_random_matrix,
                                        nb_columns=in_channels) if use_gaussian_feature else None
        self.use_generalized_kernel = use_generalized_kernel

    def forward(self, queries, keys, values, bi=None):
        n, l, h, e = queries.shape  # batch, n_heads, length, depth
        nb_features = int(e * math.log(e)) if l < e else l
        gaussian_feature = self.gaussian_feature(nb_features=nb_features, device=queries.device)
        feature_map = partial(generalized_kernel,
                              projection_matrix=gaussian_feature,
                              kernel_fn=torch.nn.ELU()) if self.use_generalized_kernel \
            else partial(softmax_kernel, projection_matrix=gaussian_feature)
        q = feature_map(queries)
        k = feature_map(keys)

        if bi is None:
            kv = torch.einsum("nshd, nshm -> nhmd", k, values)
            z = 1 / (torch.einsum("nlhd, nhd -> nlh", q, k.sum(dim=1)) + self.eps)
            v = torch.einsum("nlhd, nhmd, nlh -> nlhm", q, kv, z)
        else:
            # change the dimensions of values to (N, H, L, 1, D) and keys to (N, H, L, D, 1)
            q = rearrange(q, 'n l h d -> n h l d')
            k = rearrange(k, 'n l h d -> n h l d')
            kv = torch.matmul(rearrange(k, 'n h l d -> n h l d 1'),
                              rearrange(values, 'n l h d -> n h l 1 d'))  # N H L D1 D2
            kv = scatter_sum(kv, bi, dim=-3).index_select(dim=-3, index=bi)  # N H (L) D1 D2
            k_ = scatter_sum(k, bi, dim=-2).index_select(dim=-2, index=bi)
            z = 1 / torch.sum(q * k_, dim=-1)
            v = torch.matmul(rearrange(q, 'n h l d -> n h l 1 d'),
                             kv).squeeze(dim=-2) * z.unsqueeze(-1)
            # return rearrange(v, 'n h l d -> n l h d').contiguous()
            v = fn.dropout(rearrange(v, 'n h l d -> n l h d'), p=self.dropout)
        return v.contiguous()


class GlobalContextAttention(nn.Module):
    def __init__(self, in_channels):
        super(GlobalContextAttention, self).__init__()
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, in_channels))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, batch_index):
        """
        :param x: tensor(joints, frames, channels)
        :param batch_index: batch index
        :return: reduced tensor
        """
        # Global context
        gc = torch.matmul(scatter_mean(x, batch_index, dim=1), self.weight)
        # extended according to batch index
        gc = torch.tanh(gc).index_select(dim=-2, index=batch_index)  # [..., batch_index, :]
        gc_ = torch.sigmoid(torch.sum(torch.mul(x, gc), dim=-1, keepdim=True))
        return scatter_mean(gc_ * x, index=batch_index, dim=1)


class ContextAttention(nn.Module):
    def __init__(self, in_channels):
        super(ContextAttention, self).__init__()
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, in_channels))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, batch_index):
        """
        :param x: tensor(joints, frames, channels)
        :param batch_index: batch index
        :return: reduced tensor
        """
        # Global context
        gc = torch.matmul(scatter_mean(x, batch_index, dim=0), self.weight)
        # extended according to batch index
        gc = torch.tanh(gc).index_select(dim=0, index=batch_index)
        gc_ = torch.sigmoid(torch.sum(torch.mul(x, gc), dim=-1, keepdim=True))
        return gc_ * x


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, beta, dropout, post_norm, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.beta = beta
        self.post_norm = post_norm
        if self.post_norm:
            self.ln = LayerNorm(normalized_shape, affine=True)
        if self.beta:
            self.lin_beta = Linear(3 * normalized_shape, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        # self.ln.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x, y, bi=None):
        if self.beta:
            b = self.lin_beta(torch.cat([y, x, y - x], dim=-1))
            b = b.sigmoid()
            return self.ln(b * x + (1 - b) * self.dropout(y))

        if self.post_norm:
            return self.ln(self.dropout(y) + x, batch=bi)  # self.dropout(y) + x

        return self.dropout(y) + x


class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0., init_factor=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels),
            nn.Dropout(dropout)
        )
        self.init_factor = init_factor
        self.reset_parameters()

    def reset_parameters(self):
        temp_state_dic = {}
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                # layer.reset_parameters()
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.)
        self.fixup_initialization(self.init_factor)

    def fixup_initialization(self, init_factor):
        import collections
        temp_state_dic = collections.OrderedDict()
        if init_factor:
            for name, param in self.named_parameters():
                if name in ["net.0.weight",
                            "net.3.weight"
                            ]:
                    temp_state_dic[name] = (9 * init_factor) ** (- 1. / 4.) * param
                # elif name in ["self_attn.v_proj.weight", "encoder_attn.v_proj.weight", ]:
                #     temp_state_dic[name] = (9 * init_factor) ** (- 1. / 4.) * (param * (2 ** 0.5))

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)

    def forward(self, x):
        return self.net(x)


class SpatialEncoderLayer(nn.Module):
    def __init__(self,
                 in_channels=6,
                 mdl_channels=64,
                 heads=8,
                 beta=True,
                 dropout=None,
                 init_factor=5,
                 pre_norm=False,
                 post_norm=True):
        super(SpatialEncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.mdl_channels = mdl_channels
        self.heads = heads
        self.beta = beta
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        if dropout is None:
            self.dropout = [0.5, 0.5, 0.5, 0.5]  # temp_conv, sparse_attention, add_norm, ffn
        else:
            self.dropout = dropout

        # self.tree_key_weights = nn.Parameter(torch.randn(in_channels, in_channels), requires_grad=True)
        # self.tree_value_weights = nn.Parameter(torch.randn(in_channels, in_channels), requires_grad=True)

        self.lin_qkv = Linear(in_channels, mdl_channels * 3, bias=True)

        self.multi_head_attn = SparseAttention(in_channels=mdl_channels // heads,
                                               attention_dropout=dropout[1])

        self.add_norm_att = AddNorm(self.mdl_channels, False, self.dropout[2], self.post_norm)
        self.add_norm_ffn = AddNorm(self.mdl_channels, False, self.dropout[2], self.post_norm)
        self.ffn = FeedForward(self.mdl_channels, self.mdl_channels // 2, self.dropout[3], init_factor)
        if self.pre_norm:
            self.ln_att = nn.LayerNorm(self.mdl_channels)
            self.ln_ffn = nn.LayerNorm(self.mdl_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_qkv.weight, gain=1 / math.sqrt(2))
        # nn.init.xavier_normal_(self.lin_qkv.bias)
        nn.init.zeros_(self.lin_qkv.bias)
        '''self.lin_qkv.reset_parameters()
        self.add_norm_att.reset_parameters()
        self.add_norm_ffn.reset_parameters()
        self.ffn.reset_parameters()'''

    def forward(self, x, adj=None):  # , tree_encoding=None):
        f, n, c = x.shape
        if self.pre_norm:
            x = self.ln_att(x)
        query, key, value = self.lin_qkv(x).chunk(3, dim=-1)

        # tree_pos_enc_key = torch.matmul(tree_encoding, self.tree_key_weights)
        # tree_pos_enc_value = torch.matmul(tree_encoding, self.tree_value_weights)

        # key = key + tree_pos_enc_key.unsqueeze(dim=0)
        # value = value + tree_pos_enc_value.unsqueeze(dim=0)

        query = rearrange(query, 'f n (h c) -> f n h c', h=self.heads)
        key = rearrange(key, 'f n (h c) -> f n h c', h=self.heads)
        value = rearrange(value, 'f n (h c) -> f n h c', h=self.heads)

        t = self.multi_head_attn(query, key, value, adj)
        t = rearrange(t, 'f n h c -> f n (h c)', h=self.heads)

        x = self.add_norm_att(x, t)
        if self.pre_norm:
            x = self.ln_ffn(x)
        x = self.add_norm_ffn(x, self.ffn(x))

        return x


class TemporalEncoderLayer(nn.Module):
    def __init__(self,
                 in_channels=6,
                 mdl_channels=64,
                 heads=8,
                 beta=False,
                 dropout=0.1,
                 init_factor=5,
                 pre_norm=False,
                 post_norm=True):
        super(TemporalEncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.mdl_channels = mdl_channels
        self.heads = heads
        self.beta = beta
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        if dropout is None:
            self.dropout = [0.5, 0.5, 0.5, 0.5]  # temp_conv, sparse_attention, add_norm, ffn
        else:
            self.dropout = dropout

        self.lin_qkv = Linear(in_channels, mdl_channels * 3, bias=True)

        self.multi_head_attn = LinearAttention(in_channels=mdl_channels // heads,
                                               attention_dropout=self.dropout[0])

        self.add_norm_att = AddNorm(self.mdl_channels, self.beta, self.dropout[2], self.post_norm)
        self.add_norm_ffn = AddNorm(self.mdl_channels, False, self.dropout[2], self.post_norm)
        self.ffn = FeedForward(self.mdl_channels, self.mdl_channels // 2, self.dropout[3], init_factor)

        if self.pre_norm:
            self.ln_att = LayerNorm(self.mdl_channels)
            self.ln_ffn = LayerNorm(self.mdl_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_qkv.weight, gain=1 / math.sqrt(2))
        # nn.init.xavier_normal_(self.lin_qkv.bias)
        nn.init.zeros_(self.lin_qkv.bias)
        # self.add_norm_att.reset_parameters()
        # self.add_norm_ffn.reset_parameters()
        # self.ffn.reset_parameters()

    def forward(self, x, bi=None):
        f, n, c = x.shape
        if self.pre_norm:
            x = self.ln_att(x, bi)
        query, key, value = self.lin_qkv(x).chunk(3, dim=-1)

        query = rearrange(query, 'f n (h c) -> n f h c', h=self.heads)
        key = rearrange(key, 'f n (h c) -> n f h c', h=self.heads)
        value = rearrange(value, 'f n (h c) -> n f h c', h=self.heads)

        t = self.multi_head_attn(query, key, value, bi)
        t = rearrange(t, 'n f h c -> f n (h c)', h=self.heads)

        x = self.add_norm_att(x, t, bi)
        if self.pre_norm:
            x = self.ln_ffn(x, bi)
        x = self.add_norm_ffn(x, self.ffn(x), bi)

        return x
