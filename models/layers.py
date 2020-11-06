import copy
from abc import ABC
import math
from inspect import Parameter as Pr
from typing import Union, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_sparse import spmm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from einops import rearrange, reduce

from utility.linalg import batched_spmm, batched_transpose,\
    transpose_, masked_softmax, get_factorized_dim, to_band_sparse


def clones(module, k):
    return nn.ModuleList(
        copy.deepcopy(module) for _ in range(k)
    )


def self_loop_augment(num_nodes, adj):
    adj, _ = remove_self_loops(adj)
    adj, _ = add_self_loops(adj, num_nodes=num_nodes)
    return adj


class HGAConv(MessagePassing):
    """
    Heterogeneous Graph Attention Convolution
    """

    def _forward_unimplemented(self, *in_tensor: Any) -> None:
        pass

    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 heads: int = 1,
                 concat: bool = False,
                 negative_slope: float = 0.2,
                 dropout: float = 0.,
                 use_self_loops: bool = False,  # Set to False for debug
                 bias: bool = True, **kwargs):
        super(HGAConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.add_self_loops = use_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, out_channels, bias=False)

            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], out_channels, False)
            self.lin_r = Linear(in_channels[1], out_channels, False)

        self.att_l = Parameter(torch.Tensor(out_channels, heads))
        self.att_r = Parameter(torch.Tensor(out_channels, heads))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(out_channels * heads))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    @staticmethod
    def edge_score(adj, a_l, a_r):
        """
        Args:
            adj: adjacency matrix [2, num_edges] or (heads, [2, num_edges])
            a_l: Tensor           [num_nodes, heads]
            a_r: Tensor           [num_nodes, heads]
        """
        if isinstance(adj, Tensor):
            if len(a_l.shape) == 2:
                return a_l[adj[1], :] + a_r[adj[0], :]  # [num_edges, heads]
            else:
                a_l_ = rearrange(a_l, 'b n c -> n (b c)')
                a_r_ = rearrange(a_r, 'b n c -> n (b c)')
                out = a_l_[adj[1], :] + a_r_[adj[0], :]
                return rearrange(out, 'n (b c) -> b n c', c=a_l.shape[-1])
        a = []
        for i in range(len(adj)):
            a[i] = a_l[adj[i][1], i] + a_r[adj[i][0], i]
        return a  # (heads, [num_edges, 1])

    def forward(self, x, adj, size=None, return_attention_weights=None):
        """
        Args:
            x: Union[Tensor, PairTensor]
            adj: Tensor[2, num_edges] or list of Tensor
            size: Size
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(adj, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        h, c = self.heads, self.out_channels
        # assert (not isinstance(adj, Tensor)) and h == len(adj), 'Number of heads is number of adjacency matrices'

        x_l, x_r, alpha_l, alpha_r, alpha_l_, alpha_r_ = None, None, None, None, None, None

        if isinstance(x, Tensor):
            x_l, x_r = x, None
        else:
            x_l, x_r = x[0], x[1]
        # assert x_l.dim() == 2, 'Static graphs not supported in `HGAConv`.'
        x_l = self.lin_l(x_l)
        if x_l.dim() == 2:
            alpha_l = torch.mm(x_l, self.att_l)
        else:  # x_l is 3D shape, matmul is in batched mode
            alpha_l = torch.matmul(x_l, self.att_l)

        if x_r is not None:
            x_r = self.lin_r(x_r)
            alpha_r = torch.mm(x_r, self.att_r)
            alpha_r_ = torch.mm(x_l, self.att_r)
            alpha_l_ = torch.mm(x_r, self.att_l)
            self.add_self_loops = False
        else:
            if x_l.dim() == 2:
                alpha_r = torch.mm(x_l, self.att_r)
            else:
                alpha_r = torch.matmul(x_l, self.att_r)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            num_nodes = x_l.shape[-2]
            num_nodes = size[1] if size is not None else num_nodes
            num_nodes = x_r.shape[-2] if x_r is not None else num_nodes
            if isinstance(adj, Tensor):
                adj = self_loop_augment(num_nodes, adj)  # TODO Bug found
            else:
                for i in range(len(adj)):
                    adj[i] = self_loop_augment(num_nodes, adj[i])

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        _x_ = (x_l, x_r) if x_r is not None else x_l
        _alpha_ = (alpha_l, alpha_r)
        alpha_ = (alpha_l_, alpha_r_)
        out = self.propagate(adj,
                             x=_x_,
                             alpha=_alpha_,
                             alpha_=alpha_,
                             size=size)

        alpha = self._alpha
        self._alpha = None

        if isinstance(out, Tensor):  # reshape here is equivalent to concatenation
            if len(x_l.shape) == 2:
                out = rearrange(out, '(h n) c -> n (h c)', h=h)
            else:
                out = rearrange(out, 't (h n) c -> t n (h c)', h=h)
        else:
            out = (out[0].reshape(-1, h * c), out[1].reshape(-1, h * c))

        if not self.concat:  # calculate mean
            if isinstance(out, Tensor):
                if len(x_l.shape) == 2:
                    out = reduce(out, 'n (h c) -> n c', 'mean', h=h)
                else:
                    out = reduce(out, 't n (h c) -> t n c', 'mean', h=h)
            else:
                out = (out[0].mean(dim=1), out[1].mean(dim=1))

        if self.bias is not None:
            if isinstance(out, Tensor):
                out += self.bias
            else:
                out = (out[0] + self.bias, out[1] + self.bias)
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (adj, alpha)
        else:
            return out

    def propagate(self, adj, size=None, **kwargs):
        # propagate_type: (x: OptPairTensor, alpha: PairTensor)
        size = self.__check_input__(adj, size)

        x = kwargs.get('x', Pr.empty)  # OptPairTensor
        alpha = kwargs.get('alpha', Pr.empty)  # PairTensor
        score = self.edge_score(adj=adj, a_l=alpha[0], a_r=alpha[1])
        if not isinstance(x, Tensor):
            alpha_ = kwargs.get('alpha_', Pr.empty)
            score_ = self.edge_score(adj=adj, a_l=alpha_[1], a_r=alpha_[0])
            score = (score, score_)

        out = self.message_and_aggregate(adj, x=x, score=score)

        return self.update(out)

    def _attention(self, adj, score):  # score: [num_edges, heads]
        alpha = fn.leaky_relu(score, self.negative_slope)
        if len(alpha.shape) == 2:
            alpha = softmax(alpha, adj[1])
        else:
            c = alpha.shape[-1]
            al = softmax(rearrange(alpha, 'b n c -> n (b c)'), adj[1])
            alpha = rearrange(al, 'n (b c) -> b n c', c=c)
        self._alpha = alpha
        return fn.dropout(alpha, p=self.dropout, training=self.training)

    def message_and_aggregate(self,
                              adj,
                              x,
                              score):
        """
        Args:
            adj:   Tensor or list(Tensor)
            x:     Union(Tensor, PairTensor) for bipartite graph
            score: Tensor or list(Tensor)
        """
        # for bipartite graph, x_l -> out_ and x_r -> out_l (interleaved)
        x_l, x_r, out_, out_l = None, None, None, None
        n, m = 0, 0
        if isinstance(x, Tensor):
            x_l = x
            n = m = x_l.shape[-2]
        else:
            x_l, x_r = x[0], x[1]
            (m, c2) = x_r.size()
            n = x_l.size(0)
            out_l = torch.zeros((m, c2, self.heads))

        if isinstance(adj, Tensor):
            if isinstance(score, Tensor):
                alpha = self._attention(adj, score)  # [num_edges, heads]
            else:
                alpha = self._attention(adj, score[0])  # [num_edges, heads]
                alpha_ = self._attention(torch.stack(
                    (adj[1], adj[0])), score[1])  # [num_edges, heads]

        else:  # adj is list of Tensor
            alpha = []
            for i in range(self.heads):
                alpha.append(self._attention(adj[i], score[i]))

        out_ = batched_spmm(alpha, adj, x_l, m, n)
        if x_r is None:
            return out_
            # return out_.permute(1, 0, 2)
        else:
            adj, alpha_ = batched_transpose(adj, alpha_)
            out_l = batched_spmm(alpha_, adj, x_r, n, m)
            return out_l, out_
            # return out_l.permute(1, 0, 2), out_.permute(1, 0, 2)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

#
# class DotProductAttention(nn.Module, ABC):
#     def __init__(self, dropout, **kwargs):
#         super(DotProductAttention, self).__init__(**kwargs)
#         self.dropout = nn.Dropout(dropout)
#
#     # `query`: (`batch_size`, #queries, `d`)
#     # `key`: (`batch_size`, #kv_pairs, `d`)
#     # `value`: (`batch_size`, #kv_pairs, `dim_v`)
#     # `valid_len`: either (`batch_size`, ) or (`batch_size`, xx)
#     def forward(self, query, key, value, valid_len=None):
#         d = query.shape[-1]
#         # Set transpose_b=True to swap the last two dimensions of key
#         scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
#         attention_weights = self.dropout(masked_softmax(scores, valid_len))
#         return torch.bmm(attention_weights, value)
#
#
# class MultiHeadAttention(nn.Module, ABC):
#     def __init__(self,
#                  key_size,
#                  query_size,
#                  value_size,
#                  num_hidden,
#                  num_heads,
#                  dropout,
#                  bias=False,
#                  **kwargs):
#         super(MultiHeadAttention, self).__init__(**kwargs)
#         self.num_heads = num_heads
#         self.attention = DotProductAttention(dropout)
#         self.wq = nn.Linear(query_size, num_hidden, bias=bias)
#         self.wk = nn.Linear(key_size, num_hidden, bias=bias)
#         self.wv = nn.Linear(value_size, num_hidden, bias=bias)
#         self.wo = nn.Linear(num_hidden, num_hidden, bias=bias)
#
#     def forward(self, query, key, value, valid_len):
#         # For self-attention, `query`, `key`, and `value` shape:
#         # (`batch_size`, `seq_len`, `dim`), where `seq_len` is the length of
#         # input sequence. `valid_len` shape is either (`batch_size`, ) or
#         # (`batch_size`, `seq_len`).
#
#         # Project and transpose `query`, `key`, and `value` from
#         # (`batch_size`, `seq_len`, `num_hidden`) to
#         # (`batch_size` * `num_heads`, `seq_len`, `num_hidden` / `num_heads`)
#         query = transpose_(self.wq(query), self.num_heads)
#         key = transpose_(self.wk(key), self.num_heads)
#         value = transpose_(self.wv(value), self.num_heads)
#
#         if valid_len is not None:
#             valid_len = torch.repeat_interleave(valid_len, repeats=self.num_heads, dim=0)
#
#         # For self-attention, `output` shape:
#         # (`batch_size` * `num_heads`, `seq_len`, `num_hidden` / `num_heads`)
#         output = self.attention(query, key, value, valid_len)
#
#         # `output_concat` shape: (`batch_size`, `seq_len`, `num_hidden`)
#         output_concat = transpose_(output, self.num_heads, reverse=True)
#         return self.wo(output_concat)
#
#
# class SynthesizedAttention(nn.Module, ABC):
#     def __init__(self, in_channels, dim_attention, out_channels,
#                  num_heads=3, num_layers=2,
#                  bias=True, banded=False, factorized=False,
#                  dropout=0.5):
#         """
#         Args:
#             bias: bool
#             banded: bool
#             factorized: bool
#             in_channels: int
#             dim_attention: int
#             out_channels: int
#             num_heads: int
#             num_layers: int
#         """
#         super(SynthesizedAttention, self).__init__()
#         self.banded = banded
#         self.dropout = dropout
#         self.num_heads = num_heads
#         self.factorized = factorized
#         self.dim_attention = dim_attention
#         self.synthesizers, self.synthesizers_ = None, None
#
#         dim_att = get_factorized_dim(dim_attention) if factorized else dim_attention
#         channels = [in_channels] * num_layers + [dim_att]
#         self.synthesizers = nn.ModuleList([
#             nn.Linear(in_features=channels[i],
#                       out_features=channels[i + 1],
#                       bias=bias) for i in range(num_layers)
#         ])
#         channels = [in_channels] * num_layers + [int(dim_attention / dim_att)]
#         self.synthesizers_ = nn.ModuleList([
#             nn.Linear(in_features=channels[i],
#                       out_features=channels[i + 1],
#                       bias=bias) for i in range(num_layers)
#         ]) if self.factorized else None
#         self.wv = nn.Linear(in_features=in_channels,
#                             out_features=out_channels,
#                             bias=bias)
#
#     def forward(self, x):
#         m, _ = x.shape[-2:]
#         v = fn.relu(self.wv(x))
#         y = x if self.factorized else None
#         for i in range(len(self.synthesizers)):
#             x = fn.relu(self.synthesizers[i](x))
#             if self.factorized:
#                 y = fn.relu(self.synthesizers_[i](y))
#         if self.factorized:
#             l_s, r_s = x.shape[-1], y.shape[-1]
#             x = x.repeat(1, 1, r_s)
#             y = y.repeat(1, 1, l_s)
#             x = x * y
#         if self.banded:
#             indices, values = to_band_sparse(x)
#             values = softmax(values, index=indices)
#             return spmm(indices, values, m, m, v)
#         return torch.bmm(fn.softmax(x, dim=-1), v)
#
#
# class AddNorm(nn.Module, ABC):
#     def __init__(self, normalized_shape, dropout, **kwargs):
#         super(AddNorm, self).__init__(**kwargs)
#         self.dropout = nn.Dropout(dropout)
#         self.ln = nn.LayerNorm(normalized_shape)
#
#     def forward(self, x, y):
#         return self.ln(self.dropout(y) + x)
