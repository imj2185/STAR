import copy
from inspect import Parameter as Pr
from typing import Union, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as fn
from einops import rearrange, reduce
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros, ones
from torch_geometric.typing import OptTensor
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_scatter import scatter_sum
from torch_geometric.utils import degree

from models.attentions import CrossViewAttention, AddNorm, FeedForward, SparseAttention, LinearAttention
from utility.linalg import batched_spmm, batched_transpose, softmax_


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
            if len(a_l.shape) == 2:  # [num_edges, heads]
                return a_l[adj[1], :] + a_r[adj[0], :]
            else:  # [batch, num_edges, heads]
                a_l_ = rearrange(a_l, 'b n h -> n (b h)')
                a_r_ = rearrange(a_r, 'b n h -> n (b h)')
                out = a_l_[adj[1], :] + a_r_[adj[0], :]
                return rearrange(out, 'n (b h) -> b n h', h=a_l.shape[-1])
        a = []
        for i in range(len(adj)):
            a[i] = a_l[adj[i][1], i] + a_r[adj[i][0], i]
        return a  # (heads, [num_edges, 1])

    def _attention(self, adj, score):  # score: [num_edges, heads]
        alpha = fn.leaky_relu(score, self.negative_slope)
        alpha = softmax_(alpha, index=adj[1])  # , num_nodes=alpha.shape[-2])
        self._alpha = alpha
        return fn.dropout(alpha, p=self.dropout, training=self.training)

    def message_and_aggregate(self, adj, x, score):
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

    def forward(self, x, adj, size=None, return_attention_weights=None):
        """
        Args:
            x: Union[Tensor, PairTensor]
            adj: Tensor[2, num_edges] or list of Tensor
            size: Size
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(adj, attention_weights)`, holding the computed
                attention weight for each edge. (default: :obj:`None`)
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

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 activation=False,
                 dropout=0.,
                 bias=True):
        super(TemporalConv, self).__init__()
        pad = int((kernel_size - 1) / 2)

        self.conv = WSConv1d(  # nn.Conv1d
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
        # x = self.conv(x)
        return self.relu(x) if self.activation else x


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


class WSConv1d(nn.Conv1d):
    r"""Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters,
          of size
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    Note:

        Depending of the size of your kernel, several (of the last)
        columns of the input might be lost, because it is a valid
        `cross-correlation`_, and not a full `cross-correlation`_.
        It is up to the user to add proper padding.

    Note:

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(C_\text{in}=C_{in}, C_\text{out}=C_{in} \times K, ..., \text{groups}=C_{in})`.

    Note:
        In some circumstances when using the CUDA backend with CuDNN, this operator
        may select a nondeterministic algorithm to increase performance. If this is
        undesirable, you can try to make the operation deterministic (potentially at
        a performance cost) by setting ``torch.backends.cudnn.deterministic =
        True``.
        Please see the notes on :doc:`/notes/randomness` for background.


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weight of the module of shape
            :math:`(\text{out\_channels},
            \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`.
            The values of these weight are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weight are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.weight.size()[0], requires_grad=True))

    def standardize_weight(self, eps):
        var, mean = torch.var_mean(self.weight, dim=(1, 2), keepdims=True)
        fan_in = torch.prod(torch.tensor(self.weight.shape))

        scale = torch.rsqrt(torch.max(
            var * fan_in, torch.tensor(eps).to(var.device))) * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, x, eps=1e-4):
        weight = self.standardize_weight(eps)
        return fn.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(torch.nn.Module):
    r"""Applies layer normalization over each individual example in a batch
    of node features as described in the `"Instance Normalization: The Missing
    Ingredient for Fast Stylization" <https://arxiv.org/abs/1607.08022>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated across all nodes and all
    node channels separately for each object in a mini-batch.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
    """

    def __init__(self, in_channels, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()

        self.in_channels = in_channels
        self.eps = eps

        if affine:
            self.weight = Parameter(torch.Tensor([in_channels]))
            self.bias = Parameter(torch.Tensor([in_channels]))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, batch: OptTensor = None) -> Tensor:
        """"""
        if batch is None:
            x = x - x.mean()
            out = x / (x.std(unbiased=False) + self.eps)

        else:
            batch_size = int(batch.max()) + 1
            norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
            if len(x.shape) == 2:
                norm = norm.mul_(x.size()[-1]).view(-1, 1)
            else:
                norm = norm.mul_(x.size()[-1]).view(-1, 1, 1)

            mean = scatter_sum(x, batch, dim=0).sum(dim=-1, keepdim=True) / norm
            x = x - mean[batch]
            var = scatter_sum(x * x, batch, dim=0, dim_size=batch_size
                              ).sum(dim=-1, keepdim=True)
            var = var / norm
            out = x / (var.sqrt()[batch] + self.eps)

        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'


class CrossViewEncoderLayer(nn.Module):
    def __init__(self,
                 in_channels=6,
                 mdl_channels=64,
                 heads=8,
                 dropout=None):
        super(CrossViewEncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.mdl_channels = mdl_channels
        self.heads = heads
        if dropout is None:
            self.dropout = [0.5, 0.5, 0.5, 0.5]  # temp_conv, sparse_attention, add_norm, ffn
        else:
            self.dropout = dropout

        self.lin_qv = Linear(in_channels, mdl_channels * 2, bias=False)
        self.multi_head_attn = CrossViewAttention(in_channels=mdl_channels // heads,
                                                  dropout=dropout[1])

        self.add_norm_att = AddNorm(self.mdl_channels, False, self.dropout[2], self.heads)
        self.add_norm_ffn = AddNorm(self.mdl_channels, False, self.dropout[2], self.heads)
        self.ffn = FeedForward(self.mdl_channels, self.mdl_channels, self.dropout[3])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_qv.reset_parameters()
        self.add_norm_att.reset_parameters()
        self.add_norm_ffn.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, adj=None):
        # f, n, c = x.shape
        query, value = self.lin_qv(x).chunk(2, dim=-1)

        query = rearrange(query, 'f n (h c) -> f n h c', h=self.heads)
        t = self.multi_head_attn(query, adj)
        t = rearrange(t, 'f n h c -> f n (h c)', h=self.heads)

        x = self.add_norm_att(value, t)
        x = self.add_norm_ffn(x, self.ffn(x))

        return x


class SpatialEncoderLayer(nn.Module):
    def __init__(self,
                 in_channels=6,
                 mdl_channels=64,
                 heads=8,
                 beta=True,
                 dropout=None):
        super(SpatialEncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.mdl_channels = mdl_channels
        self.heads = heads
        self.beta = beta
        if dropout is None:
            self.dropout = [0.5, 0.5, 0.5, 0.5]  # temp_conv, sparse_attention, add_norm, ffn
        else:
            self.dropout = dropout

        # self.tree_key_weights = nn.Parameter(torch.randn(in_channels, in_channels), requires_grad=True)
        # self.tree_value_weights = nn.Parameter(torch.randn(in_channels, in_channels), requires_grad=True)

        self.lin_qkv = Linear(in_channels, mdl_channels * 3, bias=False)

        self.multi_head_attn = SparseAttention(in_channels=mdl_channels // heads,
                                               attention_dropout=dropout[1])

        self.add_norm_att = AddNorm(self.mdl_channels, False, self.dropout[2], self.heads)
        self.add_norm_ffn = AddNorm(self.mdl_channels, False, self.dropout[2], self.heads)
        self.ffn = FeedForward(self.mdl_channels, self.mdl_channels, self.dropout[3])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_qkv.reset_parameters()
        self.add_norm_att.reset_parameters()
        self.add_norm_ffn.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, adj=None):
        f, n, c = x.shape
        query, key, value = self.lin_qkv(x).chunk(3, dim=-1)

        # tree_pos_enc_key = torch.matmul(tree_encoding, self.tree_key_weights)
        # tree_pos_enc_value = torch.matmul(tree_encoding, self.tree_value_weights)

        # key = key + tree_pos_enc_key.unsqueeze(dim=0)
        # value = value + tree_pos_enc_value.unsqueeze(dim=0)

        query = rearrange(query, 'f n (h c) -> f n h c', h=self.heads)
        key = rearrange(key, 'f n(h c) -> f n h c', h=self.heads)
        value = rearrange(value, 'f n (h c) -> f n h c', h=self.heads)

        t = self.multi_head_attn(query, key, value, adj)
        t = rearrange(t, 'f n h c -> f n (h c)', h=self.heads)

        x = self.add_norm_att(x, t)
        x = self.add_norm_ffn(x, self.ffn(x))

        return x


class TemporalEncoderLayer(nn.Module):
    def __init__(self,
                 in_channels=6,
                 mdl_channels=64,
                 heads=8,
                 beta=False,
                 dropout=0.1):
        super(TemporalEncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.mdl_channels = mdl_channels
        self.heads = heads
        self.beta = beta
        if dropout is None:
            self.dropout = [0.5, 0.5, 0.5, 0.5]  # temp_conv, sparse_attention, add_norm, ffn
        else:
            self.dropout = dropout

        self.lin_qkv = Linear(in_channels, mdl_channels * 3, bias=False)

        self.multi_head_attn = LinearAttention(in_channels=mdl_channels // heads,
                                               attention_dropout=self.dropout[0])

        self.add_norm_att = AddNorm(self.mdl_channels, self.beta, self.dropout[2], self.heads)
        self.add_norm_ffn = AddNorm(self.mdl_channels, False, self.dropout[2], self.heads)
        self.ffn = FeedForward(self.mdl_channels, self.mdl_channels, self.dropout[3])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_qkv.reset_parameters()
        self.add_norm_att.reset_parameters()
        self.add_norm_ffn.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, bi=None):
        f, n, c = x.shape

        query, key, value = self.lin_qkv(x).chunk(3, dim=-1)

        query = rearrange(query, 'n f (h c) -> n f h c', h=self.heads)
        key = rearrange(key, 'n f (h c) -> n f h c', h=self.heads)
        value = rearrange(value, 'n f (h c) -> n f h c', h=self.heads)

        t = self.multi_head_attn(query, key, value, bi)
        t = rearrange(t, 'n f h c -> n f (h c)', h=self.heads)

        x = self.add_norm_att(x, t)
        x = self.add_norm_ffn(x, self.ffn(x))

        return x
