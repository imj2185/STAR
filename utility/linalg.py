from typing import Optional

import torch
import torch.functional as fn
from einops import rearrange, repeat
from fast_transformers.masking import BaseMask, FullMask
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add, scatter
from torch_sparse import transpose, spspmm  # , spmm


def power_adj(adj, dim, p):
    nnz = torch.ones(adj.shape[1])
    ic, vc = spspmm(adj, nnz, adj, nnz, dim, dim, dim)
    if p > 2:
        for _ in range(p - 2):
            ic, vc = spspmm(ic, vc, adj, nnz, dim, dim, dim)
    return ic


def softmax(src: Tensor, index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None, num_nodes: Optional[int] = None,
            dim: int = 0) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.
    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)
    :rtype: :class:`Tensor`
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        out = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src, index, dim, dim_size=N, reduce='max')
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)


def spmm_(indices, nz, m, n, dense, dim=-3):
    """Sparse matrix multiplication, it supports tensor
    with dimension size more than 2, and the code is inspired by:
    "PyTorch Sparse"[https://tinyurl.com/ycn2nkdr]
    :argument
        indices (:class: `LongTensor`): tensor of indices of sparse matrix.
        nz (:class: `Tensor`): tensor of nonzero of sparse matrix.
        m (int): The first dimension of corresponding dense matrix.
        n (int): The second dimension of corresponding dense matrix.
        d (:class:`Tensor`): tensor of dense matrix
    """
    assert n == dense.shape[dim]
    rows, cols = indices
    dense = dense if dense.dim() > 1 else dense.unsqueeze(-1)
    out = dense.index_select(dim, cols) * nz.unsqueeze(-1)
    return scatter_add(out, rows, dim=dim, dim_size=m)


def batched_spmm(nzt, adj, x, m=None, n=None, dim=-2):
    """
    Args:
        nzt: Tensor [num_edges, heads]    -- non-zero tensor
        adj: Tensor or list(Tensor)       -- adjacency matrix (COO)
        x:   Tensor [num_nodes, channels] -- feature matrix
        m:   int
        n:   int
        dim: int
    """
    num_edges, heads = nzt.shape[-2:]
    num_nodes, channels = x.shape[-2:]
    # preparation of data
    x_ = repeat(x, '... n c -> ... (h n) c', h=heads).to(x.device)
    nzt_ = rearrange(nzt, '... e h -> ... (h e)')
    if isinstance(adj, Tensor):
        m = maybe_num_nodes(adj[0], m)
        n = max(num_nodes, maybe_num_nodes(adj[1], n))
        offset = torch.tensor([[m], [n]]).to(x.device)
        adj_ = torch.cat([adj + offset * k for k in range(heads)], dim=1)
    else:  # adj is list of adjacency matrices
        assert heads == len(
            adj), "the number of heads and the number of adjacency matrices are not matched"
        m = max([maybe_num_nodes(adj_[0], m) for adj_ in adj])
        n = max([maybe_num_nodes(adj_[1], n) for adj_ in adj])
        offset = torch.tensor([[m], [n]]).to(x.device)
        adj_ = torch.cat([adj[k] + offset * k for k in range(heads)], dim=1)
    return spmm_(adj_, nzt_, heads * m, heads * n, x_, dim=dim)


def batched_transpose(adj, value, m=None, n=None):
    """
    Args:
        adj: Tensor or list of Tensor
        value: Tensor [num_edges, ]
        m: int
        n: int
    """
    if isinstance(adj, Tensor):
        m = maybe_num_nodes(adj[0], m)
        n = maybe_num_nodes(adj[1], n)
        return transpose(adj, value, m, n)
    else:  # adj is a list of Tensor
        adj_ = [None] * value.shape[1]
        vs = torch.zeros(value.shape)
        m = max([maybe_num_nodes(a_[0], m) for a_ in adj])
        n = max([maybe_num_nodes(a_[1], n) for a_ in adj])
        for j in range(len(adj)):
            adj_[j], vs[:, j] = transpose(adj[j], value[:, j], m, n)
        return adj_, vs


def transpose_(x, num_heads, reverse=False):
    shape = (-1, num_heads, x.shape[1], x.shape[2]) if reverse \
        else (x.shape[0], x.shape[1], num_heads, -1)
    x = torch.reshape(x, shape)
    x = x.permute(0, 2, 1, 3)
    shape = (x.shape[0], x.shape[1], -1) if reverse \
        else (-1, x.shape[2], x.shape[3])
    output = torch.reshape(x, shape)
    return output


def masked_softmax(x, valid_len):
    """Perform softmax by filtering out some elements."""
    # x: 3-D tensor, valid_len: 1-D or 2-D tensor
    if valid_len is None:
        return fn.softmax(x, dim=-1)
    else:
        shape = x.shape
        if valid_len.dim() == 1:
            valid_len = torch.repeat_interleave(valid_len, repeats=shape[1],
                                                dim=0)
        else:
            valid_len = valid_len.reshape(-1)
        # Fill masked elements with a large negative, whose exp is 0
        x = sequence_mask(x.reshape(-1, shape[-1]), valid_len, value=-1e6)
        return fn.softmax(x.reshape(shape), dim=-1)


def sequence_mask(x, valid_len, value=0):
    max_len = x.size(1)
    mask = torch.arange(max_len, dtype=torch.float32,
                        device=x.device)[None, :] < valid_len[:, None]
    x[~mask] = value
    return x


def get_factorized_dim(dim):
    import math
    s = math.sqrt(dim)
    for i in range(int(s), dim):
        if dim % i == 0:
            return i
    return s


def to_band_sparse(x, lower=True):
    num_nodes, num_band = x.shape[-2:]
    import itertools as its
    indices = torch.tensor([(i, j) for (i, j) in its.product(range(num_nodes),
                                                             range(num_nodes))
                            if i >= j and i - j < num_band]).transpose(1, 0)
    if not lower:
        idx = indices.clone()
        indices[0, :] = idx[1, :]
        indices[1, :] = idx[0, :]
        t = torch.sort(indices[0, :])
        indices = indices[:, t]
    b = x.view(-1, torch.prod(torch.tensor(x.shape[-2:])))
    return indices, b[:, 0: indices.shape[-1]]


class BatchedMask(BaseMask):
    @property
    def bool_matrix(self):
        idx = self.bi + 1
        _idx = idx.unsqueeze(dim=-1)
        idx_ = idx.unsqueeze(dim=-2)
        msk = _idx * idx_
        return (msk / (_idx ** 2)) == (msk / (idx_ ** 2))

    def __init__(self, bi=None):
        super(BatchedMask, self).__init__()
        self.bi = bi


def make_attn_mask(seq_len, bi):
    msk = FullMask()
    return None


if __name__ == "__main__":
    from data.dataset import skeleton_parts

    sk_adj = skeleton_parts()

    x = torch.ones(3, 5, 3)
    x[0] *= 2
    x[1] *= 3
    for i in range(m):
        x[..., i, :] *= i
    adj = torch.tensor([[0, 1, 1, 1, 2, 3, 4, 4], [1, 0, 2, 4, 1, 4, 1, 3]])
