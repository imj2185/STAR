import torch
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import spmm, transpose, spspmm
import torch.functional as fn
from einops import rearrange, repeat


def power_adj(adj, dim, p):
    val = torch.ones(adj.shape[1])
    ic, vc = spspmm(adj, val, adj, val, dim, dim, dim)
    if p > 2:
        for i in range(p - 2):
            ic, vc = spspmm(ic, vc, adj, val, dim, dim, dim)
    return ic


def batched_spmm(nzt, adj, x, m=None, n=None):
    """
    Args:
        nzt: Tensor [num_edges, heads]    -- non-zero tensor
        adj: Tensor or list(Tensor)       -- adjacency matrix (COO)
        x:   Tensor [num_nodes, channels] -- feature matrix
        m:   int
        n:   int
    """
    num_edges, heads = nzt.shape[-2:]
    num_nodes, channels = x.shape[-2:]
    # preparation of data
    # x_ = torch.cat(heads * [x])  # duplicate x for heads times
    # nzt_ = nzt.view(-1)
    x_ = repeat(x, 't n c -> t (h n) c', h=heads)
    nzt_ = rearrange(nzt, 't e h -> t (h e)')
    if isinstance(adj, Tensor):
        m = maybe_num_nodes(adj[0], m)
        n = max(num_nodes, maybe_num_nodes(adj[1], n))
        offset = torch.tensor([[m], [n]])
        adj_ = torch.cat([adj + offset * i for i in range(heads)], dim=1)
    else:  # adj is list of adjacency matrices
        assert heads == len(
            adj), "the number of heads and the number of adjacency matrices are not matched"
        m = max([maybe_num_nodes(adj_[0], m) for adj_ in adj])
        n = max([maybe_num_nodes(adj_[1], n) for adj_ in adj])
        offset = torch.tensor([[m], [n]])
        adj_ = torch.cat([adj[i] + offset * i for i in range(heads)], dim=1)
    if len(x.shape) == 2:
        out = spmm(adj_, nzt_, heads * m, heads * n, x_)
        return out.view(-1, m, channels)  # [heads, m, channels]
    else:
        _size = x_.shape[0]
        out = torch.stack([spmm(adj_, nzt_[i], heads * m, heads * n, x_[i]) for i in range(_size)])
        return out  # [batch, heads * num_nodes, channels]


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


if __name__ == "__main__":
    from data.dataset import skeleton_parts

    sk_adj = skeleton_parts()
