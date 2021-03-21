import networkx as nx
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_tree
# sequential relative
# tree-based
from torch import nn


# from utility.linalg import bfs_enc


def bfs_enc(edges, root, device):
    if not isinstance(edges, torch.Tensor):
        edges = torch.tensor(edges)
        if edges.shape[0] > edges.shape[1]:
            edges = torch.transpose(edges, 1, 0)
    num_nodes = max(edges[0]) + 1
    hops2root = torch.ones(num_nodes).to(device)
    ancestors = torch.ones(num_nodes, dtype=torch.long).to(device)
    # for j in range(edges.shape[1]):
    #     ancestors[edges[1, j]] = edges[0, j]
    ancestors[edges[1]] = edges[0]
    ancestors[root] = root
    while torch.sum(torch.eq(ancestors, root)) != num_nodes:
        ancestors = ancestors[ancestors]
        hops2root = torch.where(torch.eq(ancestors, root), hops2root, hops2root + 1)
    return hops2root


def tree_struct_pos_enc(adj, max_chs, func=None, device=None):
    row = np.array(adj[0].cpu().tolist())
    col = np.array(adj[1].cpu().tolist())
    num_tokens = max(row) + 1
    data = np.ones(len(row))
    gr = coo_matrix((data, (row, col)), shape=(25, 25))
    enc = torch.zeros(num_tokens, max_chs).to(device)
    for i in range(min(max_chs, num_tokens)):
        g = nx.from_scipy_sparse_matrix(breadth_first_tree(gr, i, directed=False).tocoo(),
                                        create_using=nx.DiGraph)
        edges = torch.tensor(list(g.edges), dtype=torch.long).transpose(1, 0)
        enc[:, i] = bfs_enc(edges, root=i, device=device)
    if func is not None:
        enc = func(enc)
    return enc


class SeqPosEncoding(nn.Module):
    def __init__(self,
                 model_dim: int,
                 use_weight=False):
        """ Sequential Positional Encoding
            This kind of encoding uses the trigonometric functions to
            incorporate the relative position information into the input
            sequence
        :param model_dim (int): the dimension of the token (feature channel length)
        """
        super(SeqPosEncoding, self).__init__()
        self.model_dim = model_dim
        scale = model_dim ** -0.5
        if use_weight:
            self.weight = nn.Parameter(torch.randn(model_dim, model_dim) * scale)
        else:
            self.weight = None

    @staticmethod
    def segment(pos, bi, device):
        idx = torch.tensor((torch.cat([torch.tensor([1]).to(device),
                                       bi[1:] - bi[:-1]]) == 1))
        offset = torch.nonzero(idx, as_tuple=True)[0]
        return pos - offset[bi]

    def forward(self, x, bi=None) -> torch.Tensor:
        d = self.model_dim
        sequence_length = x.shape[-2]
        pos = torch.arange(sequence_length, dtype=torch.float).to(x.device)
        if bi is not None:
            pos = self.segment(pos, bi, x.device)
        pos = pos.reshape(1, -1, 1).to(x.device)
        dim = torch.arange(d, dtype=torch.float).reshape(1, 1, -1).to(x.device)
        phase = (pos / 1e4) ** (dim / d)
        assert x.shape[-2] == sequence_length and x.shape[-1] == self.model_dim
        if self.weight is not None:
            return x + torch.matmul(torch.where(dim.long() % 2 == 0,
                                                torch.sin(phase),
                                                torch.cos(phase)),
                                    self.weight)
        return x + torch.where(dim.long() % 2 == 0,
                               torch.sin(phase),
                               torch.cos(phase))


def test():
    from ..data.dataset3 import skeleton_parts
    adj = skeleton_parts(cat=False)
    enc = tree_struct_pos_enc(adj, 25, 25, None, 'cuda')
    print(enc)


if __name__ == "__main__":
    test()
