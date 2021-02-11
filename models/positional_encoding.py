import torch
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import breadth_first_tree
import networkx as nx
# from utility.linalg import bfs_enc


# sequential relative
# tree-based


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


def test():
    adj = torch.tensor(
       [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24],
        [1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 22, 7, 24, 11]
    ]).to('cuda')

    enc = tree_struct_pos_enc(adj, 25, 25, None, 'cuda')
    print(enc)


if __name__ == "__main__":
    test()
