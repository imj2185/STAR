import torch
#from utility.linalg import bfs_enc


# sequential relative

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

# tree-based
def tree_struct_pos_enc(edges, num_tokens, max_chs, func=None, device=None):
    enc = torch.zeros(num_tokens, max_chs)
    for i in range(max_chs):
        enc[:, i] = bfs_enc(edges, root=i, device=device)
    if func is not None:
        enc = func(enc)
    return enc



adj = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24],
                    [1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 22, 7, 24, 11]]).to('cuda')

enc = tree_struct_pos_enc(adj, 25, 25, None, 'cuda')
print(enc)
