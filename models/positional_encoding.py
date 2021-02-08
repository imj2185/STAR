import torch

from utility.linalg import bfs_enc


# sequential relative

# tree-based
def tree_struct_pos_enc(edges, num_tokens, max_chs, func=None, device=None):
    enc = torch.zeros(num_tokens, max_chs)
    for i in range(max_chs):
        enc[:, i] = bfs_enc(edges, root=i, device=device)
    if func is not None:
        enc = func(enc)
    return enc
