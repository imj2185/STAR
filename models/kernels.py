import torch
import math
from einops import rearrange


def orthogonal_matrix_chunks(cols, batch, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((batch, cols, cols), device=device)
    q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t.to(device), (q, r))
    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = torch.diagonal(r, 0, dim1=1, dim2=2)
        q *= d.unsqueeze(-1).sign()
    return q.transpose(2, 1)


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None):
    nb_full_blocks = int(nb_rows / nb_columns) + 1

    q = orthogonal_matrix_chunks(nb_columns, nb_full_blocks, qr_uniform_q=qr_uniform_q, device=device)

    final_matrix = rearrange(q, 'b c r -> (b c) r')[0: nb_rows]

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix
