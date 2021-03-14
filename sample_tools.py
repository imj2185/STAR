import torch
from einops import rearrange


def down_sample(t, step, random_sample=True):
    start = torch.randint(step, size=[1]) if random_sample else 0
    return t[start::step, ...].contiguous()


def temporal_slice(t, step):
    return


def mean_subtract(t, mean):
    if mean == 0:
        return
    return t - mean


def auto_padding(t, size, random_pad=True):
    if t.shape[0] < size:
        start = torch.randint(size - t.shape[0], size=[1]) if random_pad else 0
        t_ = torch.zeros([size] + list(t.shape[1:]))
        t_[start: start + t.shape[0], ...] = t
        return t_.contiguous()
    else:
        return t


def random_choose(t, size, auto_pad=True):
    f, n, c = t.shape
    if f == size:
        return t
    elif f < size:
        return t if not auto_pad else auto_padding(t, size, True)
    else:
        start = torch.randint(f - size)
        return t[start: start + size, ...].contiguous()


