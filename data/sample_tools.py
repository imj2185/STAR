import math
import random

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
    idx = torch.nonzero(((t != 0.).sum(-1).sum(-1) > 0) + 0)
    t[idx[0]:idx[-1], ...] = t[idx[0]:idx[-1], ...] - mean
    return t


def auto_padding(t, size, random_pad=True):
    if t.shape[0] < size:
        s = torch.randint(size - t.shape[0], size=[1]) if random_pad else 0
        t_ = torch.zeros([size] + list(t.shape[1:]))
        t_[s: s + t.shape[0], ...] = t
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


def choice(t, num_samples, p=None):
    if p is None:
        p = torch.ones(t.shape[0]) / t.shape[0]
    idx = torch.multinomial(p, num_samples)
    return t[idx]


def random_move(t,
                angle_candidate=None,
                scale_candidate=None,
                transform_candidate=None,
                move_time_candidate=None):
    if move_time_candidate is None:
        move_time_candidate = [1]
    if transform_candidate is None:
        transform_candidate = [0.9, 1.0, 1.1]
    if scale_candidate is None:
        scale_candidate = [-0.2, -0.1, 0.0, 0.1, 0.2]
    if angle_candidate is None:
        angle_candidate = [-10., -5., 0., 5., 10.]
    # input dimension
    f, n, c = t.shape
    mt = random.choice(move_time_candidate)  # move time
    node = torch.cat([torch.arange(0, f, f * 1. / mt).round().int(), torch.tensor([f])])
    num_nodes = node.shape[0]
    angles = choice(angle_candidate, num_nodes)
    scales = choice(scale_candidate, num_nodes)
    transform_x = choice(transform_candidate, num_nodes)
    transform_y = choice(transform_candidate, num_nodes)

    a = torch.zeros(f)
    s = torch.zeros(f)
    t_x = torch.zeros(f)
    t_y = torch.zeros(f)
    pi = torch.tensor([math.pi])
    for i in range(num_nodes - 1):
        a[node[i]: node[i + 1]] = torch.linspace(angles[i],
                                                 angles[i + 1],
                                                 node[i + 1] - node[i]) * pi / 180
        s[node[i]: node[i + 1]] = torch.linspace(scales[i],
                                                 scales[i + 1],
                                                 node[i + 1] - node[i])
        t_x[node[i]: node[i + 1]] = torch.linspace(transform_x[i],
                                                   transform_x[i + 1],
                                                   node[i + 1] - node[i])
        t_y[node[i]: node[i + 1]] = torch.linspace(transform_y[i],
                                                   transform_y[i + 1],
                                                   node[i + 1] - node[i])
    theta = torch.tensor([[torch.cos(a) * s, -torch.sin(a) * s],
                          [torch.sin(a) * s, torch.cos(a) * s]]).permute(2, 0, 1)

    # perform transformation
    for i in range(f):
        xy = t[i, ..., 0:2]
        new_xy = torch.dot(theta[..., i], xy.transpose(-1, -2))
        new_xy[..., 0] += t_x[i]
        new_xy[..., 1] += t_y[i]
        t[i, ..., 0: 2] = new_xy.reshape(-1, 2)
    return t
