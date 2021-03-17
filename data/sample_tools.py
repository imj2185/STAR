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


def random_shift(t):
    f, n, c = t.shape
    data_shift = torch.zeros(t.shape)
    idx = torch.nonzero(((t != 0.).sum(-1).sum(-1) > 0) + 0)
    size = idx[-1] - idx[0]
    bias = torch.randint(0, f - size)
    data_shift[bias: bias + size, ...] = t[idx[0]: idx[-1], ...]
    return data_shift


def openpose_match(t):
    # C, T, V, M = t.shape
    t = rearrange('f, m, n, c -> c, f, n, m')
    c, f, n, m = t.shape
    assert (c == 3)
    score = t[2, :, :, :].sum(1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:f - 1]).argsort(0).reshape(f - 1, m)

    # data of frame 1
    xy1 = t[0:2, 0:f - 1, ...].unsqueeze(-1)
    # data of frame 2
    xy2 = t[0:2, 1:f, ...].unsqueeze(-2)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = torch.zeros((f, m), dtype=torch.int) - 1
    forward_map[0] = range(m)
    for p in range(m):
        choose = (rank == p)
        forward = distance[choose].argmin(axis=1)
        for s in range(f - 1):
            distance[s, :, forward[s]] = torch.tensor([math.inf])
        forward_map[1:][choose] = forward
    assert (torch.all(forward_map >= 0))

    # string data
    for s in range(f - 1):
        forward_map[s + 1] = forward_map[s + 1][forward_map[s]]

    # generate data
    new_data_torch = torch.zeros(t.shape)
    for s in range(t):
        new_data_torch[:, s, :, :] = t[:, s, :, forward_map[s]].transpose(1, 2, 0)
    t = new_data_torch

    # score sort
    trace_score = t[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    t = t[:, :, :, rank]

    return rearrange(t, 'c, f, n, m -> f, m, n, c')
