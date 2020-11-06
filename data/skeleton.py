from __future__ import with_statement

import torch
from utility.linalg import power_adj


def skeleton_parts(num_joints=25):
    sk_adj = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24],
        [1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 22, 7, 24, 11]])
    return torch.cat([sk_adj,
                      power_adj(sk_adj, max(num_joints, max(sk_adj[1]) + 1), 2),
                      power_adj(sk_adj, max(num_joints, max(sk_adj[1]) + 1), 3)], dim=1)


def process_skeleton(path,
                     num_joints=25,
                     num_features=3,
                     use_motion_vector=False):
    import os.path as osp
    t = osp.split(path)[-1][-12:-9]
    with open(path, 'r') as f:
        lines = f.readlines()
        num_frames = int(lines[0])
        start = 1
        num_persons = int(lines[1])
        offset = int((len(lines) - 1) / num_frames)
        frames = [lines[start + 3 + i * offset:
                        start + 3 + i * offset + num_joints] for i in range(num_frames)]
        frames = process_frames(frames, num_joints, num_features, use_motion_vector)
        if num_persons == 2:
            frames_ = [lines[start + (i + 1) * offset - num_joints:
                             start + (i + 1) * offset + num_joints] for i in range(num_frames)]
            frames_ = process_frames(frames_, num_joints, num_features, use_motion_vector)
            frames = torch.cat([frames, frames_], dim=0)
        return frames, int(t)


def motion_vector(frames):
    # dimensions: num_frames num_joints num_features (x, y, z)
    mgd = torch.sqrt(torch.sum(torch.square(frames), dim=2))  # Magnitude
    mv = torch.zeros(frames.shape)
    mv[1:, :, :] = frames[1:, :, :] - frames[0: -1, :, :]
    mv = torch.div(mv.view(-1, 3), mgd.view(-1, 1)).view(frames.shape)
    # switch the order to {z, x, y} before applying acos
    mv = torch.cat([torch.acos(mv[:, :, [2, 0, 1]]),
                    torch.unsqueeze(mgd, dim=2)], dim=-1)
    return mv


def process_frames(frames, num_joints, num_features, use_motion_vector=False):
    fv = torch.zeros((len(frames), num_joints, num_features))
    for i in range(len(frames)):
        f = frames[i]
        for j in range(num_joints):
            vs = [float(n) for n in f[j].split()][0: num_features]
            fv[i, j, :] = torch.tensor(vs)
    if use_motion_vector:
        fv = torch.cat([fv, motion_vector(fv)], dim=-1)
    return fv
