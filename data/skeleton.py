from __future__ import with_statement

import torch

from utility.linalg import power_adj


def skeleton_parts(num_joints=25, dataset='ntu'):
    sk_adj = []
    if 'ntu' in dataset:
        sk_adj = torch.tensor(
            # [[0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
            #   9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17, 17, 18,
            #   18, 19, 20, 20, 20, 20, 21, 22, 22, 23, 24, 24],
            #  [1, 16, 12, 20, 0, 3, 20, 2, 20, 5, 4, 6, 7, 5, 22, 6, 20, 9,
            #   8, 10, 9, 11, 24, 10, 0, 13, 12, 14, 15, 13, 14, 0, 17, 16, 18, 17,
            #   19, 18, 8, 1, 4, 2, 22, 7, 21, 24, 11, 23]]
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24],
             [1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 22, 7, 24, 11]])
    elif 'kinetics' in dataset:
        sk_adj = torch.tensor(
            # [[4, 3, 7, 6, 13, 12, 10, 9, 11, 8, 5, 2, 0, 15, 14, 17,
            #   16, 3, 2, 6, 5, 12, 11, 9, 8, 5, 2, 1, 1, 1, 0, 0, 15, 14],
            #  [3, 2, 6, 5, 12, 11, 9, 8, 5, 2, 1, 1, 1, 0, 0, 15, 14,
            #   4, 3, 7, 6, 13, 12, 10, 9, 11, 8, 5, 2, 0, 15, 14, 17, 16]]
            [[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
             [1, 1, 2, 3, 1, 5, 6, 2, 8, 9, 5, 11, 12, 0, 0, 14, 15]]
        )
    else:
        sk_adj = None
    return torch.cat([sk_adj,
                      power_adj(sk_adj, max(num_joints, max(sk_adj[1]) + 1), 2),
                      power_adj(sk_adj, max(num_joints, max(sk_adj[1]) + 1), 3)], dim=1)


def process_skeleton(path,
                     num_joints,
                     dataset_name='ntu',
                     num_features=3,
                     use_motion_vector=True):
    import os.path as osp
    t = osp.split(path)[-1][-12:-9] if ('ntu' in dataset_name) else 0
    frames = []
    num_persons = 0
    if 'ntu' in dataset_name:
        with open(path, 'r') as f:
            lines = f.readlines()
            i = 1
            while i < len(lines) - 1:
                if lines[i] == '0\n':
                    i += 1
                    continue
                num_persons = int(lines[i])
                for j in range(num_persons):
                    frames.append(lines[i + 3 + j * 27: i + 1 + (j + 1) * 27])
                i += (1 + num_persons * 27)
        frames = process_frames(frames, num_persons, num_joints, num_features, use_motion_vector)

    elif 'kinetics' in dataset_name:
        # reference
        # https://github.com/yysijie/st-gcn/blob/master/feeder/feeder_kinetics.py
        import json
        with open(path, 'r') as f:
            video = json.load(f)
            num_frames = len(video['data'])
            if num_frames == 0:
                return None, None
            num_persons = max([len(video['data'][i]['skeleton']) for i in range(num_frames)])
            frames = torch.zeros(num_frames * num_persons, num_joints, num_features)
            i = 0
            for data in video['data']:
                fid = data['frame_index']
                for m, s in enumerate(data['skeleton']):  # m is person id, s is skeleton
                    if len(s) == 0:
                        continue
                    ft = torch.tensor([s['pose'][0::2],  # x
                                       s['pose'][1::2],  # y
                                       s['score']])
                    frames[i + m * num_frames] = ft.transpose(1, 0)
                i += 1
            t = video['label_index']
    return frames, int(t)


def motion_vector(frames):
    # dimensions: num_frames num_joints num_features (x, y, z)
    # Magnitude, 1e-6 for not dividing by zero
    mgd = torch.sqrt(torch.sum(torch.square(frames), dim=2)) + 1e-6
    mv = torch.zeros(frames.shape)
    mv[1:, :, :] = frames[1:, :, :] - frames[0: -1, :, :]
    mv = torch.div(mv.view(-1, 3), mgd.view(-1, 1)).view(frames.shape)
    # switch the order to {z, x, y} before applying acos
    mv = torch.cat([torch.acos(mv[:, :, [2, 0, 1]]),
                    torch.unsqueeze(mgd, dim=2)], dim=-1)
    return mv


def motion_vector_v2(frames, batch_index):
    pass


def process_frames(frames, num_persons, num_joints, num_features, use_motion_vector=False):
    fv = torch.zeros((len(frames), num_joints, num_features))
    frame_count = 0
    for p in range(num_persons):
        per_person = [frames[index] for index in range(len(frames)) if index % num_persons == p]
        for i in range(len(per_person)):
            f = per_person[i]
            for j in range(num_joints):
                vs = [float(n) for n in f[j].split()][0: num_features]
                fv[frame_count, j, :] = torch.tensor(vs)
            frame_count += 1
    if use_motion_vector:
        fv = torch.cat([fv, motion_vector(fv)], dim=-1)
    return fv
