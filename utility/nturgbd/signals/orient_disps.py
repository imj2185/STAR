import numpy as np


def get_oriented_displacements(sample):
    # input: C, T, V, M
    c, t, v, m = sample.shape
    final_sample = np.zeros((c, t, v, m))

    valid_frames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = valid_frames.argmax()
    end = len(valid_frames) - valid_frames[::-1].argmax()
    sample = sample[:, start: end, :, :]

    c, t, v, m = sample.shape
    # Shape: C, t-1, V, M
    disp = sample[:, 1:, :, :] - sample[:, :-1, :, :]
    person1 = disp[:, :, :, 0]
    person2 = disp[:, :, :, 1]
    cog1 = person1.mean(axis=2).mean(axis=1)
    cog2 = person2.mean(axis=2).mean(axis=1)

    person1 = (person1.transpose(2, 1, 0) - cog1).transpose((2, 1, 0))
    person2 = (person2.transpose(2, 1, 0) - cog2).transpose((2, 1, 0))
    disp = np.stack([person1, person2], axis=3)

    flatten_x = disp[0, :, :, :].reshape(((t - 1) * v * m))
    flatten_y = disp[1, :, :, :].reshape(((t - 1) * v * m))
    flatten_z = disp[2, :, :, :].reshape(((t - 1) * v * m))
    od_xy = np.arctan2(flatten_y, flatten_x + 1e-10) * (180 / np.pi)
    od_yz = np.arctan2(flatten_z, flatten_y + 1e-10) * (180 / np.pi)
    od_xz = np.arctan2(flatten_z, flatten_x + 1e-10) * (180 / np.pi)

    xy = od_xy.reshape(((t - 1), v, m))
    yz = od_yz.reshape(((t - 1), v, m))
    xz = od_xz.reshape(((t - 1), v, m))

    # Shape: C, t-1, V, M
    orient_disps = np.stack([xy, yz, xz])
    # Shape C, T, V, M
    final_sample[:, start:end - 1, :, :] = orient_disps

    return final_sample
