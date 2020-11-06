import numpy as np


def get_relative_coordinate_angles(sample,
                                   references=(4, 8, 12, 16)):
    # input: C, T, V, M
    c, t, v, m = sample.shape
    final_sample = np.zeros((4 * c, t, v, m))

    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    rel_coords = []
    for i in range(len(references)):
        ref_loc = sample[:, :, references[i], :]
        coords_diff = (sample.transpose((2, 0, 1, 3)) - ref_loc).transpose((1, 2, 0, 3))
        rel_coords.append(coords_diff)

    rel_angles = []
    for coords in rel_coords:
        flatten_x = coords[0, :, :, :].reshape((t * v * m))
        flatten_y = coords[1, :, :, :].reshape((t * v * m))
        flatten_z = coords[2, :, :, :].reshape((t * v * m))
        od_xy = np.arctan2(flatten_y, flatten_x + 1e-10) * (180 / np.pi)
        od_yz = np.arctan2(flatten_z, flatten_y + 1e-10) * (180 / np.pi)
        od_xz = np.arctan2(flatten_z, flatten_x + 1e-10) * (180 / np.pi)

        xy = od_xy.reshape((t, v, m))
        yz = od_yz.reshape((t, v, m))
        xz = od_xz.reshape((t, v, m))
        rel_angles.append(np.stack([xy, yz, xz]))

    # Shape: 4*C, t, V, M
    rel_angles = np.vstack(rel_angles)

    # Shape: C, T, V, M
    final_sample[:, start:end, :, :] = rel_angles
    return final_sample
