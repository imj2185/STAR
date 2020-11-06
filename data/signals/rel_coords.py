import numpy as np


def get_relative_coordinates(sample,
                             references=(4, 8, 12, 16)):
    # input: C, T, V, M
    c, t, v, m = sample.shape
    final_sample = np.zeros((4 * c, t, v, m))

    valid_frames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = valid_frames.argmax()
    end = len(valid_frames) - valid_frames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    rel_coords = []
    for i in range(len(references)):
        ref_loc = sample[:, :, references[i], :]
        coords_diff = (sample.transpose((2, 0, 1, 3)) - ref_loc).transpose((1, 2, 0, 3))
        rel_coords.append(coords_diff)

    # Shape: 4*C, t, V, M 
    rel_coords = np.vstack(rel_coords)
    # Shape: C, T, V, M
    final_sample[:, start:end, :, :] = rel_coords
    return final_sample
