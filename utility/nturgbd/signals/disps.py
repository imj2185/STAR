import numpy as np


def get_displacements(sample):
    final_sample = np.zeros(sample.shape)
    valid_frames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = valid_frames.argmax()
    end = len(valid_frames) - valid_frames[::-1].argmax()
    sample = sample[:, start: end, :, :]

    t = sample.shape[1]
    # Shape: C, t-1, V, M
    displacement = sample[:, 1:, :, :] - sample[:, :-1, :, :]
    # Shape: C, T, V, M
    final_sample[:, start:end - 1, :, :] = displacement

    return final_sample
