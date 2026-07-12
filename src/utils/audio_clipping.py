from __future__ import annotations

import numpy as np


MAX_AUDIO_SECONDS = 15.0


def middle_clip_audio(
    array: np.ndarray,
    sample_rate: int,
    max_seconds: float = MAX_AUDIO_SECONDS,
) -> np.ndarray:
    if max_seconds <= 0 or sample_rate <= 0:
        return array

    max_samples = int(round(sample_rate * max_seconds))
    if max_samples <= 0:
        return array

    sample_axis = 0
    if array.ndim == 2 and array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
        sample_axis = 1

    num_samples = array.shape[sample_axis]
    if num_samples <= max_samples:
        return array

    start = max((num_samples - max_samples) // 2, 0)
    end = start + max_samples

    if sample_axis == 0:
        return array[start:end]

    return array[:, start:end]
