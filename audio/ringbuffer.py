from __future__ import annotations

import numpy as np


class FloatRingBuffer:
    def __init__(self, max_samples: int):
        self._max_samples = max(1, int(max_samples))
        self._data = np.empty((0,), dtype=np.float32)

    @property
    def size(self) -> int:
        return int(self._data.size)

    def clear(self) -> None:
        self._data = np.empty((0,), dtype=np.float32)

    def append(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        samples = samples.astype(np.float32, copy=False)
        self._data = np.concatenate((self._data, samples))
        if self._data.size > self._max_samples:
            self._data = self._data[-self._max_samples:]

    def get_last(self, count: int) -> np.ndarray:
        if self._data.size == 0:
            return np.empty((0,), dtype=np.float32)
        if self._data.size <= count:
            return self._data.copy()
        return self._data[-count:].copy()
