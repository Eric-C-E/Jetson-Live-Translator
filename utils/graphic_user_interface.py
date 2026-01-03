from __future__ import annotations

import time
import threading
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Tuple

import numpy as np


@dataclass
class _SampleWindow:
    window_seconds: float
    target_hz: float


class AudioIntensityPlotter:
    """
    Lightweight Tkinter waveform plotter for troubleshooting audio input.
    Stores a quantized history of intensity values and scrolls over time.
    """

    def __init__(self, window_seconds: float = 10.0, target_hz: float = 20.0):
        self._cfg = _SampleWindow(window_seconds=window_seconds, target_hz=target_hz)
        self._period = 1.0 / max(target_hz, 1e-6)
        self._samples: Deque[Tuple[float, float]] = deque()
        self._lock = threading.Lock()
        self._accum_sum = 0.0
        self._accum_count = 0
        self._last_emit = 0.0
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._on_close: Callable[[], None] | None = None

    def push_samples(self, samples: np.ndarray, sample_rate: int) -> None:
        if samples.size == 0:
            return
        level = float(np.mean(np.abs(samples)))
        level = max(0.0, min(1.0, level))
        self._push_level(level, time.monotonic())

    def _push_level(self, level: float, ts: float) -> None:
        with self._lock:
            self._accum_sum += level
            self._accum_count += 1

            if self._last_emit == 0.0:
                self._last_emit = ts

            if (ts - self._last_emit) < self._period:
                return

            avg = self._accum_sum / max(self._accum_count, 1)
            self._accum_sum = 0.0
            self._accum_count = 0
            self._last_emit = ts
            self._samples.append((ts, avg))

            cutoff = ts - self._cfg.window_seconds
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

    def run(self, on_close: Callable[[], None] | None = None) -> None:
        self._on_close = on_close
        self._root = tk.Tk()
        self._root.title("Jetson-Live-Translator Audio Intensity")
        self._root.protocol("WM_DELETE_WINDOW", self._handle_close)
        self._canvas = tk.Canvas(self._root, width=800, height=200, bg="#0b0b0b")
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._schedule_redraw()
        self._root.mainloop()

    def _handle_close(self) -> None:
        if self._on_close is not None:
            self._on_close()
        if self._root is not None:
            self._root.destroy()

    def _schedule_redraw(self) -> None:
        if self._root is None:
            return
        self._redraw()
        self._root.after(int(self._period * 1000), self._schedule_redraw)

    def _redraw(self) -> None:
        if self._canvas is None:
            return
        width = self._canvas.winfo_width()
        height = self._canvas.winfo_height()
        if width <= 2 or height <= 2:
            return

        self._canvas.delete("waveform")
        now = time.monotonic()
        cutoff = now - self._cfg.window_seconds
        with self._lock:
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()
            samples = list(self._samples)

        if not samples:
            return

        points = []
        for ts, level in samples:
            x = (ts - cutoff) / self._cfg.window_seconds
            x = max(0.0, min(1.0, x))
            px = x * width
            py = height - (level * height)
            points.extend((px, py))

        if len(points) >= 4:
            self._canvas.create_line(*points, fill="#4de26c", width=2, tags="waveform")
