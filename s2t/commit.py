from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


def _lcp_all(items: Deque[str]) -> str:
    if not items:
        return ""
    shortest = min(items, key=len)
    for i, ch in enumerate(shortest):
        for s in items:
            if s[i] != ch:
                return shortest[:i]
    return shortest


def _overlap_suffix_prefix(left: str, right: str) -> int:
    """
    Return the longest suffix length of left that is a prefix of right.
    """
    max_len = min(len(left), len(right))
    for size in range(max_len, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


@dataclass
class CommitConfig:
    history_len: int = 3
    min_commit_chars: int = 1
    min_overlap_chars: int = 4


class SimpleCommitter:
    """
    Tracks stable prefixes across recent transcripts and only commits
    text that stays consistent for a short history window.
    """

    def __init__(self, config: CommitConfig | None = None):
        self.config = config or CommitConfig()
        self._history: Deque[str] = deque(maxlen=self.config.history_len)
        self._committed = ""

    def reset(self) -> None:
        self._history.clear()
        self._committed = ""

    @property
    def committed(self) -> str:
        return self._committed

    def feed(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if self._committed and not text.startswith(self._committed):
            overlap = _overlap_suffix_prefix(self._committed, text)
            if overlap >= self.config.min_overlap_chars:
                self._committed = self._committed[-overlap:] if overlap else ""
            self._history.clear()
        self._history.append(text)

        stable = _lcp_all(self._history)
        if len(stable) <= len(self._committed):
            return ""
        if len(stable) - len(self._committed) < self.config.min_commit_chars:
            return ""

        delta = stable[len(self._committed):]
        self._committed = stable
        return delta

    def finalize(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if text.startswith(self._committed):
            delta = text[len(self._committed):]
            self._committed = text
            return delta
        if self._committed:
            overlap = _overlap_suffix_prefix(self._committed, text)
            if overlap:
                delta = text[overlap:]
                self._committed = text
                self._history.clear()
                return delta
        self._committed = text
        self._history.clear()
        return text
        return ""
