from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel


@dataclass
class WhisperConfig:
    model_size: str = "tiny"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: Optional[str] = None
    beam_size: int = 1
    no_speech_threshold: float = 0.9


class WhisperEngine:
    def __init__(self, config: WhisperConfig | None = None):
        self.config = config or WhisperConfig()
        logging.info(
            "Loading Whisper model size=%s device=%s compute_type=%s language=%s",
            self.config.model_size,
            self.config.device,
            self.config.compute_type,
            self.config.language,
        )
        self.model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        if audio.size == 0:
            return ""
        lang = self.config.language if language is None else language
        if not lang:
            raise ValueError("Whisper language must be provided to avoid detection.")
        segments, _info = self.model.transcribe(
            audio,
            language=lang,
            beam_size=self.config.beam_size,
            no_speech_threshold=self.config.no_speech_threshold,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        parts = [seg.text for seg in segments]
        return "".join(parts).strip()
