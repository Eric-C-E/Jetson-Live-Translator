from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from audio.format import decode_packed_24bit_stereo_to_mono
from audio.ringbuffer import FloatRingBuffer
from mt.opusmt_ct2 import OpusMTConfig, OpusMTTranslator
from net.protocol import MSG_TYPE_AUDIO, MSG_TYPE_TEXT, StreamParser, build_packet
from net.tcp_client import TCPServer
from s2t.commit import CommitConfig, SimpleCommitter
from s2t.whisper_engine import WhisperConfig, WhisperEngine
from utils.timing_helpers import RateLimiter

FLAG_LANG1_IN = 0x01
FLAG_LANG2_IN = 0x02
FLAG_LANG1_OUT = 0x04
FLAG_LANG2_OUT = 0x08


@dataclass
class PipelineConfig:
    host: str
    port: int
    sample_rate: int = 16000
    channels: int = 2
    text_max_payload: int = 128
    max_payload: int = 4096
    window_seconds: float = 4.0
    step_hz: float = 1.0
    min_window_seconds: float = 1.0
    max_buffer_seconds: float = 30.0
    lang1_label: str = "lang1"
    lang2_label: str = "lang2"
    whisper: WhisperConfig | None = None
    opus: OpusMTConfig | None = None
    commit: CommitConfig | None = None


@dataclass
class AudioChunk:
    samples: np.ndarray
    lang: str


def _chunk_bytes(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i:i + size]


def _flags_to_lang(flags: int, current: str, lang1: str, lang2: str) -> str:
    if flags & FLAG_LANG1_IN:
        return lang1
    if flags & FLAG_LANG2_IN:
        return lang2
    return current


def _lang_to_channel(lang: str, lang1: str, lang2: str) -> str:
    if lang == lang1:
        return "left"
    if lang == lang2:
        return "right"
    return "left"


def _output_flag(lang: str, lang1: str, lang2: str) -> int:
    if lang == lang1:
        return FLAG_LANG1_OUT
    return FLAG_LANG2_OUT


class PipelineWorker(threading.Thread):
    def __init__(self, config: PipelineConfig, audio_q: queue.Queue, tx_q: queue.Queue, stop: threading.Event):
        super().__init__(daemon=True)
        self.config = config
        self.audio_q = audio_q
        self.tx_q = tx_q
        self.stop = stop

        self.whisper = WhisperEngine(config.whisper)
        self.translator = OpusMTTranslator(config.opus)
        self.committer = SimpleCommitter(config.commit)

        self.buffer = FloatRingBuffer(int(config.max_buffer_seconds * config.sample_rate))
        self.current_lang = config.lang1_label
        self.rate = RateLimiter(config.step_hz)
        self.last_audio_ts = time.monotonic()

    def _transcribe_window(self) -> str:
        window_samples = int(self.config.window_seconds * self.config.sample_rate)
        audio = self.buffer.get_last(window_samples)
        return self.whisper.transcribe(audio, language=self.current_lang)

    def _process_text(self, text: str, src_lang: str, finalize: bool = False) -> None:
        logging.debug("Transcript lang=%s finalize=%s text=%r", src_lang, finalize, text)
        delta = self.committer.feed(text)
        if finalize:
            delta += self.committer.finalize(text)
        if not delta:
            return
        logging.debug("Commit delta lang=%s text=%r", src_lang, delta)
        translated = self.translator.translate(delta, src_lang)
        if translated:
            logging.debug("Translated lang=%s text=%r", src_lang, translated)
            self.tx_q.put((translated, src_lang))

    def _flush(self) -> None:
        if self.buffer.size == 0:
            return
        text = self._transcribe_window()
        if text:
            self._process_text(text, self.current_lang, finalize=True)
        self.buffer.clear()
        self.committer.reset()

    def run(self) -> None:
        logging.info("Pipeline worker started")
        while not self.stop.is_set():
            try:
                chunk: AudioChunk = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                if self.buffer.size and (time.monotonic() - self.last_audio_ts) >= self.config.min_window_seconds:
                    logging.debug("Idle flush after %.2fs without audio", time.monotonic() - self.last_audio_ts)
                    self._flush()
                continue

            self.last_audio_ts = time.monotonic()
            if chunk.lang != self.current_lang:
                logging.info("Language switch %s -> %s", self.current_lang, chunk.lang)
                self._flush()
                self.current_lang = chunk.lang

            self.buffer.append(chunk.samples)
            enough = self.buffer.size >= int(self.config.min_window_seconds * self.config.sample_rate)
            if enough and self.rate.allow():
                text = self._transcribe_window()
                if text:
                    self._process_text(text, self.current_lang)


class Coordinator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stop = threading.Event()
        self.audio_q: queue.Queue[AudioChunk] = queue.Queue(maxsize=200)
        self.tx_q: queue.Queue[tuple[str, str]] = queue.Queue()

        logging.info(
            "Initializing pipeline host=%s port=%s sample_rate=%s channels=%s window=%.2fs step_hz=%.2f",
            config.host,
            config.port,
            config.sample_rate,
            config.channels,
            config.window_seconds,
            config.step_hz,
        )
        self.server = TCPServer(config.host, config.port)
        self.parser = StreamParser(max_payload=config.max_payload)
        self.worker = PipelineWorker(config, self.audio_q, self.tx_q, self.stop)

        self.current_lang = config.lang1_label
        self.carry = b""

    def start(self) -> None:
        self.worker.start()
        if self.server.bound_host != self.config.host:
            logging.info(
                "Listening on %s:%s (requested %s:%s)",
                self.server.bound_host,
                self.config.port,
                self.config.host,
                self.config.port,
            )
        else:
            logging.info("Listening on %s:%s", self.config.host, self.config.port)
        try:
            while not self.stop.is_set():
                self._poll_network()
                self._drain_tx()
        except KeyboardInterrupt:
            logging.info("Shutdown requested")
            self.stop.set()
        except Exception:
            logging.exception("Coordinator crashed")
            self.stop.set()
            raise

    def _poll_network(self) -> None:
        data = self.server.poll(timeout_s=0.01)
        if not data:
            return
        for pkt in self.parser.feed(data):
            if pkt.msg_type != MSG_TYPE_AUDIO:
                continue

            self.current_lang = _flags_to_lang(
                pkt.flags, self.current_lang, self.config.lang1_label, self.config.lang2_label
            )
            payload = self.carry + pkt.payload
            frame_bytes = 3 * self.config.channels
            trim_len = len(payload) - (len(payload) % frame_bytes)
            self.carry = payload[trim_len:]
            payload = payload[:trim_len]
            if not payload:
                continue

            channel = _lang_to_channel(self.current_lang, self.config.lang1_label, self.config.lang2_label)
            samples = decode_packed_24bit_stereo_to_mono(
                payload, channels=self.config.channels, channel_select=channel
            )
            if samples.size == 0:
                continue

            try:
                self.audio_q.put_nowait(AudioChunk(samples=samples, lang=self.current_lang))
            except queue.Full:
                logging.warning("Audio queue full; dropping audio chunk")

    def _drain_tx(self) -> None:
        while True:
            try:
                text, src_lang = self.tx_q.get_nowait()
            except queue.Empty:
                return
            out_lang = self.config.lang2_label if src_lang == self.config.lang1_label else self.config.lang1_label
            flags = _output_flag(out_lang, self.config.lang1_label, self.config.lang2_label)
            raw = text.encode("utf-8")
            for chunk in _chunk_bytes(raw, self.config.text_max_payload):
                logging.debug(
                    "TX packet msg_type=%s flags=0x%02X payload_len=%s out_lang=%s text=%r",
                    MSG_TYPE_TEXT,
                    flags,
                    len(chunk),
                    out_lang,
                    chunk.decode("utf-8", errors="replace"),
                )
                packet = build_packet(MSG_TYPE_TEXT, flags, chunk)
                if not self.server.send(packet):
                    logging.warning("No active connection; drop text")
                    return
