#!/usr/bin/env python3
import socket
import struct
import time
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required for waveform display: pip install matplotlib") from exc

HOST = "192.168.0.165"
PORT = 3333

# ESP32 msg_hdr_t: magic(1), version(1), msg_type(1), flags(1), payload_len(4)
HDR_FMT = "!BBBBI"
HDR_SIZE = struct.calcsize(HDR_FMT)
MAGIC = 0xAA
VERSION = 1
MSG_TYPE_AUDIO = 1

# Audio format based on ESP32 I2S config (24-bit, stereo, 16 kHz)
SAMPLE_RATE = 16000
CHANNELS = 2
CHANNEL_SELECT = "left"  # left, right, or mix
SAMPLE_BYTES = 3  # packed 24-bit samples (little-endian)
VIEW_SECONDS = 2.0
MAX_BUFFER_SECONDS = 30.0
MAX_PAYLOAD = 4096

INPUT_LANGUAGE = "unknown"


def recv_all(sock: socket.socket, length: int) -> Optional[bytes]:
    data = bytearray()
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def decode_audio_payload(payload: bytes) -> np.ndarray:
    if len(payload) < SAMPLE_BYTES:
        return np.empty((0,), dtype=np.float32)
    trim_len = len(payload) - (len(payload) % SAMPLE_BYTES)
    if trim_len != len(payload):
        payload = payload[:trim_len]

    data = np.frombuffer(payload, dtype=np.uint8).reshape(-1, SAMPLE_BYTES)
    if data.size == 0:
        return np.empty((0,), dtype=np.float32)

    vals = (data[:, 0].astype(np.int32) |
            (data[:, 1].astype(np.int32) << 8) |
            (data[:, 2].astype(np.int32) << 16))
    neg = (vals & 0x800000) != 0
    vals[neg] -= 1 << 24

    if CHANNELS > 1:
        trim = vals.size - (vals.size % CHANNELS)
        frames = vals[:trim].reshape(-1, CHANNELS)
        if CHANNEL_SELECT == "right":
            mono = frames[:, 1]
        elif CHANNEL_SELECT == "mix":
            mono = frames.mean(axis=1)
        else:
            mono = frames[:, 0]
    else:
        mono = vals

    # Normalize to [-1.0, 1.0] from 24-bit signed range.
    return mono.astype(np.float32) / float(1 << 23)


def update_language(flags: int) -> str:
    if flags & 0x01:
        return "lang1"
    if flags & 0x02:
        return "lang2"
    return INPUT_LANGUAGE


def run_server() -> None:
    global INPUT_LANGUAGE
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    window_len = int(SAMPLE_RATE * VIEW_SECONDS)
    max_len = int(SAMPLE_RATE * MAX_BUFFER_SECONDS)
    audio_stream = np.empty((0,), dtype=np.float32)

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot(np.zeros(window_len, dtype=np.float32))
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0, window_len)
    ax.set_title("Waiting for audio...")
    fig.tight_layout()

    print(f"Listening on {HOST}:{PORT} ...")
    while True:
        conn, addr = srv.accept()
        print(f"Connected by {addr}")
        with conn:
            while True:
                hdr_bytes = recv_all(conn, HDR_SIZE)
                if hdr_bytes is None:
                    print("Connection closed")
                    break

                magic, version, msg_type, flags, payload_len = struct.unpack(HDR_FMT, hdr_bytes)
                if magic != MAGIC or version != VERSION:
                    print(f"Bad header: magic={magic} version={version}")
                    break

                if payload_len > MAX_PAYLOAD:
                    print(f"Payload too large ({payload_len}), discarding")
                    discard = recv_all(conn, payload_len)
                    if discard is None:
                        break
                    continue

                payload = recv_all(conn, payload_len)
                if payload is None:
                    print("Connection closed while receiving payload")
                    break

                if msg_type != MSG_TYPE_AUDIO:
                    continue

                new_lang = update_language(flags)
                if new_lang != INPUT_LANGUAGE:
                    INPUT_LANGUAGE = new_lang

                samples = decode_audio_payload(payload)
                if samples.size == 0:
                    continue

                audio_stream = np.concatenate((audio_stream, samples))
                if audio_stream.size > max_len:
                    audio_stream = audio_stream[-max_len:]
                window = audio_stream[-window_len:]
                if window.size < window_len:
                    pad = np.zeros(window_len - window.size, dtype=np.float32)
                    window = np.concatenate((pad, window))

                line.set_ydata(window)
                ax.set_title(f"Live Audio ({INPUT_LANGUAGE})")
                fig.canvas.draw_idle()
                plt.pause(0.001)

        print("Waiting for reconnect...")
        time.sleep(0.2)


if __name__ == "__main__":
    run_server()
