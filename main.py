# Constructs audio_q, final_q, tx_q
# constructs whisper (s2t)
# constructs translator (Opus-MT Marian)
# constructs committer
# starts threads + asyncio client/server

# main.py
import numpy as np
from net.tcp_client import TCPServer
from net.protocol import StreamParser, MSG_TYPE_AUDIO, MSG_TYPE_TEXT, build_packet
from audio.format import decode_packed_24bit_stereo_to_mono
from utils.timing_helpers import RateLimiter

HOST = "192.168.0.165"
PORT = 3333

SAMPLE_RATE = 16000
CHANNELS = 2
CHANNEL_SELECT = "left"

# Flags you already use
FLAG_SCREEN1 = 0x04
FLAG_SCREEN2 = 0x08

TEXT_MAX_PAYLOAD = 128
TEXT_FLAGS = FLAG_SCREEN1

def chunk_bytes(b: bytes, n: int):
    for i in range(0, len(b), n):
        yield b[i:i+n]

def main():
    srv = TCPServer(HOST, PORT)
    parser = StreamParser()
    live_rate = RateLimiter(hz=2.0)   # you said 1–2 Hz is fine

    audio_stream = np.empty((0,), dtype=np.float32)
    carry = b""

    while True:
        data = srv.poll(timeout_s=0.01)
        if data:
            for pkt in parser.feed(data):
                if pkt.msg_type != MSG_TYPE_AUDIO:
                    continue

                payload = carry + pkt.payload

                # Ensure full frames (3 bytes * channels)
                frame_bytes = 3 * CHANNELS
                trim_len = len(payload) - (len(payload) % frame_bytes)
                carry = payload[trim_len:]
                payload = payload[:trim_len]

                samples = decode_packed_24bit_stereo_to_mono(
                    payload, channels=CHANNELS, channel_select=CHANNEL_SELECT
                )
                if samples.size == 0:
                    continue

                # Accumulate or forward to Whisper worker later
                audio_stream = np.concatenate((audio_stream, samples))

                # Non-prod demo: every 2 Hz send a dumb status line
                if live_rate.allow():
                    txt = f"samples={audio_stream.size}"
                    raw = txt.encode("utf-8")

                    # Your current ESP32 expects TEXT_MAX_PAYLOAD chunks
                    for c in chunk_bytes(raw, TEXT_MAX_PAYLOAD):
                        srv.send(build_packet(MSG_TYPE_TEXT, TEXT_FLAGS, c))

if __name__ == "__main__":
    main()