# Audio Decode
# audio/decode_24bit.py
import numpy as np

def decode_packed_24bit_stereo_to_mono(payload: bytes,
                                       channels: int = 2,
                                       channel_select: str = "left") -> np.ndarray:
    SAMPLE_BYTES = 3

    if len(payload) < SAMPLE_BYTES:
        return np.empty((0,), dtype=np.float32)

    trim_len = len(payload) - (len(payload) % SAMPLE_BYTES)
    payload = payload[:trim_len]

    data = np.frombuffer(payload, dtype=np.uint8).reshape(-1, SAMPLE_BYTES)
    if data.size == 0:
        return np.empty((0,), dtype=np.float32)

    vals = (data[:, 0].astype(np.int32) |
            (data[:, 1].astype(np.int32) << 8) |
            (data[:, 2].astype(np.int32) << 16))
    neg = (vals & 0x800000) != 0
    vals[neg] -= 1 << 24

    if channels > 1:
        trim = vals.size - (vals.size % channels)
        frames = vals[:trim].reshape(-1, channels)
        if channel_select == "right":
            mono = frames[:, 1]
        elif channel_select == "mix":
            mono = frames.mean(axis=1)
        else:
            mono = frames[:, 0]
    else:
        mono = vals

    return mono.astype(np.float32) / float(1 << 23)