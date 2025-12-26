# net/protocol.py
from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import List, Tuple

HDR_FMT = "!BBBBI"
HDR_SIZE = struct.calcsize(HDR_FMT)

MAGIC = 0xAA
VERSION = 1

MSG_TYPE_AUDIO = 1
MSG_TYPE_TEXT  = 2

MAX_PAYLOAD = 4096

@dataclass
class Packet:
    msg_type: int
    flags: int
    payload: bytes

def pack_header(msg_type: int, flags: int, payload_len: int) -> bytes:
    return struct.pack(HDR_FMT, MAGIC, VERSION, msg_type, flags, payload_len)

def build_packet(msg_type: int, flags: int, payload: bytes) -> bytes:
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("payload too large for 32-bit len")
    return pack_header(msg_type, flags, len(payload)) + payload

class StreamParser:
    """
    Incremental TCP stream parser for your msg_hdr_t framing.
    Feed bytes, get back 0+ packets.
    """
    def __init__(self, max_payload: int = MAX_PAYLOAD):
        self._buf = bytearray()
        self._max_payload = max_payload

    def feed(self, data: bytes) -> List[Packet]:
        self._buf.extend(data)
        out: List[Packet] = []

        while True:
            if len(self._buf) < HDR_SIZE:
                break

            magic, version, msg_type, flags, payload_len = struct.unpack(
                HDR_FMT, self._buf[:HDR_SIZE]
            )

            if magic != MAGIC or version != VERSION:
                # Resync strategy: clear buffer (your current behavior)
                self._buf.clear()
                break

            if payload_len > self._max_payload:
                # Discard this oversized payload if fully received, otherwise wait.
                if len(self._buf) < HDR_SIZE + payload_len:
                    break
                del self._buf[:HDR_SIZE + payload_len]
                continue

            if len(self._buf) < HDR_SIZE + payload_len:
                break

            payload = bytes(self._buf[HDR_SIZE:HDR_SIZE + payload_len])
            del self._buf[:HDR_SIZE + payload_len]
            out.append(Packet(msg_type=msg_type, flags=flags, payload=payload))

        return out