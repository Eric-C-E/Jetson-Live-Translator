## TCP TEST PROGRAM
# receives audio TCP packets
# sends text TCP packets


#!/usr/bin/env python3
import socket
import struct

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox
except ImportError as exc:
    raise SystemExit("matplotlib is required for waveform display: pip install matplotlib") from exc

import select

HOST = "192.168.0.165"
PORT = 3333

# ESP32 msg_hdr_t: magic(1), version(1), msg_type(1), flags(1), payload_len(4)
HDR_FMT = "!BBBBI"
HDR_SIZE = struct.calcsize(HDR_FMT)
MAGIC = 0xAA
VERSION = 1
MSG_TYPE_AUDIO = 1
MSG_TYPE_TEXT = 2

FLAG_SCREEN1 = 0x04
FLAG_SCREEN2 = 0x08

# Audio format based on ESP32 I2S config (24-bit, stereo, 16 kHz)
SAMPLE_RATE = 16000
CHANNELS = 2
CHANNEL_SELECT = "left"  # left, right, or mix
SAMPLE_BYTES = 3  # packed 24-bit samples (little-endian)
FRAME_BYTES = SAMPLE_BYTES * CHANNELS
VIEW_SECONDS = 2.0
MAX_BUFFER_SECONDS = 30.0
MAX_PAYLOAD = 4096
TEXT_MAX_PAYLOAD = 128  # Matches TEXT_BUF_SIZE in ESP32-LLL/main/app_tcp.h
TEXT_FLAGS = FLAG_SCREEN1

INPUT_LANGUAGE = "unknown"

# Decode packed 24-bit little-endian audio samples to float32 numpy array.
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

#takes flags from header and returns language string
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
    srv.setblocking(False)

    window_len = int(SAMPLE_RATE * VIEW_SECONDS)
    max_len = int(SAMPLE_RATE * MAX_BUFFER_SECONDS)
    audio_stream = np.empty((0,), dtype=np.float32)
    carry = b""
    buffer = bytearray()
    conn = None

    plt.ion()
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.18)
    line, = ax.plot(np.zeros(window_len, dtype=np.float32))
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0, window_len)
    ax.set_title("Waiting for audio...")
    status_text = fig.text(0.02, 0.01, "Disconnected")
    count_text = fig.text(0.72, 0.01, "Chars: 0")

    text_ax = fig.add_axes([0.08, 0.06, 0.58, 0.06])
    text_box = TextBox(text_ax, "Send:", initial="")
    btn_ax = fig.add_axes([0.70, 0.06, 0.12, 0.06])
    send_btn = Button(btn_ax, "Send")

    def update_count(text: str) -> None:
        count_text.set_text(f"Chars: {len(text)}")
        fig.canvas.draw_idle()

    def send_text_payload(text: str) -> None:
        nonlocal conn
        if not text:
            return
        if conn is None:
            print("No ESP32 connection; text not sent.")
            return
        data = text.encode("utf-8")
        offset = 0
        while offset < len(data):
            chunk = data[offset:offset + TEXT_MAX_PAYLOAD]
            header = struct.pack(HDR_FMT, MAGIC, VERSION, MSG_TYPE_TEXT,
                                 TEXT_FLAGS, len(chunk))
            try:
                conn.sendall(header + chunk)
            except OSError as exc:
                print(f"Failed to send text: {exc}")
                return
            offset += len(chunk)

    def submit_text(text: str | None = None) -> None:
        current = text if text is not None else text_box.text
        send_text_payload(current)
        text_box.set_val("")
        update_count("")

    text_box.on_text_change(update_count)
    text_box.on_submit(submit_text)
    send_btn.on_clicked(lambda _evt: submit_text())

    #select used here for a non-blocking event loop using select
    #waits 50 ms to see whether server socket or active connection socket ready.
    #args read_list, write_list, except_list, timeout
    #posix io multiplexing style
    print(f"Listening on {HOST}:{PORT} ...")
    while True:
        readable, _, _ = select.select([srv] + ([conn] if conn is not None else []), [], [], 0.01)
        if srv in readable:
            new_conn, addr = srv.accept()
            if conn is not None:
                conn.close()
            conn = new_conn
            conn_addr = addr
            conn.setblocking(False)
            buffer.clear()
            carry = b""
            status_text.set_text(f"Connected: {addr[0]}:{addr[1]}")
            fig.canvas.draw_idle()
            print(f"Connected by {addr}")

        if conn is not None and conn in readable:
            try:
                data = conn.recv(4096)
            except BlockingIOError:
                data = b""
            except OSError as exc:
                print(f"Socket error: {exc}")
                data = b""

            #checks for availability of data 
            #if data is empty, connection is closed
            #if data present, check practical bounds and process
            #srv active listening socket, conn active connection socket
            if not data:
                print("Connection closed")
                conn.close()
                conn = None
                conn_addr = None
                status_text.set_text("Disconnected")
                fig.canvas.draw_idle()
            else:
                buffer.extend(data)
                while True:
                    if len(buffer) < HDR_SIZE:
                        break
                    magic, version, msg_type, flags, payload_len = struct.unpack(
                        HDR_FMT, buffer[:HDR_SIZE]
                    )
                    if magic != MAGIC or version != VERSION:
                        print(f"Bad header: magic={magic} version={version}")
                        buffer.clear()
                        break
                    if payload_len > MAX_PAYLOAD:
                        if len(buffer) < HDR_SIZE + payload_len:
                            break
                        print(f"Payload too large ({payload_len}), discarding")
                        del buffer[:HDR_SIZE + payload_len]
                        continue
                    if len(buffer) < HDR_SIZE + payload_len:
                        break
                    
                    ## start processing packet to draw waveform

                    payload = bytes(buffer[HDR_SIZE:HDR_SIZE + payload_len])
                    del buffer[:HDR_SIZE + payload_len]

                    if msg_type != MSG_TYPE_AUDIO:
                        continue

                    new_lang = update_language(flags)
                    if new_lang != INPUT_LANGUAGE:
                        INPUT_LANGUAGE = new_lang

                    if FRAME_BYTES > 0:
                        payload = carry + payload
                        trim_len = len(payload) - (len(payload) % FRAME_BYTES)
                        carry = payload[trim_len:]
                        payload = payload[:trim_len]

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


if __name__ == "__main__":
    run_server()
