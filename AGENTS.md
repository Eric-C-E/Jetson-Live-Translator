# Jetson-Live-Translator Agent Guide

## Project overview
This project runs a push-to-talk, bidirectional live translation pipeline on Jetson.
It listens for framed TCP audio packets, transcribes with Whisper, stabilizes text
with a simple commit algorithm, translates with Opus-MT (CT2), and sends framed
TCP text packets back to the headset.

## Entry point
- `main.py` builds configs (Whisper, Opus-MT, commit) and starts the coordinator.

## Pipeline flow
1. `net.tcp_client.TCPServer` accepts a single TCP client and reads frames.
2. `net.protocol.StreamParser` decodes frames with header format `!BBBBI`.
3. `audio.format.decode_packed_24bit_stereo_to_mono` converts 24-bit packed PCM
   to mono, selecting left/right based on language flags.
4. `pipeline.coordinator.PipelineWorker` buffers audio in a ring buffer and
   runs Whisper on sliding windows at `step_hz`.
5. `s2t.commit.SimpleCommitter` commits stable transcript prefixes.
6. `mt.opusmt_ct2.OpusMTTranslator` translates committed text.
7. `pipeline.coordinator.Coordinator` frames UTF-8 text into TCP packets and sends.

## Key modules
- `pipeline/coordinator.py`: Orchestrates networking, buffering, and worker thread.
- `s2t/whisper_engine.py`: Whisper model wrapper (faster-whisper).
- `mt/opusmt_ct2.py`: Opus-MT translation with CTranslate2 + Hugging Face tokenizer.
- `s2t/commit.py`: Stable-prefix commit algorithm.
- `audio/ringbuffer.py`: Float ring buffer for sliding windows.
- `net/protocol.py`: Packet framing and stream parsing.

## Protocol details
- Header format: `!BBBBI` (magic, version, msg_type, flags, payload_len)
- `MAGIC = 0xAA`, `VERSION = 1`
- `MSG_TYPE_AUDIO = 1`, `MSG_TYPE_TEXT = 2`
- Flags:
  - `0x01` language1 input, `0x02` language2 input
  - `0x04` language1 output, `0x08` language2 output
- Audio payload: packed 24-bit PCM, default 2-channel. Channel selection is
  tied to language (lang1 -> left, lang2 -> right).
- Text payload: UTF-8, chunked to `text_max_payload` bytes (default 128).

## Configuration (CLI)
See `main.py` for full list. Important flags:
- `--host`, `--port`: TCP server bind target.
- `--sample-rate`, `--channels`: PCM stream format.
- `--window-seconds`, `--step-hz`, `--min-window-seconds`, `--max-buffer-seconds`.
- `--lang1-label`, `--lang2-label`: language tags used throughout pipeline.
- Whisper: `--whisper-model`, `--whisper-device`, `--whisper-compute-type`,
  `--whisper-language`, `--whisper-no-speech-threshold`.
- Opus-MT: model paths + tokenizer ids, `--tokenizer-allow-network`.
- Commit: `--commit-history`, `--commit-min-chars`.

## Development notes
- Whisper requires a language set; if `--whisper-language` is empty, the
  pipeline must provide per-chunk language from flags.
- Opus-MT tokenizers default to offline mode. Use `--tokenizer-allow-network`
  once to download from Hugging Face.
- The worker thread processes transcription/translation independently of TCP IO.

## Setup & run
From `README.md`:
```bash
./scripts/setup_venv.sh
source .venv/bin/activate
python main.py
```

## Test utilities
- `tcp_test.py` contains TCP framing experiments (not part of the pipeline).
