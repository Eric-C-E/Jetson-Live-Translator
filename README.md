# Jetson-Live-Translator
Part of the LLL Edge-Inference Translator Headset project.

Manually push-to-talk self-contained bidirectional translation pipeline.
Audio from TCP -> openAI-Whisper -> latest win algo -> Opus-MT Translator -> Text to TCP


# Functionality
[insert block diagram]

Default Inputs: TCP Packets of PCM audio, header format BBBBI {magic = 0xAA, version = 1, type = 1 (audio) = 2 (text), flags = 1 (input lang1) = 2 (input lang2) = 4 (output lang1) = 8 (output lang2), payload_len = length of payload}
24 bit packed
Default size: payload len of 3072 max - this is what the ESP32 sends this.

Default Outputs: TCP Packets of text UTF-8 encoded, same header format BBBBI {magic, version, type, flags, payload_len}
Default size: payload of 128 max - this is what the ESP32 expects.

The program can take input in the form of an audio stream (for transcription or transcription-translation pipeline) or unix-domain character stream (for translation only pipeline).

Change the "settings" fields to easily change the program's languages.

As packets of audio come in (as controlled by the FSM on ESP32 side), headers are decoded, and audio is placed along with metadata (its language) into a buffer.

This buffer will have a simple sliding-window implementation to grab partial windows from it into Whisper. From Whisper, text should appear in transcript form. Whisper shall finish processing the whole utterance of one language before moving on to another language (metadata changed). All input into Whisper to this point would have had metadata tightly attached - this is intended such that all data in the pipeline is assigned a specific language, since this language needs to stay with the information from beginning to end, from rx to tx.

Between transcription and feeding into Opus-MT, we run a lightweight deconflicting committal algorithm which will only "commit" piece of sliding window transcription that are finalized, having been consistent enough for a past number of characters to "commit" as a truth (since information we send to the display should be "True" as there is no ghost-text functionality now).

After whisper, text with metadata is fed into Opus-MT with the argument set to always translate the current lang to the other lang. Upon exit from Opus-MT, text should be assigned metadata that describes the lang. At this point, if the original audio input was lang1, this text sequence would be lang2. When packing into TCP packets to be sent to the ESP32 headset side, header flag type should be set to text, and language of the text snippet should be set via flags = 0x04 or 0x08. 

The operation of Whisper and Opus-MT does not block TCP rx/tx.

# How to run it?

## Setup (venv)

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

You can pass extra pip args via `PIP_EXTRA_ARGS`, for example:

```bash
PIP_EXTRA_ARGS="--no-index --find-links $HOME/wheels" ./scripts/setup_venv.sh
```

## Offline tokenizer cache

By default the OpusMT tokenizer loads in offline mode. If you want to download
tokenizers from Hugging Face, run once with:

```bash
python main.py --tokenizer-allow-network
```

System packages you may need (varies by OS):
- `ffmpeg` (Whisper audio decode)
The program shouldn't need it but you'll need it to play with whisper and audio files.
- PortAudio headers/libs (for `pyaudio`)
- cTranslate2

On Jetson, install NVIDIA-provided PyTorch Vers. < 2 (JetPack) first, then install `requirements.txt`.

# Future Work

Optimize algorithms for speed and memory efficiency to be able to pack the system into ever smaller form factors. 

Create working ghost-text on receiver side for smooth live captions feel.
