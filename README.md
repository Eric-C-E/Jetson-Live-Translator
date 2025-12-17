# Jetson-Live-Translator
Part of the LLL Edge-Inference Translator Headset project.

Manually toggled self-contained bidirectional translation pipeline.
Audio -> WhisperTRT -> Translator -> Text


# Functionality
[insert block diagram]

Default Inputs:

Default Outputs:


The program can take input in the form of an audio stream (for transcription or transcription-translation pipeline) or unix-domain character stream (for translation only pipeline).

Change the "settings" flags to easily change the functionality of the program and what it expects. The device outputs to a UNIX-domain socket which can be used for example, to send translated text over TCP to a receiver.

# How to run it?

## Setup (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

System packages you may need (varies by OS):
- `ffmpeg` (Whisper audio decode)
- PortAudio headers/libs (for `pyaudio`)

On Jetson, install NVIDIA-provided PyTorch + TensorRT (JetPack) first, then install `requirements.txt`.
