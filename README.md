# Jetson-Live-Translator
Part of the LLL Edge-Inference Translator Headset project.

Manually toggled self-contained bidirectional translation pipeline.
Audio -> WhisperTRT -> Translator -> Text


# Functionality
[insert block diagram]

Default Inputs:

Default Outputs:


The program can take input in the form of an audio stream (for transcription or transcription-translation pipeline) or unix-domain character stream (for translation only pipeline).

Change the "settings" flags to easily change the functionality of the program and what it expects. The device outputs to a UNIX-domain socket which can be used for example, to send translated text over UDP to a receiver.

# How to run it?
