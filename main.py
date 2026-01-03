from __future__ import annotations

import argparse
import logging
import threading

from mt.opusmt_ct2 import OpusMTConfig
from pipeline.coordinator import PipelineConfig, Coordinator
from s2t.commit import CommitConfig
from s2t.whisper_engine import WhisperConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jetson-Live-Translator")
    parser.add_argument("--host", default="192.168.0.165")
    parser.add_argument("--port", type=int, default=3333)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--step-hz", type=float, default=1.0)
    parser.add_argument("--min-window-seconds", type=float, default=1.0)
    parser.add_argument("--max-buffer-seconds", type=float, default=30.0)
    parser.add_argument("--text-max-payload", type=int, default=128)
    parser.add_argument("--lang1-label", default="en")
    parser.add_argument("--lang2-label", default="fr")

    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--whisper-device", default="cuda")
    parser.add_argument("--whisper-compute-type", default="int8")
    parser.add_argument("--whisper-language", default=None)
    parser.add_argument("--whisper-no-speech-threshold", type=float, default=0.3)

    parser.add_argument("--opus-en-fr", default="/home/eric/models/opus/ct2/en-fr")
    parser.add_argument("--opus-fr-en", default="/home/eric/models/opus/ct2/fr-en")
    parser.add_argument("--opus-en-fr-tokenizer", default="Helsinki-NLP/opus-mt-en-fr")
    parser.add_argument("--opus-fr-en-tokenizer", default="Helsinki-NLP/opus-mt-fr-en")
    parser.add_argument("--tokenizer-local-only", action="store_true", default=True)
    parser.add_argument("--tokenizer-allow-network", action="store_true", default=False)
    parser.add_argument("--ct2-device", default="cuda")
    parser.add_argument("--ct2-compute-type", default="int8_float16")

    parser.add_argument("--commit-history", type=int, default=3)
    parser.add_argument("--commit-min-chars", type=int, default=1)

    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--plot-audio", action="store_true", default=False)
    parser.add_argument("--plot-window-seconds", type=float, default=10.0)
    parser.add_argument("--plot-hz", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logging.info("Starting Jetson-Live-Translator")

    whisper = WhisperConfig(
        model_size=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
        language=args.whisper_language,
        no_speech_threshold=args.whisper_no_speech_threshold,
    )
    opus = OpusMTConfig(
        en_fr_path=args.opus_en_fr,
        fr_en_path=args.opus_fr_en,
        en_fr_tokenizer=args.opus_en_fr_tokenizer,
        fr_en_tokenizer=args.opus_fr_en_tokenizer,
        tokenizer_local_only=args.tokenizer_local_only and not args.tokenizer_allow_network,
        lang1_label=args.lang1_label,
        lang2_label=args.lang2_label,
        device=args.ct2_device,
        compute_type=args.ct2_compute_type,
    )
    commit = CommitConfig(
        history_len=args.commit_history,
        min_commit_chars=args.commit_min_chars,
    )
    config = PipelineConfig(
        host=args.host,
        port=args.port,
        sample_rate=args.sample_rate,
        channels=args.channels,
        window_seconds=args.window_seconds,
        step_hz=args.step_hz,
        min_window_seconds=args.min_window_seconds,
        max_buffer_seconds=args.max_buffer_seconds,
        text_max_payload=args.text_max_payload,
        lang1_label=args.lang1_label,
        lang2_label=args.lang2_label,
        whisper=whisper,
        opus=opus,
        commit=commit,
    )

    try:
        if args.plot_audio:
            from utils.graphic_user_interface import AudioIntensityPlotter

            plotter = AudioIntensityPlotter(
                window_seconds=args.plot_window_seconds,
                target_hz=args.plot_hz,
            )
            config.plotter = plotter
            coordinator = Coordinator(config)
            thread = threading.Thread(target=coordinator.start, daemon=True)
            thread.start()
            plotter.run(on_close=coordinator.stop.set)
            coordinator.stop.set()
            thread.join(timeout=2.0)
        else:
            Coordinator(config).start()
    except Exception:
        logging.exception("Fatal error")
        raise


if __name__ == "__main__":
    main()
