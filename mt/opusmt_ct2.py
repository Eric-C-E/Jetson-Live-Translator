from __future__ import annotations

from dataclasses import dataclass
import logging

import ctranslate2
from transformers import AutoTokenizer


@dataclass
class OpusMTConfig:
    en_fr_path: str
    fr_en_path: str
    en_fr_tokenizer: str | None = None
    fr_en_tokenizer: str | None = None
    tokenizer_local_only: bool = True
    lang1_label: str = "en"
    lang2_label: str = "fr"
    device: str = "cuda"
    compute_type: str = "float16"
    inter_threads: int = 1
    intra_threads: int = 0


class _CT2Model:
    def __init__(self, model_path: str, tokenizer_path: str, config: OpusMTConfig):
        logging.info(
            "Loading OpusMT model at %s device=%s compute_type=%s",
            model_path,
            config.device,
            config.compute_type,
        )
        self.translator = ctranslate2.Translator(
            model_path,
            device=config.device,
            compute_type=config.compute_type,
            inter_threads=config.inter_threads,
            intra_threads=config.intra_threads,
        )
        if config.tokenizer_local_only:
            logging.info("Loading OpusMT tokenizer (offline) from %s", tokenizer_path)
        else:
            logging.info("Loading OpusMT tokenizer from %s", tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=False,
            local_files_only=config.tokenizer_local_only,
        )

    def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        token_ids = self.tokenizer.encode(text)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        result = self.translator.translate_batch([tokens])[0]
        output_tokens = result.hypotheses[0]
        output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)


class OpusMTTranslator:
    def __init__(self, config: OpusMTConfig):
        self.config = config
        en_fr_tokenizer = config.en_fr_tokenizer or config.en_fr_path
        fr_en_tokenizer = config.fr_en_tokenizer or config.fr_en_path
        self.en_fr = _CT2Model(config.en_fr_path, en_fr_tokenizer, config)
        self.fr_en = _CT2Model(config.fr_en_path, fr_en_tokenizer, config)

    def translate(self, text: str, src_lang: str) -> str:
        if src_lang == self.config.lang1_label:
            return self.en_fr.translate(text)
        if src_lang == self.config.lang2_label:
            return self.fr_en.translate(text)
        logging.warning("Unknown source language %s; defaulting to %s", src_lang, self.config.lang1_label)
        return self.en_fr.translate(text)
