from __future__ import annotations

from dataclasses import dataclass
import logging

import ctranslate2
from transformers import AutoTokenizer


@dataclass
class OpusMTConfig:
    en_fr_path: str
    fr_en_path: str
    device: str = "cuda"
    compute_type: str = "float16"
    inter_threads: int = 1
    intra_threads: int = 0


class _CT2Model:
    def __init__(self, model_path: str, config: OpusMTConfig):
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

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
        self.en_fr = _CT2Model(config.en_fr_path, config)
        self.fr_en = _CT2Model(config.fr_en_path, config)

    def translate(self, text: str, src_lang: str) -> str:
        if src_lang == "lang1":
            return self.en_fr.translate(text)
        return self.fr_en.translate(text)
