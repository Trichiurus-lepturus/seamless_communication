#!/usr/bin/env python3
"""
Build multilingual FLEURS manifests for SeamlessM4T.

Generates manifests for all bilingual pairs among the languages
['eng','fra','spa','cmn','rus','arb'] (source -> target), skipping same->same.

By default the script creates one manifest file per pair:
  {split}_{src}2{tgt}_manifest.json

Use --merge to create a single merged manifest for the whole split:
  {split}_all_pairs_manifest.json

The produced manifest lines follow the same structure as the original
SeamlessM4T helper script, keeping `sample.target.waveform = None` (units
are extracted, waveform is cleared).

Requirements:
  - seamless_communication package available in PYTHONPATH
  - torch (cpu or cuda)

"""
import argparse
import dataclasses
import json
import logging
from pathlib import Path
import sys

import torch

from seamless_communication.datasets.huggingface import (
    Speech2SpeechFleursDatasetBuilder,
    SpeechTokenizer,
)
from seamless_communication.models.unit_extractor import UnitExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger("multilang-builder")

# Default languages (M4T codes) requested by the user
DEFAULT_LANGS = ["eng", "fra", "spa", "cmn", "rus", "arb"]

# Map M4T codes -> FLEURS dataset langcodes
UNITY_TO_FLEURS_LANG_MAPPING = {
    "eng": "en_us",
    "fra": "fr_fr",
    "spa": "es_419",
    # NOTE: user requested `cmn` (SeamlessM4T) for Chinese
    "cmn": "cmn_hans_cn",
    "rus": "ru_ru",
    "arb": "ar_eg",
}


def _check_lang_code_mapping(lang: str) -> None:
    if lang not in UNITY_TO_FLEURS_LANG_MAPPING:
        raise ValueError(
            f"No language code mapping for {lang}(M4T)->??(FLEURS). "
            "Please expand `UNITY_TO_FLEURS_LANG_MAPPING`"
        )


class UnitSpeechTokenizer(SpeechTokenizer):
    MODEL_NAME = "xlsr2_1b_v2"
    KMEANS_MODEL_URI = (
        "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy"
    )
    OUTPUT_LAYER_IDX = 34

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.unit_extractor = UnitExtractor(
            model_name_or_card=self.MODEL_NAME,
            kmeans_uri=self.KMEANS_MODEL_URI,
            device=self.device,
        )

    def encode(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return self.unit_extractor.predict(
            wav.to(self.device), out_layer_idx=self.OUTPUT_LAYER_IDX, sample_rate=sample_rate
        )


def download_fleurs_pair(source_lang: str, target_lang: str, split: str, save_directory: Path, device: torch.device):
    """Download and preprocess a single src→tgt pair from FLEURS and return the number
    of samples written (and an iterator if needed).
    """
    _check_lang_code_mapping(source_lang)
    _check_lang_code_mapping(target_lang)

    tokenizer = UnitSpeechTokenizer(device=device)

    dataset_iterator = Speech2SpeechFleursDatasetBuilder(
        source_lang=UNITY_TO_FLEURS_LANG_MAPPING[source_lang],
        target_lang=UNITY_TO_FLEURS_LANG_MAPPING[target_lang],
        dataset_cache_dir=str(save_directory),
        speech_tokenizer=tokenizer,
        skip_source_audio=True,
        skip_target_audio=False,
        split=split,
    )

    # Return a generator so caller can decide whether to write to pair manifest or merged file
    return dataset_iterator.__iter__()


def build_multilingual_dataset(split: str, save_dir: str, langs=None, merge=False, overwrite=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if langs is None:
        langs = DEFAULT_LANGS

    # Validate mapping exists for all languages
    for l in langs:
        _check_lang_code_mapping(l)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # If merge: open a single file for the whole split
    merged_fp = None
    if merge:
        merged_path = save_dir / f"{split}_all_pairs_manifest.json"
        if merged_path.exists() and not overwrite:
            logger.error(
                f"Merged manifest {merged_path} already exists. Use --overwrite to replace it."
            )
            sys.exit(1)
        merged_fp = open(merged_path, "w")
        logger.info(f"Writing merged manifest to: {merged_path}")

    try:
        for src in langs:
            for tgt in langs:
                if src == tgt:
                    continue

                logger.info(f"Preparing pair {src} -> {tgt} (split={split})")

                # get iterator that yields dataset samples (already runs unit extraction)
                try:
                    iterator = download_fleurs_pair(src, tgt, split, save_dir, device)
                except Exception as e:
                    logger.exception(f"Failed to create iterator for {src}->{tgt}: {e}")
                    continue

                pair_manifest_path = save_dir / f"{split}_{src}2{tgt}_manifest.json"
                if pair_manifest_path.exists() and not overwrite and not merge:
                    logger.error(
                        f"Pair manifest {pair_manifest_path} already exists. Use --overwrite to replace it. Skipping."
                    )
                    continue

                pair_fp = None
                if not merge:
                    pair_fp = open(pair_manifest_path, "w")
                    logger.info(f"Writing pair manifest to: {pair_manifest_path}")

                written = 0
                for idx, sample in enumerate(iterator, start=1):
                    # Normalize lang codes back to M4T form
                    sample.source.lang = src
                    sample.target.lang = tgt
                    # remove waveform because we've encoded target audio into units
                    sample.target.waveform = None

                    json_line = json.dumps(dataclasses.asdict(sample)) + "\n"

                    # write to pair file and/or merged file
                    if pair_fp is not None:
                        pair_fp.write(json_line)
                    if merged_fp is not None:
                        merged_fp.write(json_line)

                    written += 1

                if pair_fp is not None:
                    pair_fp.close()

                logger.info(f"Saved {written} samples for {src}→{tgt} (split={split})")
    finally:
        if merged_fp is not None:
            merged_fp.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build multilingual FLEURS manifests for SeamlessM4T. "
            "Default languages: eng, fra, spa, cmn, rus, arb."
        )
    )
    parser.add_argument("--split", type=str, required=True, help="train / validation / test")
    parser.add_argument("--save_dir", type=str, required=True, help="Where to save dataset + manifests")
    parser.add_argument(
        "--langs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional space-separated list of M4T langcodes to build (default: eng fra spa cmn rus arb)"
        ),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="If set, write a single merged manifest for the split instead of one manifest per pair.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing manifest files if they exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_multilingual_dataset(
        split=args.split, save_dir=args.save_dir, langs=args.langs, merge=args.merge, overwrite=args.overwrite
    )
