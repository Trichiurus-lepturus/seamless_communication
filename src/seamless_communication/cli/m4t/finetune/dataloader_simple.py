#!/usr/bin/env python3
"""
Simplified dataloader that uses original implementation without runtime augmentation.
This is used when augmentation is done during preprocessing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from seamless_communication.cli.m4t.finetune.dataloader import (
    UnitYDataLoader,
    BatchingConfig
)

logger = logging.getLogger(__name__)


def create_simple_dataloader(
    text_tokenizer,
    unit_tokenizer,
    dataset_manifest_path: str,
    batching_config: BatchingConfig,
    max_src_tokens_per_batch: int = 100000,
    num_workers: int = 0
) -> UnitYDataLoader:
    """
    Create simple dataloader without runtime augmentation.
    Assumes augmentation was done during preprocessing.
    """

    # Validate inputs
    if not Path(dataset_manifest_path).exists():
        raise FileNotFoundError(f"Dataset manifest not found: {dataset_manifest_path}")

    # FIXED: Properly update num_workers in batching config (it does exist!)
    # Create a copy to avoid modifying the original
    import copy
    updated_config = copy.deepcopy(batching_config)
    updated_config.num_workers = num_workers

    logger.info(f"Creating simple dataloader for: {dataset_manifest_path}")
    logger.info(f"Using num_workers: {num_workers}")

    return UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        dataset_manifest_path=dataset_manifest_path,
        batching_config=updated_config,
        max_src_tokens_per_batch=max_src_tokens_per_batch
    )