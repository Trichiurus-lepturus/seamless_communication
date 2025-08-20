#!/usr/bin/env python3
"""
Create stage-specific datasets from comprehensive augmentation pool.
Implements the 3:1 ratio with different variants per stage.
"""
import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_variant_for_stage(variants: List, stage: str) -> Optional[Dict]:
    """Robust variant selection with proper fallbacks"""
    if not variants:
        return None

    stage_mapping = {
        "speech_encoder": 0,  # Variant A
        "text_decoder": 1,    # Variant B
        "full": 2             # Variant C
    }

    preferred_idx = stage_mapping.get(stage, 0)

    # Select based on availability
    if len(variants) > preferred_idx:
        return variants[preferred_idx]
    elif variants:
        # Fallback to last available variant
        return variants[-1]
    else:
        return None

def create_stage_dataset(input_manifest_path: str,
                         output_manifest_path: str,
                         stage: str,
                         ratio_normal_to_augmented: float = 3.0,  # 3:1 ratio
                         seed: int = 42):
    """Create stage-specific dataset with 3:1 ratio and different variants"""

    # Load comprehensive pool
    with open(input_manifest_path, 'r') as f:
        all_samples = [json.loads(line) for line in f]

    logger.info(f"Loaded {len(all_samples)} samples from pool")

    # Group samples by original sample_idx
    sample_groups = {}
    for sample in all_samples:
        aug_info = sample.get('augmentation_info', {})
        sample_idx = aug_info.get('sample_idx', -1)

        if sample_idx not in sample_groups:
            sample_groups[sample_idx] = {
                'original': None,
                'variants': []
            }

        if aug_info.get('is_augmented', False):
            sample_groups[sample_idx]['variants'].append(sample)
        else:
            sample_groups[sample_idx]['original'] = sample

    logger.info(f"Grouped into {len(sample_groups)} sample groups")

    # Calculate how many samples to augment for 3:1 ratio
    total_groups = len(sample_groups)
    augmented_groups_count = int(total_groups / (ratio_normal_to_augmented + 1))
    normal_groups_count = total_groups - augmented_groups_count

    logger.info(f"Target: {normal_groups_count} normal + {augmented_groups_count} augmented = {total_groups} total")
    logger.info(f"Ratio: {normal_groups_count / augmented_groups_count:.1f}:1")

    # Create stage-specific selection with deterministic sampling
    stage_random = random.Random(seed)
    group_indices = list(sample_groups.keys())
    stage_random.shuffle(group_indices)

    # Select which groups to augment (first N groups after shuffle)
    augmented_group_indices = set(group_indices[:augmented_groups_count])

    selected_samples = []
    normal_count = 0
    augmented_count = 0

    for sample_idx, group in sample_groups.items():
        original = group['original']
        variants = group['variants']

        if original is None:
            logger.warning(f"No original found for sample {sample_idx}")
            continue

        # Always include original
        selected_samples.append(original)

        # Add augmented version if this group is selected for augmentation
        if sample_idx in augmented_group_indices and variants:
            # Select different variant based on stage
            selected_variant = select_variant_for_stage(variants, stage)

            if selected_variant:
                selected_samples.append(selected_variant)
                augmented_count += 1

        if sample_idx not in augmented_group_indices:
            normal_count += 1

    # Save stage dataset
    output_path = Path(output_manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in selected_samples:
            f.write(json.dumps(sample) + '\n')

    # Log final statistics
    actual_ratio = normal_count / augmented_count if augmented_count > 0 else float('inf')

    logger.info(f"Stage '{stage}' dataset created:")
    logger.info(f"  Normal samples: {normal_count}")
    logger.info(f"  Augmented samples: {augmented_count}")
    logger.info(f"  Total samples: {len(selected_samples)}")
    logger.info(f"  Actual ratio: {actual_ratio:.1f}:1")
    logger.info(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create stage-specific dataset with 3:1 ratio")
    parser.add_argument("--input_manifest", required=True, help="Input comprehensive manifest")
    parser.add_argument("--output_manifest", required=True, help="Output stage-specific manifest")
    parser.add_argument("--stage", required=True, choices=["speech_encoder", "text_decoder", "full"])
    parser.add_argument("--ratio", type=float, default=3.0, help="Normal:augmented ratio (default: 3.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    create_stage_dataset(
        input_manifest_path=args.input_manifest,
        output_manifest_path=args.output_manifest,
        stage=args.stage,
        ratio_normal_to_augmented=args.ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()