#!/usr/bin/env python3

import argparse
import logging
import torch
from pathlib import Path

from seamless_communication.models.unity import load_unity_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("merge_lora")


def load_lora_checkpoint(checkpoint_path):
    """Load LoRA checkpoint and extract adapters"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'lora_adapters' in checkpoint:
        # LoRA adapter format
        return checkpoint['lora_adapters'], checkpoint.get('lora_config', {}), checkpoint['model_name']
    else:
        # Full model format - return None to indicate no LoRA
        return None, {}, checkpoint.get('model_name', 'seamlessM4T_medium')


def merge_lora_weights(base_weight, lora_A, lora_B, scaling):
    """Merge LoRA matrices into base weight matrix"""
    # LoRA contribution: scaling * (B @ A)
    lora_weight = scaling * (lora_B @ lora_A)
    return base_weight + lora_weight


def apply_lora_merge(model, lora_adapters, lora_config):
    """Merge LoRA adapters into the base model weights"""

    scaling = lora_config.get('scaling', lora_config.get('lora_alpha', 16) / lora_config.get('r', 8))

    logger.info(f"Merging LoRA with scaling factor: {scaling}")

    merged_count = 0

    # Group LoRA pairs (A and B matrices)
    lora_pairs = {}
    for name, weight in lora_adapters.items():
        if '.lora_A' in name:
            base_name = name.replace('.lora_A', '')
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            lora_pairs[base_name]['A'] = weight
        elif '.lora_B' in name:
            base_name = name.replace('.lora_B', '')
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            lora_pairs[base_name]['B'] = weight

    # Merge each LoRA pair into the base model
    for module_name, lora_matrices in lora_pairs.items():
        if 'A' not in lora_matrices or 'B' not in lora_matrices:
            logger.warning(f"Incomplete LoRA pair for {module_name}, skipping")
            continue

        try:
            # Get the base module
            base_module = model.get_submodule(module_name)

            if not hasattr(base_module, 'weight'):
                logger.warning(f"Module {module_name} has no weight parameter, skipping")
                continue

            # Get original weight
            original_weight = base_module.weight.data

            # Get LoRA matrices
            lora_A = lora_matrices['A']
            lora_B = lora_matrices['B']

            logger.info(f"Merging LoRA for {module_name}: {original_weight.shape} + LoRA({lora_B.shape} @ {lora_A.shape})")

            # Merge LoRA into original weight
            merged_weight = merge_lora_weights(original_weight, lora_A, lora_B, scaling)

            # Update the module weight
            base_module.weight.data = merged_weight
            merged_count += 1

        except Exception as e:
            logger.error(f"Failed to merge LoRA for {module_name}: {e}")
            continue

    logger.info(f"Successfully merged {merged_count} LoRA adapters")
    return model


def save_merged_model(model, model_name, output_path):
    """Save the merged model in a clean format"""

    # Create a clean state dict
    clean_state_dict = {}
    for name, param in model.named_parameters():
        clean_state_dict[name] = param.data.cpu()

    # Save in a format compatible with the original loading function
    torch.save({
        'model_name': model_name,
        'model': clean_state_dict,
        'merged_from_lora': True,
    }, output_path)

    logger.info(f"Merged model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--lora_checkpoint", type=Path, required=True,
                       help="Path to LoRA checkpoint file")
    parser.add_argument("--output_model", type=Path, required=True,
                       help="Path to save merged model")
    parser.add_argument("--model_name", type=str, default="seamlessM4T_medium",
                       help="Base model name")

    args = parser.parse_args()

    logger.info(f"Loading LoRA checkpoint: {args.lora_checkpoint}")
    lora_adapters, lora_config, checkpoint_model_name = load_lora_checkpoint(args.lora_checkpoint)

    if lora_adapters is None:
        logger.error("No LoRA adapters found in checkpoint!")
        return

    # Use model name from checkpoint if available
    model_name = checkpoint_model_name or args.model_name

    logger.info(f"Loading base model: {model_name}")
    base_model = load_unity_model(model_name, device=torch.device("cpu"), dtype=torch.float32)

    # Remove unused components (same as training)
    if base_model.t2u_model is not None:
        base_model.t2u_model = None
    if base_model.text_encoder is not None:
        base_model.text_encoder = None

    logger.info("Merging LoRA adapters into base model...")
    merged_model = apply_lora_merge(base_model, lora_adapters, lora_config)

    logger.info("Saving merged model...")
    save_merged_model(merged_model, model_name, args.output_model)

    logger.info("111 Merge completed successfully!")

    # Basic verification
    total_params = sum(p.numel() for p in merged_model.parameters())
    logger.info(f"Merged model has {total_params:,} parameters")


if __name__ == "__main__":
    main()
