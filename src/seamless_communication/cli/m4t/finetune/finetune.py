import argparse
import logging
import os
from pathlib import Path
import math

import torch
import torch.nn as nn

from seamless_communication.cli.m4t.finetune import dataloader, dist_utils, trainer
from seamless_communication.models.unity import (
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
)

logger = logging.getLogger("finetune")


class LoRALinear(nn.Module):
    """Enhanced manual LoRA implementation with better debugging"""

    def __init__(self, original_module, r=4, lora_alpha=1, lora_dropout=0.0):
        super().__init__()

        # Store original module but don't register it as a parameter
        self.original_module = original_module
        self.r = r
        self.lora_alpha = lora_alpha

        # Get dimensions more robustly
        if hasattr(original_module, 'input_dim') and hasattr(original_module, 'output_dim'):
            in_features = original_module.input_dim
            out_features = original_module.output_dim
        elif hasattr(original_module, 'in_features') and hasattr(original_module, 'out_features'):
            in_features = original_module.in_features
            out_features = original_module.out_features
        elif hasattr(original_module, 'weight') and len(original_module.weight.shape) == 2:
            out_features, in_features = original_module.weight.shape
        else:
            raise ValueError(f"Cannot determine dimensions for module: {type(original_module)}")

        logger.info(f"Creating LoRA adapter: {in_features} -> {out_features}, rank={r}")

        # CRITICAL: Freeze original parameters immediately
        for param in original_module.parameters():
            param.requires_grad = False
        original_module.eval()

        # LoRA matrices - these are the ONLY trainable parameters
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialize LoRA matrices properly
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = self.lora_alpha / self.r

        # Verify parameters are set correctly
        assert self.lora_A.requires_grad, "LoRA A matrix should be trainable"
        assert self.lora_B.requires_grad, "LoRA B matrix should be trainable"

    def forward(self, x):
        # Original forward pass (frozen)
        with torch.no_grad():
            result = self.original_module(x)

        # LoRA forward pass (trainable)
        lora_result = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T

        return result + lora_result * self.scaling


def discover_target_modules(model, sample_layer_ranges=None):
    """Automatically discover linear modules in the model for LoRA"""
    target_modules = []

    logger.info("=== DISCOVERING TARGET MODULES ===")

    # Get all named modules
    for name, module in model.named_modules():
        # Look for linear-like modules
        if (hasattr(module, 'weight') and
            len(module.weight.shape) == 2 and
            hasattr(module, 'forward')):

            # Filter for attention and FFN layers
            if any(keyword in name.lower() for keyword in [
                'self_attn', 'encoder_decoder_attn', 'cross_attn',
                'q_proj', 'k_proj', 'v_proj', 'out_proj', 'output_proj',
                'ffn', 'fc1', 'fc2', 'inner_proj'
            ]):
                logger.info(f"Found potential target: {name} - {module}")
                target_modules.append(name)

    logger.info(f"Discovered {len(target_modules)} potential target modules")
    return target_modules


def apply_lora_to_model(model, target_modules=None, r=4, lora_alpha=16, lora_dropout=0.1):
    """Apply LoRA to specified modules in the model"""

    # STEP 1: Freeze ALL parameters in the entire model first
    logger.info("Freezing all model parameters first...")
    for param in model.parameters():
        param.requires_grad = False

    # STEP 2: Auto-discover modules if not provided
    if target_modules is None:
        target_modules = discover_target_modules(model)

    # Conservative selection: last layers only for stability
    conservative_targets = []
    for name in target_modules:
        # Speech encoder: last 3 layers only
        if 'speech_encoder' in name and any(f'layers.{i}.' in name for i in [9, 10, 11]):
            conservative_targets.append(name)
        # Text decoder: last 4 layers for cross-attention
        elif 'text_decoder' in name and any(f'layers.{i}.' in name for i in [8, 9, 10, 11]):
            if 'encoder_decoder_attn' in name and any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj']):
                conservative_targets.append(name)

    logger.info(f"Selected {len(conservative_targets)} conservative target modules")
    lora_modules = {}

    # STEP 3: Apply LoRA to target modules
    for name in conservative_targets:
        try:
            module = model.get_submodule(name)

            # Verify it's a linear module
            if not (hasattr(module, 'weight') and len(module.weight.shape) == 2):
                logger.warning(f"Skipping {name}: not a linear module")
                continue

            logger.info(f"Applying LoRA to: {name}")

            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            if parent_name:
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module = model

            # Replace with LoRA module
            lora_module = LoRALinear(
                module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            setattr(parent_module, attr_name, lora_module)
            lora_modules[name] = lora_module

        except Exception as e:
            logger.error(f"Failed to apply LoRA to {name}: {e}")
            continue

    logger.info(f"Successfully applied LoRA to {len(lora_modules)} modules")
    return lora_modules


def debug_parameter_counts(model):
    """Enhanced debugging with detailed parameter analysis"""
    total_params = 0
    trainable_params = 0
    lora_params = 0
    frozen_params = 0

    logger.info("=== PARAMETER ANALYSIS ===")

    for name, param in model.named_parameters():
        total_params += param.numel()

        if param.requires_grad:
            if 'lora_A' in name or 'lora_B' in name:
                lora_params += param.numel()
                logger.info(f"✓ TRAINABLE (LoRA): {name} - {param.numel():,} params")
            else:
                logger.warning(f"⚠ TRAINABLE (Non-LoRA): {name} - {param.numel():,} params")
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()

    logger.info(f"=== SUMMARY ===")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"LoRA parameters: {lora_params:,}")
    logger.info(f"Frozen parameters: {frozen_params:,}")
    logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.4f}%")

    if trainable_params == 0:
        logger.error("❌ NO TRAINABLE PARAMETERS FOUND!")
        return False
    elif lora_params == 0:
        logger.warning("⚠ NO LORA PARAMETERS FOUND!")
        return False
    elif trainable_params > total_params * 0.05:  # More than 5%
        logger.warning(f"⚠ High trainable ratio: {100 * trainable_params / total_params:.2f}%")

    return True


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example finetuning script for M4T models with manual LoRA"
    )
    parser.add_argument(
        "--train_dataset",
        type=Path,
        required=True,
        help="Path to manifest with train samples",
    )
    parser.add_argument(
        "--eval_dataset",
        type=Path,
        required=True,
        help="Path to manifest with eval samples",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="seamlessM4T_medium",
        help="Base model name (`seamlessM4T_medium`, `seamlessM4T_large`)",
    )
    parser.add_argument(
        "--save_model_to",
        type=Path,
        required=True,
        help="Path to save best finetuned model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2343,
        help="Randomizer seed value",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help=(
            "Set early termination after `patience` number of evaluations "
            "without eval loss improvements"
        ),
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help=("Max number of training epochs"),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-7,
        help=("Finetuning learning rate"),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help=("Number of steps with linearly increasing learning rate"),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help=("Get eval loss after each `eval_steps` training steps "),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help=("Log inner loss after each `log_steps` training steps"),
    )
    parser.add_argument(
        "--max_src_tokens",
        type=int,
        default=7000,
        help=("Maximum number of src_tokens per batch, used to avoid GPU OOM and maximize the effective batch size"),
    )
    parser.add_argument(
        "--mode",
        type=trainer.FinetuneMode,
        choices=list(trainer.FinetuneMode),
        default=trainer.FinetuneMode.SPEECH_TO_TEXT,
        help=(
            "* `SPEECH_TO_SPEECH` -- finetune S2T and T2U parts of the model; "
            "* `TEXT_TO_SPEECH` -- finetune only T2U; "
            "* `SPEECH_TO_TEXT` -- finetune only S2T"
        ),
    )
    parser.add_argument(
        "--freeze_layers",
        nargs="*",
        required=False,
        default=None,
        help=("A list of modules to freeze in the model. If empty, everything will be trained."),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=("Device to fine-tune on. See `torch.device`."),
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before optimizer.step()",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha parameter",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout",
    )
    return parser


def main() -> None:
    args = init_parser().parse_args()

    dist_utils.init_distributed([logger, trainer.logger])
    float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16

    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)

    finetune_params = trainer.FinetuneParams(
        model_name=args.model_name,
        finetune_mode=args.mode,
        save_model_path=args.save_model_to,
        device=torch.device(args.device),
        float_dtype=float_dtype,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        patience=args.patience,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        grad_accum_steps=args.grad_accum_steps,
    )

    logger.info(f"Finetune Params: {finetune_params}")

    # Load model on CPU first to save memory
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float32)
    assert model.target_vocab_info == text_tokenizer.vocab_info

    # Remove unused components for SPEECH_TO_TEXT mode
    if (
        finetune_params.finetune_mode == trainer.FinetuneMode.SPEECH_TO_TEXT
        and model.t2u_model is not None
    ):
        model.t2u_model = None

    if model.text_encoder is not None:
        model.text_encoder = None

    # Apply manual LoRA with conservative settings
    logger.info("Applying manual LoRA implementation...")
    lora_modules = apply_lora_to_model(
        model,
        target_modules=None,  # Auto-discover
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # Verify LoRA was applied correctly
    success = debug_parameter_counts(model)
    if not success:
        raise RuntimeError("LoRA application failed! Check the logs above.")

    # Move model to device after LoRA application
    logger.info(f"Moving model to device: {finetune_params.device}")
    model = model.to(finetune_params.device)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")

    # Setup data loaders with better configuration
    train_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=finetune_params.train_batch_size,
            rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            max_audio_length_sec=20.0,  # Increased from 15.0
            float_dtype=finetune_params.float_dtype,
        ),
        dataset_manifest_path=args.train_dataset,
        max_src_tokens_per_batch=args.max_src_tokens
    )

    eval_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=finetune_params.eval_batch_size,
            rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            max_audio_length_sec=30.0,  # Reduced from 75.0 for stability
            float_dtype=finetune_params.float_dtype,
        ),
        dataset_manifest_path=args.eval_dataset
    )

    finetune = trainer.UnitYFinetune(
        model=model,
        params=finetune_params,
        train_data_loader=train_dataloader,
        eval_data_loader=eval_dataloader,
        freeze_modules=args.freeze_layers
    )

    finetune.run()


if __name__ == "__main__":
    main()
