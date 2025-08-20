import argparse
import logging
import os
import sys
from pathlib import Path
import math
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.distributed as dist

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
    """Enhanced manual LoRA implementation with freezing support"""

    def __init__(self, original_module, r=4, lora_alpha=1, lora_dropout=0.0):
        super().__init__()

        # Store original module but don't register it as a parameter
        self.original_module = original_module
        self.r = r
        self.lora_alpha = lora_alpha

        # CRITICAL FIX: Get dimensions more robustly
        in_features, out_features = self._get_module_dimensions(original_module)

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

        # Start with LoRA frozen (will be activated by training stage)
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False
        self.is_active = False

    def _get_module_dimensions(self, module):
        """CRITICAL FIX: Robust dimension detection for SeamlessM4T modules"""

        # Method 1: Standard linear layer attributes
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            return module.in_features, module.out_features

        # Method 2: Weight tensor inspection
        elif hasattr(module, 'weight') and len(module.weight.shape) == 2:
            out_features, in_features = module.weight.shape
            return in_features, out_features

        # Method 3: SeamlessM4T specific attributes
        elif hasattr(module, 'model_dim') and hasattr(module, 'num_heads'):
            # For attention modules
            return module.model_dim, module.model_dim

        # Method 4: Forward pass inspection (last resort)
        else:
            try:
                # Create dummy input to infer dimensions
                with torch.no_grad():
                    dummy_input = torch.randn(1, 512)  # Common hidden size
                    output = module(dummy_input)
                    return dummy_input.shape[-1], output.shape[-1]
            except Exception as e:
                raise ValueError(f"Cannot determine dimensions for {type(module)}: {e}")

    def activate(self):
        """Activate this LoRA adapter for training"""
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
        self.is_active = True

    def deactivate(self):
        """Deactivate this LoRA adapter (freeze it)"""
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False
        self.is_active = False

    def forward(self, x):
        # Original forward pass (always frozen)
        with torch.no_grad():
            result = self.original_module(x)

        # LoRA forward pass (only if active)
        if self.is_active:
            lora_result = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            return result + lora_result * self.scaling
        else:
            return result


def discover_target_modules(model, sample_layer_ranges=None):
    """Automatically discover linear modules in the model for LoRA"""
    target_modules = []

    for name, module in model.named_modules():
        if (hasattr(module, 'weight') and
            len(module.weight.shape) == 2 and
            hasattr(module, 'forward')):

            # Updated keywords for actual SeamlessM4T structure
            if any(keyword in name.lower() for keyword in [
                'self_attn', 'encoder_decoder_attn',  # attention modules
                'q_proj', 'k_proj', 'v_proj', 'output_proj',  # attention projections
                'ffn1', 'ffn2', 'ffn',  # FFN modules (speech has ffn1/ffn2, text has ffn)
                'inner_proj'  # FFN inner projections
            ]):
                target_modules.append(name)

    logger.info(f"Discovered {len(target_modules)} potential target modules")
    return target_modules


def apply_lora_to_model(model, target_modules=None, r=4, lora_alpha=16, lora_dropout=0.1):
    """Apply LoRA to all target modules (but keep them inactive initially)"""

    # STEP 1: Freeze ALL parameters in the entire model first
    logger.info("Freezing all model parameters first...")
    for param in model.parameters():
        param.requires_grad = False

    # STEP 2: Auto-discover modules if not provided
    if target_modules is None:
        target_modules = discover_target_modules(model)

    # STEP 3: Define all potential target modules for progressive training
    all_targets = []
    for name in target_modules:
        # Speech encoder with .inner. path and dual FFN
        if 'speech_encoder.inner.layers.' in name and any(f'layers.{i}.' in name for i in [8, 9, 10, 11]):
            if any(module_type in name for module_type in ['self_attn', 'ffn1', 'ffn2']):
                all_targets.append(name)

        # Text decoder - include both self_attn and encoder_decoder_attn
        elif 'text_decoder.layers.' in name and any(f'layers.{i}.' in name for i in [8, 9, 10, 11]):
            if any(module_type in name for module_type in ['self_attn', 'encoder_decoder_attn', 'ffn']):
                all_targets.append(name)

    logger.info(f"Applying LoRA to {len(all_targets)} modules (initially inactive)")
    lora_modules = {}

    # STEP 4: Apply LoRA to all target modules (but keep inactive)
    for name in all_targets:
        try:
            module = model.get_submodule(name)

            # Verify it's a linear module
            if not (hasattr(module, 'weight') and len(module.weight.shape) == 2):
                logger.warning(f"Skipping {name}: not a linear module")
                continue

            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            if parent_name:
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module = model

            # Replace with LoRA module (initially inactive)
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


def activate_lora_for_stage(model, stage: str):
    """FIXED: Activate specific LoRA modules based on training stage"""

    # First, deactivate all LoRA modules
    deactivated_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.deactivate()
            deactivated_count += 1

    logger.info(f"Deactivated {deactivated_count} LoRA modules")

    # Then activate modules for the current stage
    activated_count = 0

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module_path = name

            if stage == "speech_encoder":
                # FIXED: Only activate speech encoder LoRA modules
                if 'speech_encoder' in module_path and 'inner.layers.' in module_path:
                    # Focus on higher layers for better efficiency
                    if any(f'layers.{i}.' in module_path for i in [9, 10, 11]):
                        if any(comp in module_path for comp in ['self_attn', 'ffn1', 'ffn2']):
                            module.activate()
                            activated_count += 1
                            logger.debug(f"Activated speech encoder LoRA: {module_path}")

            elif stage == "text_decoder":
                # FIXED: Only activate text decoder LoRA modules
                if 'text_decoder' in module_path and 'layers.' in module_path:
                    # Focus on higher layers
                    if any(f'layers.{i}.' in module_path for i in [9, 10, 11]):
                        if any(comp in module_path for comp in ['self_attn', 'encoder_decoder_attn', 'ffn']):
                            module.activate()
                            activated_count += 1
                            logger.debug(f"Activated text decoder LoRA: {module_path}")

            elif stage == "full":
                # FIXED: Activate both speech encoder and text decoder (selective)
                activated_this_module = False

                # Speech encoder (selective activation)
                if ('speech_encoder' in module_path and 'inner.layers.' in module_path and
                        any(f'layers.{i}.' in module_path for i in [10, 11]) and
                        ('self_attn' in module_path or 'ffn2' in module_path)):
                    module.activate()
                    activated_this_module = True

                # Text decoder (selective activation)
                elif ('text_decoder' in module_path and 'layers.' in module_path and
                      any(f'layers.{i}.' in module_path for i in [10, 11]) and
                      'encoder_decoder_attn' in module_path):
                    module.activate()
                    activated_this_module = True

                if activated_this_module:
                    activated_count += 1
                    logger.debug(f"Activated full stage LoRA: {module_path}")

            elif stage == "conservative":
                # Original conservative approach (unchanged)
                if ('speech_encoder' in module_path and 'inner.layers.' in module_path and
                        any(f'layers.{i}.' in module_path for i in [9, 10, 11])):
                    module.activate()
                    activated_count += 1
                elif ('text_decoder' in module_path and 'layers.' in module_path and
                      'encoder_decoder_attn' in module_path and
                      any(f'layers.{i}.' in module_path for i in [8, 9, 10, 11]) and
                      any(proj in module_path for proj in ['q_proj', 'k_proj', 'v_proj'])):
                    module.activate()
                    activated_count += 1

    logger.info(f"Stage '{stage}': Activated {activated_count} LoRA modules")
    return activated_count


def debug_parameter_counts(model):
    """Enhanced debugging with detailed parameter analysis"""
    total_params = 0
    trainable_params = 0
    lora_params = 0
    frozen_params = 0
    active_lora_params = 0

    logger.info("=== PARAMETER ANALYSIS ===")

    for name, param in model.named_parameters():
        total_params += param.numel()

        if param.requires_grad:
            if 'lora_A' in name or 'lora_B' in name:
                lora_params += param.numel()
                active_lora_params += param.numel()
                logger.info(f"ACTIVE LoRA: {name} - {param.numel():,} params")
            else:
                logger.warning(f"TRAINABLE (Non-LoRA): {name} - {param.numel():,} params")
            trainable_params += param.numel()
        else:
            if 'lora_A' in name or 'lora_B' in name:
                lora_params += param.numel()
            frozen_params += param.numel()

    logger.info(f"=== SUMMARY ===")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total LoRA parameters: {lora_params:,}")
    logger.info(f"Active LoRA parameters: {active_lora_params:,}")
    logger.info(f"Frozen parameters: {frozen_params:,}")
    if total_params > 0:
        logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.4f}%")

    if trainable_params == 0:
        logger.error("NO TRAINABLE PARAMETERS FOUND!")
        return False
    elif active_lora_params == 0:
        logger.warning("NO ACTIVE LORA PARAMETERS FOUND!")
        return False

    return True


def load_previous_stage_lora(model, previous_stage_path: Path):
    """Load LoRA weights from previous training stage with error recovery"""
    if not previous_stage_path.exists():
        logger.info(f"Previous stage checkpoint not found: {previous_stage_path}")
        return False

    try:
        logger.info(f"Loading previous stage LoRA from: {previous_stage_path}")
        checkpoint = torch.load(previous_stage_path, map_location='cpu')

        # Verify checkpoint integrity
        if 'lora_adapters' not in checkpoint:
            logger.warning(f"Invalid checkpoint format: {previous_stage_path}")
            return False

        # Check if previous stage completed successfully
        if not checkpoint.get('completed', True):
            logger.warning(f"Previous stage did not complete successfully: {previous_stage_path}")
            # Still load the weights, but warn the user

        # Load LoRA weights into existing adapters
        lora_adapters = checkpoint['lora_adapters']
        loaded_count = 0

        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_a_key = f"{name}.lora_A"
                lora_b_key = f"{name}.lora_B"

                if lora_a_key in lora_adapters and lora_b_key in lora_adapters:
                    try:
                        module.lora_A.data.copy_(lora_adapters[lora_a_key])
                        module.lora_B.data.copy_(lora_adapters[lora_b_key])
                        loaded_count += 1
                        logger.info(f"Loaded LoRA weights for: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to load LoRA weights for {name}: {e}")

        logger.info(f"Successfully loaded {loaded_count} LoRA modules from previous stage")
        return loaded_count > 0

    except Exception as e:
        logger.error(f"Failed to load previous stage LoRA: {e}")
        return False


def verify_stage_prerequisites(stage: str, output_dir: Path, save_model_to: Path) -> bool:
    """Verify that prerequisites for a stage are met"""

    if stage == "text_decoder":
        # Check if speech_encoder stage completed
        speech_encoder_checkpoint = output_dir / f"{save_model_to.stem}_speech_encoder_lora.pt"
        if not speech_encoder_checkpoint.exists():
            logger.warning(f"Speech encoder stage checkpoint not found: {speech_encoder_checkpoint}")
            return False

        # Verify completion
        try:
            checkpoint = torch.load(speech_encoder_checkpoint, map_location='cpu')
            if not checkpoint.get('completed', True):
                logger.warning("Speech encoder stage did not complete successfully")
                return False
        except Exception as e:
            logger.error(f"Failed to verify speech encoder checkpoint: {e}")
            return False

    elif stage == "full":
        # Check if both previous stages completed
        for prev_stage in ["speech_encoder", "text_decoder"]:
            checkpoint_path = output_dir / f"{save_model_to.stem}_{prev_stage}_lora.pt"
            if not checkpoint_path.exists():
                logger.warning(f"Previous stage checkpoint not found: {checkpoint_path}")
                return False

            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if not checkpoint.get('completed', True):
                    logger.warning(f"Previous stage {prev_stage} did not complete successfully")
                    return False
            except Exception as e:
                logger.error(f"Failed to verify {prev_stage} checkpoint: {e}")
                return False

    return True


def get_optimal_num_workers() -> int:
    """Automatically determine optimal num_workers based on setup"""

    # Check if we're in distributed mode
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size == 1:
            return 0  # Single GPU - disable workers to avoid .throw() error
        else:
            return 2  # Multi-GPU - can use workers

    # Check total available GPUs
    elif torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count == 1:
            return 0  # Single GPU
        else:
            return min(2, gpu_count)  # Multi-GPU

    else:
        return 0  # CPU training - no workers needed


def create_dataloader_with_augmentation(
        text_tokenizer,
        unit_tokenizer,
        dataset_manifest_path: Path,
        batching_config: BatchingConfig,
        max_src_tokens_per_batch: int,
        training_stage: str = "none",
        is_training: bool = True
):
    """FIXED: Create dataloader - now properly handles stage-specific datasets"""

    # For training, try to use stage-specific dataset if available
    if is_training and training_stage != "conservative" and training_stage != "none":

        # Check for stage-specific dataset (from your new data preparation)
        manifest_dir = dataset_manifest_path.parent

        # Look for stage-specific dataset
        stage_dataset_path = manifest_dir / f"train_{training_stage}_dataset.json"

        if stage_dataset_path.exists():
            logger.info(f"Using stage-specific dataset: {stage_dataset_path}")
            dataset_manifest_path = stage_dataset_path
        else:
            logger.warning(f"Stage-specific dataset not found: {stage_dataset_path}")
            logger.info("Falling back to original manifest")

    # FIXED: Use simple dataloader with proper num_workers handling
    try:
        from seamless_communication.cli.m4t.finetune.dataloader_simple import create_simple_dataloader

        optimal_num_workers = get_optimal_num_workers()
        logger.info(f"Using num_workers={optimal_num_workers} for simple dataloader")

        return create_simple_dataloader(
            text_tokenizer=text_tokenizer,
            unit_tokenizer=unit_tokenizer,
            dataset_manifest_path=str(dataset_manifest_path),
            batching_config=batching_config,
            max_src_tokens_per_batch=max_src_tokens_per_batch,
            num_workers=optimal_num_workers
        )

    except ImportError:
        # Fallback to original dataloader
        logger.info("Simple dataloader not available, using original dataloader")

        # FIXED: Update the batching config num_workers for original dataloader too
        import copy
        updated_config = copy.deepcopy(batching_config)
        updated_config.num_workers = get_optimal_num_workers()

        return dataloader.UnitYDataLoader(
            text_tokenizer=text_tokenizer,
            unit_tokenizer=unit_tokenizer,
            batching_config=updated_config,
            dataset_manifest_path=str(dataset_manifest_path),
            max_src_tokens_per_batch=max_src_tokens_per_batch
        )


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhanced finetuning script for M4T models with progressive LoRA and augmentation"
    )

    # Original arguments (unchanged for compatibility)
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
        help="Base model name (seamlessM4T_medium, seamlessM4T_large)",
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
            "Set early termination after patience number of evaluations "
            "without eval loss improvements"
        ),
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Max number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-7,
        help="Finetuning learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of steps with linearly increasing learning rate",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Get eval loss after each eval_steps training steps",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="Log inner loss after each log_steps training steps",
    )
    parser.add_argument(
        "--max_src_tokens",
        type=int,
        default=7000,
        help="Maximum number of src_tokens per batch, used to avoid GPU OOM and maximize the effective batch size",
    )
    parser.add_argument(
        "--mode",
        type=trainer.FinetuneMode,
        choices=list(trainer.FinetuneMode),
        default=trainer.FinetuneMode.SPEECH_TO_TEXT,
        help=(
            "SPEECH_TO_SPEECH -- finetune S2T and T2U parts of the model; "
            "TEXT_TO_SPEECH -- finetune only T2U; "
            "SPEECH_TO_TEXT -- finetune only S2T"
        ),
    )
    parser.add_argument(
        "--freeze_layers",
        nargs="*",
        required=False,
        default=None,
        help="A list of modules to freeze in the model. If empty, everything will be trained.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to fine-tune on. See torch.device.",
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

    # NEW: Progressive training arguments
    parser.add_argument(
        "--training_stage",
        type=str,
        choices=["speech_encoder", "text_decoder", "full", "conservative"],
        default="conservative",
        help=(
            "Training stage for progressive fine-tuning: "
            "speech_encoder (LoRA on speech encoder + audio augmentation), "
            "text_decoder (LoRA on text decoder + text augmentation), "
            "full (LoRA on both + light augmentation), "
            "conservative (original behavior, default)"
        ),
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="Run full progressive training pipeline (all 3 stages)",
    )
    parser.add_argument(
        "--stage_epochs",
        type=int,
        nargs=3,
        default=[3, 3, 4],
        help="Number of epochs for each progressive stage [speech_encoder, text_decoder, full]",
    )

    # NEW: Error recovery arguments
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--skip_stage_verification",
        action="store_true",
        help="Skip verification of previous stage completion",
    )

    return parser


def run_single_stage(args, stage: str, epochs: int, previous_stage_path: Optional[Path] = None) -> bool:
    """Run a single training stage with error recovery"""

    logger.info(f"Starting training stage: {stage}")

    try:
        # Verify stage prerequisites
        if not args.skip_stage_verification:
            output_dir = args.save_model_to.parent
            if not verify_stage_prerequisites(stage, output_dir, args.save_model_to):
                logger.error(f"Prerequisites for stage {stage} not met")
                return False

        # Set up distributed training
        dist_utils.init_distributed([logger, trainer.logger])
        float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16

        # Load tokenizers
        text_tokenizer = load_unity_text_tokenizer(args.model_name)
        unit_tokenizer = load_unity_unit_tokenizer(args.model_name)

        # Determine finetune mode based on stage
        if stage == "speech_encoder":
            finetune_mode = trainer.FinetuneMode.SPEECH_TO_TEXT
        elif stage == "text_decoder":
            finetune_mode = trainer.FinetuneMode.SPEECH_TO_TEXT
        elif stage == "full":
            finetune_mode = trainer.FinetuneMode.SPEECH_TO_TEXT
        else:
            finetune_mode = args.mode  # Conservative/default

        # Create finetune params with error recovery support
        finetune_params = trainer.FinetuneParams(
            model_name=args.model_name,
            finetune_mode=finetune_mode,
            save_model_path=args.save_model_to,
            device=torch.device(args.device),
            float_dtype=float_dtype,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            patience=args.patience,
            max_epochs=epochs,  # Use stage-specific epochs
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            eval_steps=args.eval_steps,
            log_steps=args.log_steps,
            grad_accum_steps=args.grad_accum_steps,
            training_stage=stage,
            checkpoint_steps=args.checkpoint_steps,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

        logger.info(f"Finetune Params for {stage}: {finetune_params}")

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
            # Keep text encoder but freeze it completely
            for param in model.text_encoder.parameters():
                param.requires_grad = False
            model.text_encoder.eval()
            logger.info("Text encoder frozen")

        # Apply LoRA to all potential target modules (initially inactive)
        logger.info("Applying LoRA to all target modules...")
        lora_modules = apply_lora_to_model(
            model,
            target_modules=None,  # Auto-discover
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )

        if len(lora_modules) == 0:
            raise RuntimeError("No LoRA modules were applied to the model!")

        # Load previous stage weights if available
        if previous_stage_path and stage != "speech_encoder":
            load_previous_stage_lora(model, previous_stage_path)

        # Activate LoRA modules for the current stage BEFORE creating trainer
        logger.info(f"Activating LoRA modules for stage: {stage}")
        active_count = activate_lora_for_stage(model, stage)

        if active_count == 0:
            raise RuntimeError(f"No LoRA modules activated for stage {stage}!")

        # Verify LoRA was applied correctly
        success = debug_parameter_counts(model)
        if not success:
            raise RuntimeError(f"LoRA application failed for stage {stage}! Check the logs above.")

        # Move model to device after LoRA application
        logger.info(f"Moving model to device: {finetune_params.device}")
        model = model.to(finetune_params.device)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")

        # Setup data loaders with stage-appropriate augmentation
        train_dataloader = create_dataloader_with_augmentation(
            text_tokenizer=text_tokenizer,
            unit_tokenizer=unit_tokenizer,
            dataset_manifest_path=args.train_dataset,
            batching_config=dataloader.BatchingConfig(
                batch_size=finetune_params.train_batch_size,
                rank=dist_utils.get_rank(),
                world_size=dist_utils.get_world_size(),
                max_audio_length_sec=20.0,
                float_dtype=finetune_params.float_dtype,
            ),
            max_src_tokens_per_batch=args.max_src_tokens,
            training_stage=stage,
            is_training=True
        )

        eval_dataloader = create_dataloader_with_augmentation(
            text_tokenizer=text_tokenizer,
            unit_tokenizer=unit_tokenizer,
            dataset_manifest_path=args.eval_dataset,
            batching_config=dataloader.BatchingConfig(
                batch_size=finetune_params.eval_batch_size,
                rank=dist_utils.get_rank(),
                world_size=dist_utils.get_world_size(),
                max_audio_length_sec=30.0,
                float_dtype=finetune_params.float_dtype,
            ),
            max_src_tokens_per_batch=args.max_src_tokens,
            training_stage=stage,
            is_training=False  # Never augment validation data
        )

        # Create trainer AFTER LoRA activation, so optimizer gets correct parameters
        finetune = trainer.UnitYFinetune(
            model=model,
            params=finetune_params,
            train_data_loader=train_dataloader,
            eval_data_loader=eval_dataloader,
            freeze_modules=args.freeze_layers
        )

        # # Update optimizer after stage activation
        # if hasattr(finetune, 'update_optimizer_for_stage'):
        #     finetune.update_optimizer_for_stage(stage)

        # Run training with error recovery
        success = finetune.run()

        if success:
            logger.info(f"Successfully completed training stage: {stage}")
        else:
            logger.error(f"Training stage {stage} failed or was interrupted")

        return success

    except Exception as e:
        logger.error(f"Failed to run stage {stage}: {e}")
        return False


def main() -> None:
    args = init_parser().parse_args()

    try:
        if args.progressive:
            # Run full progressive training pipeline with error recovery
            logger.info("Running PROGRESSIVE training pipeline with error recovery")

            stages = ["speech_encoder", "text_decoder", "full"]
            stage_epochs = args.stage_epochs
            all_stages_successful = True

            for i, stage in enumerate(stages):
                epochs = stage_epochs[i] if i < len(stage_epochs) else args.max_epochs

                # CRITICAL FIX: Update save path for each stage with correct suffix
                stage_save_path = args.save_model_to.parent / f"{args.save_model_to.stem}_{stage}_lora.pt"
                args.save_model_to = stage_save_path

                # CRITICAL FIX: Determine previous stage path with correct suffix
                previous_stage_path = None
                if stage == "text_decoder":
                    previous_stage_path = args.save_model_to.parent / f"{args.save_model_to.stem.replace('_text_decoder', '')}_speech_encoder_lora.pt"
                elif stage == "full":
                    previous_stage_path = args.save_model_to.parent / f"{args.save_model_to.stem.replace('_full', '')}_text_decoder_lora.pt"

                # Run stage with error recovery
                success = run_single_stage(args, stage, epochs, previous_stage_path)

                if success:
                    logger.info(f"Stage {stage} completed successfully. Model saved to: {stage_save_path}")
                else:
                    logger.error(f"Stage {stage} failed!")
                    all_stages_successful = False
                    break

            if all_stages_successful:
                logger.info("Progressive training pipeline completed successfully!")
                sys.exit(0)
            else:
                logger.error("Progressive training pipeline failed!")
                sys.exit(1)

        else:
            # Run single stage with error recovery
            stage = args.training_stage
            epochs = args.max_epochs

            logger.info(f"Running SINGLE stage training with error recovery: {stage}")
            success = run_single_stage(args, stage, epochs)

            if success:
                logger.info(f"Single stage training completed successfully: {stage}")
                sys.exit(0)
            else:
                logger.error(f"Single stage training failed: {stage}")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"Training failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
