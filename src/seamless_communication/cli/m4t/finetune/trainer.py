import logging
import time
import signal
import os
import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from fairseq2.data import VocabularyInfo
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn.padding import PaddingMask
from fairseq2.optim.lr_scheduler import MyleLR
from fairseq2.typing import Device
from torch.optim import AdamW

from seamless_communication.cli.m4t.finetune import dataloader, dist_utils
from seamless_communication.models.unity import (
    UnitYModel,
    UnitYT2UModel,
)

logger = logging.getLogger(__name__)


class FinetuneMode(Enum):
    SPEECH_TO_SPEECH = "SPEECH_TO_SPEECH"
    SPEECH_TO_TEXT = "SPEECH_TO_TEXT"
    TEXT_TO_SPEECH = "TEXT_TO_SPEECH"


@dataclass
class FinetuneParams:
    model_name: str
    """Model name of model being finetuned."""

    save_model_path: Path
    """Path were to save finetuned model."""

    finetune_mode: FinetuneMode = FinetuneMode.TEXT_TO_SPEECH
    """Allows to freeze S2T or T2U part of the model"""

    float_dtype: torch.dtype = torch.float16
    """Float Dtype"""

    max_epochs: int = 10
    """ Maximum number of training epochs"""

    label_smoothing: float = 0.2
    """ Label smoothing coefficient for nll_loss """

    warmup_steps: int = 100
    """ Number of steps with linearly increasing LR"""

    log_steps: int = 10
    """ Log inner loss after each `log_steps` training steps"""

    eval_steps: int = 50
    """ Get eval loss after each `eval_steps` training steps """

    patience: int = 3
    """ Terminate if eval loss did not improve
    over the last `patience * eval_steps` training steps"""

    learning_rate: float = 1e-5
    """ Optimizer learning rate """

    train_batch_size: int = 5
    """The batch size during train steps"""

    eval_batch_size: int = 5
    """The batch size during evaluation."""

    grad_accum_steps: int = 1
    """Number of steps to accumulate gradients before optimizer.step()"""

    device: Device = torch.device("cuda")
    """ Where to run computation"""

    # NEW: Progressive training support
    training_stage: Optional[str] = None
    """Current training stage for progressive training"""

    # NEW: Error recovery support
    checkpoint_steps: int = 100
    """Save checkpoint every N steps"""

    resume_from_checkpoint: Optional[str] = None
    """Path to checkpoint to resume from"""


class CheckpointManager:
    """Manages checkpoint saving/loading with error recovery"""

    def __init__(self, save_dir: Path, training_stage: Optional[str] = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.training_stage = training_stage

        # Checkpoint paths
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.latest_checkpoint_path = self.checkpoint_dir / "latest_checkpoint.json"
        self.emergency_checkpoint_path = self.checkpoint_dir / "emergency_checkpoint.pt"

        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")

    def get_checkpoint_path(self, step: int) -> Path:
        """Get path for a specific checkpoint"""
        stage_prefix = f"{self.training_stage}_" if self.training_stage else ""
        return self.checkpoint_dir / f"{stage_prefix}checkpoint_step_{step}.pt"

    def _get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Extract only LoRA adapter weights"""
        lora_state = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_state[f"{name}.lora_A"] = module.lora_A.cpu()
                lora_state[f"{name}.lora_B"] = module.lora_B.cpu()
        return lora_state

    def save_checkpoint(self) -> bool:
        """Save training checkpoint with LoRA-aware state"""
        try:
            if self.has_lora:
                # For LoRA training: Save only LoRA state + minimal training state
                checkpoint_state = {
                    'epoch': self.epoch_idx,
                    'update_idx': self.update_idx,
                    'best_eval_loss': self.best_eval_loss,
                    'training_stage': self.params.training_stage,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'lora_state_dict': self._get_lora_state_dict(),  # Only LoRA weights
                    'completed': False
                }
            else:
                # For full model training: Save complete state
                checkpoint_state = {
                    'epoch': self.epoch_idx,
                    'update_idx': self.update_idx,
                    'model_state_dict': self.model.state_dict(),  # Full model - only for non-LoRA
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'best_eval_loss': self.best_eval_loss,
                    'training_stage': self.params.training_stage,
                    'completed': False
                }

            # Add RNG states
            checkpoint_state['random_state'] = random.getstate()
            checkpoint_state['numpy_random_state'] = np.random.get_state()
            checkpoint_state['torch_random_state'] = torch.get_rng_state()
            if torch.cuda.is_available():
                checkpoint_state['cuda_random_state'] = torch.cuda.get_rng_state()

            success = self.checkpoint_manager.save_checkpoint(checkpoint_state, self.update_idx)
            if success:
                self.last_checkpoint_step = self.update_idx
            return success

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load checkpoint with integrity verification"""
        try:
            if checkpoint_path is None:
                checkpoint_path = self._get_latest_checkpoint_path()

            if checkpoint_path is None or not Path(checkpoint_path).exists():
                logger.info("No checkpoint found to resume from")
                return None

            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Verify checkpoint integrity
            required_keys = ['epoch', 'update_idx', 'model_state_dict']
            if not all(key in checkpoint for key in required_keys):
                logger.warning(f"Checkpoint missing required keys: {checkpoint_path}")
                return None

            logger.info(f"Successfully loaded checkpoint from step {checkpoint.get('checkpoint_step', 'unknown')}")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None

    def _get_latest_checkpoint_path(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        try:
            if not self.latest_checkpoint_path.exists():
                return None

            with open(self.latest_checkpoint_path, 'r') as f:
                latest_info = json.load(f)

            return latest_info.get('latest_checkpoint')

        except Exception as e:
            logger.warning(f"Failed to read latest checkpoint info: {e}")
            return None

    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints, keeping only the most recent"""
        try:
            stage_prefix = f"{self.training_stage}_" if self.training_stage else ""
            pattern = f"{stage_prefix}checkpoint_step_*.pt"

            checkpoints = list(self.checkpoint_dir.glob(pattern))
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for old_checkpoint in checkpoints[keep_last:]:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")

    def save_emergency_checkpoint(self, model, optimizer, lr_scheduler, step: int):
        """Save emergency checkpoint on interruption"""
        try:
            logger.info("Saving emergency checkpoint...")

            state_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'step': step,
                'emergency_save': True,
                'training_stage': self.training_stage,
                'timestamp': time.time()
            }

            torch.save(state_dict, self.emergency_checkpoint_path)
            logger.info(f"Emergency checkpoint saved: {self.emergency_checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")


class UnitYFinetuneWrapper(nn.Module):
    """Convenience wrapper that does a forward pass
    and returns S2T and T2U logits"""

    def __init__(self, model: UnitYModel, mode: FinetuneMode, device: Device):
        super().__init__()
        self.model: UnitYModel = model
        self.freeze_s2t: bool = mode == FinetuneMode.TEXT_TO_SPEECH
        self.freeze_t2u: bool = mode == FinetuneMode.SPEECH_TO_TEXT
        logger.info(f"Freeze s2t: {self.freeze_s2t}, freeze t2u: {self.freeze_t2u}")
        self.device = device

    def forward(
        self, batch: dataloader.MultimodalSeqsBatch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # dummy_context = contextmanager(lambda: iter([None]))()
        dummy_context = nullcontext()
        with torch.no_grad() if self.freeze_s2t else dummy_context:  # type:ignore
            assert batch.speech_to_text.src_tokens is not None
            seqs = batch.speech_to_text.src_tokens.to(self.device)
            assert batch.speech_to_text.src_lengths is not None
            seq_lens = batch.speech_to_text.src_lengths.to(self.device)
            speech_encoder_out, speech_encoder_padding_mask = self.model.encode_speech(
                seqs=seqs, padding_mask=PaddingMask(seq_lens, seqs.size(1))
            )
            assert batch.speech_to_text.prev_output_tokens is not None
            seqs = batch.speech_to_text.prev_output_tokens.to(self.device)
            assert batch.speech_to_text.target_lengths is not None
            seq_lens = batch.speech_to_text.target_lengths.to(self.device)
            text_decoder_out, text_decoder_padding_mask = self.model.decode(
                seqs=seqs,
                padding_mask=PaddingMask(seq_lens, seqs.size(1)),
                encoder_output=speech_encoder_out,
                encoder_padding_mask=speech_encoder_padding_mask,
            )
            assert self.model.final_proj is not None
            text_logits = self.model.final_proj(text_decoder_out)
        if self.freeze_t2u:
            return (text_logits, None)
        assert self.model.t2u_model is not None
        assert batch.text_to_units.prev_output_tokens is not None
        # dummy_context = contextmanager(lambda: iter([None]))()
        dummy_context = nullcontext()
        with torch.no_grad() if self.freeze_t2u else dummy_context:  # type:ignore
            if not isinstance(self.model.t2u_model, UnitYT2UModel):
                raise NotImplementedError(
                    "T2U finetuning implemented only for UnitYT2UModel"
                )
            (
                unit_encoder_out,
                unit_encoder_padding_mask,
            ) = self.model.t2u_model.encode(
                seqs=text_decoder_out,
                padding_mask=text_decoder_padding_mask,
            )
            seqs = batch.text_to_units.prev_output_tokens.to(self.device)
            assert batch.text_to_units.target_lengths is not None
            seq_lens = batch.text_to_units.target_lengths.to(self.device)
            unit_decoder_out, _ = self.model.t2u_model.decode(
                seqs=seqs,
                padding_mask=PaddingMask(seq_lens, seqs.size(1)),
                encoder_output=unit_encoder_out,
                encoder_padding_mask=unit_encoder_padding_mask,
            )
            unit_logits = self.model.t2u_model.final_proj(unit_decoder_out)

        return (text_logits, unit_logits)


class CalcLoss:
    """Fixed loss calculation for S2T and T2U with proper normalization"""

    def __init__(
        self,
        label_smoothing: float,
        s2t_vocab_info: VocabularyInfo,
        t2u_vocab_info: Optional[VocabularyInfo],
    ):
        self.label_smoothing = label_smoothing
        self.s2t_vocab_info = s2t_vocab_info
        self.t2u_vocab_info = t2u_vocab_info

        logger.info(f"Loss calculation initialized with label_smoothing={label_smoothing}")

    def __call__(
        self,
        batch: dataloader.MultimodalSeqsBatch,
        text_logits: torch.Tensor,
        unit_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:

        # === S2T Loss Calculation ===
        assert batch.speech_to_text.target_lengths is not None
        assert batch.speech_to_text.target_tokens is not None

        prefix_skip_len = 1  # Skip language token

        # Calculate S2T loss using fairseq2's built-in method
        s2t_sequence_output = SequenceModelOutput(
            logits=text_logits,
            vocab_info=self.s2t_vocab_info
        )

        # This returns the TOTAL loss (not per-token)
        s2t_total_loss = s2t_sequence_output.compute_loss(
            targets=batch.speech_to_text.target_tokens.to(text_logits.device),
            ignore_prefix_size=prefix_skip_len,
            label_smoothing=self.label_smoothing,
        )

        # Calculate the actual number of target tokens for proper normalization
        target_lengths = batch.speech_to_text.target_lengths.to(text_logits.device)
        s2t_num_tokens = torch.sum(torch.clamp(target_lengths - prefix_skip_len, min=1))

        # CORRECT: Normalize by actual token count
        s2t_loss_per_token = s2t_total_loss / s2t_num_tokens.float()

        # For SPEECH_TO_TEXT mode, only return S2T loss
        if unit_logits is None:
            return s2t_loss_per_token

        # === T2U Loss Calculation (if needed) ===
        assert batch.text_to_units.target_lengths is not None
        assert batch.text_to_units.target_tokens is not None
        assert self.t2u_vocab_info is not None

        s2u_sequence_output = SequenceModelOutput(
            logits=unit_logits,
            vocab_info=self.t2u_vocab_info
        )

        s2u_total_loss = s2u_sequence_output.compute_loss(
            targets=batch.text_to_units.target_tokens.to(unit_logits.device),
            ignore_prefix_size=prefix_skip_len,
            label_smoothing=self.label_smoothing,
        )

        unit_target_lengths = batch.text_to_units.target_lengths.to(unit_logits.device)
        s2u_num_tokens = torch.sum(torch.clamp(unit_target_lengths - prefix_skip_len, min=1))
        s2u_loss_per_token = s2u_total_loss / s2u_num_tokens.float()

        # Return combined loss (weighted equally)
        return s2t_loss_per_token + s2u_loss_per_token


class LossCollector:
    """Aggregates loss history across nodes"""

    def __init__(self, device: Optional[Device] = None, reduce_op: str = "avg"):
        self.n_samples: float = 0
        self.val_sum: float = 0.0
        self.reduce_op = reduce_op
        self.device = device
        self.is_distributed = dist_utils.is_dist_initialized()

    def reset(self) -> None:
        self.n_samples = 0
        self.val_sum = 0.0

    def update(self, n_samples: int, batch_loss: float) -> None:
        self.n_samples += n_samples
        self.val_sum += batch_loss

    def reduce(self) -> float:
        n_samples, val_sum = self._collect()
        if self.reduce_op == "avg":
            return val_sum / (n_samples + 1)
        if self.reduce_op == "sum":
            return val_sum
        raise ValueError()

    def _collect(self) -> Tuple[float, float]:
        if not self.is_distributed:
            return self.n_samples, self.val_sum
        local_val = torch.tensor([[self.n_samples, self.val_sum]], device=self.device)
        all_vals = [
            torch.zeros((1, 2), device=self.device)
            for _ in range(dist_utils.get_world_size())
        ]
        dist.all_gather(all_vals, local_val)
        losses = torch.concat(all_vals, dim=0)
        reduced = torch.sum(losses, dim=0).reshape(2).cpu()
        return reduced[0].item(), reduced[1].item()


class ProgressiveTrainingVerifier:
    """Verifies progressive training setup and tracks metrics"""

    def __init__(self, model, training_stage: Optional[str] = None):
        self.model = model
        self.training_stage = training_stage
        self.stage_metrics = {}

    def verify_lora_setup_for_stage(self) -> bool:
        """Verify that only expected LoRA modules are active for current stage"""
        if self.training_stage is None:
            return True

        active_speech_lora = 0
        active_text_lora = 0
        inactive_lora = 0

        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                module_path = name  # This is already the correct path

                if hasattr(module, 'is_active') and module.is_active:
                    if 'speech_encoder' in module_path:
                        active_speech_lora += 1
                    elif 'text_decoder' in module_path:
                        active_text_lora += 1
                else:
                    inactive_lora += 1

        # Verify stage-specific expectations
        if self.training_stage == "speech_encoder":
            expected = active_speech_lora > 0 and active_text_lora == 0
            logger.info(f"Stage {self.training_stage}: {active_speech_lora} speech LoRA active, "
                       f"{active_text_lora} text LoRA active, {inactive_lora} inactive")
        elif self.training_stage == "text_decoder":
            expected = active_speech_lora == 0 and active_text_lora > 0
            logger.info(f"Stage {self.training_stage}: {active_speech_lora} speech LoRA active, "
                       f"{active_text_lora} text LoRA active, {inactive_lora} inactive")
        elif self.training_stage == "full":
            expected = active_speech_lora > 0 and active_text_lora > 0
            logger.info(f"Stage {self.training_stage}: {active_speech_lora} speech LoRA active, "
                       f"{active_text_lora} text LoRA active, {inactive_lora} inactive")
        else:
            expected = True  # Conservative mode

        return expected

    def log_stage_progress(self, epoch: int, loss: float):
        """Log progress specific to training stage"""
        if self.training_stage:
            logger.info(f"Progressive Training - Stage: {self.training_stage}, "
                       f"Epoch: {epoch}, Loss: {loss:.4f}")


class UnitYFinetune:
    def __init__(
        self,
        model: UnitYModel,
        params: FinetuneParams,
        train_data_loader: dataloader.UnitYDataLoader,
        eval_data_loader: Optional[dataloader.UnitYDataLoader] = None,
        freeze_modules: Optional[List[Union[str, torch.nn.Module]]] = None
    ):
        self.params = params
        self.calc_loss = CalcLoss(
            label_smoothing=self.params.label_smoothing,
            s2t_vocab_info=model.target_vocab_info,
            t2u_vocab_info=model.t2u_model.target_vocab_info
            if model.t2u_model is not None
            else None,
        )

        self.model = self._wrap_model_for_training(model=model)

        # Check for manual LoRA implementation
        self.has_lora = self._detect_lora_modules()
        if self.has_lora:
            logger.info("Detected manual LoRA implementation")

        # Initialize progressive training verifier
        self.verifier = ProgressiveTrainingVerifier(
            self.model,
            self.params.training_stage
        )

        # Initialize checkpoint manager
        checkpoint_dir = self.params.save_model_path.parent / "training_checkpoints"
        self.checkpoint_manager = CheckpointManager(
            save_dir=checkpoint_dir,
            training_stage=self.params.training_stage
        )

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Verify LoRA setup for current stage
        if self.has_lora and self.params.training_stage:
            if not self.verifier.verify_lora_setup_for_stage():
                logger.warning(f"LoRA setup may not match expected stage: {self.params.training_stage}")

        # Only apply freeze_modules if we're NOT using LoRA (since LoRA handles freezing)
        if freeze_modules and not self.has_lora:
            logger.info("Applying additional module freezing...")
            self._freeze_modules(freeze_modules)
        elif freeze_modules and self.has_lora:
            logger.info("Skipping freeze_modules since LoRA already handles parameter freezing")

        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader

        self.grad_scaler = torch.cuda.amp.GradScaler()

        # CRITICAL: Only optimize parameters that require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"Optimizer will train {len(trainable_params)} parameter tensors")

        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found! Check LoRA implementation.")

        # Enhanced optimizer setup for progressive training
        optimizer_params = {
            "lr": self.params.learning_rate,
            "betas": (0.9, 0.98),
            "eps": 1e-08,
            "maximize": False,
            "weight_decay": 0.0,
        }

        # Add fused optimization for CUDA if available
        if self.params.device.type == "cuda":
            try:
                optimizer_params["fused"] = True
            except TypeError:
                logger.warning("Fused AdamW not available, using standard AdamW")

        self.optimizer = AdamW(params=trainable_params, **optimizer_params)

        self.lr_scheduler = MyleLR(
            optimizer=self.optimizer,
            num_warmup_steps=self.params.warmup_steps,
            start_lr=1e-9,
        )

        self.train_loss_hist = LossCollector(device=params.device)
        self.epoch_idx: int = 0
        self.update_idx: int = 0
        self.patience_left: int = self.params.patience
        self.best_eval_loss: Optional[float] = None
        self.is_best_state: bool = False

        # NEW: Error recovery state
        self.training_interrupted = False
        self.last_checkpoint_step = 0

        torch.set_float32_matmul_precision("high")

        # Log training setup
        self._log_training_setup()

    def update_training_stage(self, new_stage: str):
        """Update training stage and reconfigure optimizer"""
        logger.info(f"Updating training stage: {self.params.training_stage} -> {new_stage}")

        old_stage = self.params.training_stage
        self.params.training_stage = new_stage

        # Import and apply stage activation
        from .finetune import activate_lora_for_stage
        activated = activate_lora_for_stage(self.model, new_stage)

        if activated == 0:
            logger.error(f"222 No LoRA modules activated for stage '{new_stage}'!")
            raise RuntimeError(f"Stage '{new_stage}' activation failed!")

        # Verify everything is correct
        if not self._verify_optimizer_parameters():
            raise RuntimeError(f"Optimizer verification failed after stage update to '{new_stage}'!")

        # Update verifier
        self.verifier.training_stage = new_stage

        logger.info(f"111 Successfully updated to stage '{new_stage}' with {activated} active LoRA modules")


    def _verify_optimizer_parameters(self) -> bool:
        """Verify optimizer has all active LoRA parameters"""

        # Get all trainable LoRA parameters from model
        model_lora_params = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
                model_lora_params.add(id(param))

        # Get all parameters in optimizer
        optimizer_params = set()
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                optimizer_params.add(id(param))

        # Check for mismatches
        missing_in_optimizer = model_lora_params - optimizer_params
        extra_in_optimizer = optimizer_params - model_lora_params

        if missing_in_optimizer:
            logger.error(f"222 CRITICAL: {len(missing_in_optimizer)} active LoRA parameters missing from optimizer!")
            return False

        if extra_in_optimizer:
            logger.warning(f"222 Optimizer has {len(extra_in_optimizer)} extra parameters")

        logger.info(f"111 Optimizer parameter verification passed: {len(model_lora_params)} LoRA parameters")
        return True

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.training_interrupted = True

            # Save emergency checkpoint
            try:
                self.checkpoint_manager.save_emergency_checkpoint(
                    self.model, self.optimizer, self.lr_scheduler, self.update_idx
                )
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")

        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination

    def _log_training_setup(self):
        """Log detailed training setup information"""
        logger.info("=== TRAINING SETUP ===")
        logger.info(f"Training stage: {self.params.training_stage or 'standard'}")
        logger.info(f"Finetune mode: {self.params.finetune_mode}")
        logger.info(f"Max epochs: {self.params.max_epochs}")
        logger.info(f"Learning rate: {self.params.learning_rate}")
        logger.info(f"Batch size: {self.params.train_batch_size}")
        logger.info(f"Gradient accumulation steps: {self.params.grad_accum_steps}")
        logger.info(f"Checkpoint every: {self.params.checkpoint_steps} steps")
        if self.has_lora:
            logger.info("Using LoRA fine-tuning")
        if self.params.resume_from_checkpoint:
            logger.info(f"Will resume from: {self.params.resume_from_checkpoint}")
        logger.info("=====================")

    def _detect_lora_modules(self) -> bool:
        """Detect if the model has manual LoRA modules"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                return True
        return False

    def _reset_stats(self) -> None:
        self.train_loss_hist.reset()
        self.epoch_idx = 0
        self.update_idx = 0
        self.patience_left = self.params.patience
        self.best_eval_loss = None
        self.is_best_state = False

    def _wrap_model_for_training(self, model: UnitYModel) -> nn.Module:
        wrapped_model = UnitYFinetuneWrapper(
            model=model, mode=self.params.finetune_mode, device=self.params.device
        )
        if not dist_utils.is_dist_initialized():
            return wrapped_model
        find_unused = self.params.finetune_mode == FinetuneMode.TEXT_TO_SPEECH

        # For progressive training, we might need to handle unused parameters differently
        if self.params.training_stage in ["speech_encoder", "text_decoder"]:
            find_unused = True

        return nn.parallel.DistributedDataParallel(
            wrapped_model,
            device_ids=[dist_utils.get_local_rank()],
            find_unused_parameters=find_unused,
        )

    def _freeze_modules(self, frozen_modules: List[str] = []) -> None:
        """Freeze specified modules (only used for non-LoRA training)"""
        for icecube in frozen_modules:
            for (name, module) in self.model.named_modules():
                if name.startswith(icecube):
                    logger.info(f"Freezing Module: {name}")
                    for param in module.parameters():
                        param.requires_grad = False
                    try:
                        module.eval()
                    except Exception:
                        pass

    def _update_eval_stats(self, eval_loss: float) -> None:
        self.is_best_state = (
            self.best_eval_loss is None or eval_loss < self.best_eval_loss
        )
        self.best_eval_loss = eval_loss if self.is_best_state else self.best_eval_loss
        self.patience_left = (
            self.params.patience if self.is_best_state else self.patience_left - 1
        )
        logger.info(
            f"Eval after {self.update_idx} updates: "
            f"loss={eval_loss:.4f} "
            f"best_loss={self.best_eval_loss:.4f} "
            f"patience_steps_left={self.patience_left}"
        )

        # Log stage-specific progress
        self.verifier.log_stage_progress(self.epoch_idx, eval_loss)

    @torch.no_grad()
    def _eval_model(self, n_batches: int) -> None:
        """Calc avg loss on eval dataset and update evaluation stats"""
        if self.eval_data_loader is None:
            return
        logger.info(f"Evaluation Step {self.update_idx // self.params.eval_steps}...")
        loss_hist = LossCollector(device=self.params.device)
        self.model.eval()

        batch_count = 0
        try:
            for batch in self.eval_data_loader.get_dataloader():
                if n_batches == 0 or self.training_interrupted:
                    break
                assert batch.speech_to_text.src_tokens is not None
                with torch.autocast(device_type=self.params.device.type, dtype=self.params.float_dtype):
                    loss = self.calc_loss(batch, *self.model(batch))
                if loss.isnan():
                    logger.warning("Eval batch loss value is NaN, skipping")
                    continue
                loss_hist.update(1, loss.item())
                batch_count += 1
                n_batches -= 1

                # Clear memory after each eval batch
                del batch
                if batch_count % 10 == 0:
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM during evaluation, cleaning up and retrying with fewer batches")
                torch.cuda.empty_cache()
                return self._eval_model(min(n_batches, 50))  # Retry with fewer batches
            else:
                raise e

        eval_loss = loss_hist.reduce()
        self._update_eval_stats(eval_loss)

    def _train_step_log(self) -> None:
        """Log train stats with progressive training context"""
        if (self.update_idx + 1) % self.params.log_steps == 0:
            avg_loss = self.train_loss_hist.reduce()
            self.train_loss_hist.reset()

            stage_info = f" [{self.params.training_stage}]" if self.params.training_stage else ""
            logger.info(
                f"Epoch {str(self.epoch_idx + 1).zfill(3)} / "
                f"update {str(self.update_idx + 1).zfill(5)}{stage_info}: "
                f"train loss={avg_loss:.4f} "
                f"last lr={self.lr_scheduler.get_last_lr()[0]:.2E}"
            )

    def _train_step(self, batch: dataloader.MultimodalSeqsBatch) -> bool:
        """Enhanced train step with error recovery. Returns False if should stop training."""

        if self.training_interrupted:
            return False

        try:
            self.model.train()

            accum_steps = max(1, getattr(self.params, "grad_accum_steps", 1))

            if (self.update_idx % accum_steps) == 0:
                self.optimizer.zero_grad()

            with torch.autocast(device_type=self.params.device.type, dtype=self.params.float_dtype):
                tokens, units = self.model(batch)

            # Compute loss
            raw_loss = self.calc_loss(batch, tokens, units)

            # Debug loss calculation for first few steps
            if self.update_idx < 3:
                stage_info = f" (Stage: {self.params.training_stage})" if self.params.training_stage else ""
                logger.info(f"Step {self.update_idx} loss{stage_info}: {raw_loss.item():.6f}")

            # Check for problematic loss values
            if raw_loss.isnan().any().item():
                logger.error(f"Train loss is NaN! Raw loss: {raw_loss}")
                raise RuntimeError("Train loss is NaN!")

            if raw_loss.item() > 15.0:
                logger.warning(f"Very high loss detected: {raw_loss.item():.4f}")

            # Scale loss for gradient accumulation
            scaled_loss = raw_loss / accum_steps
            self.grad_scaler.scale(scaled_loss).backward()

            # Gradient clipping for stability
            if ((self.update_idx + 1) % accum_steps) == 0:
                # Unscale gradients before clipping
                self.grad_scaler.unscale_(self.optimizer)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0
                )

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.lr_scheduler.step()

            self.train_loss_hist.update(1, raw_loss.item())
            self._train_step_log()
            self.update_idx += 1

            return True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM during training step, cleaning up and skipping batch")
                torch.cuda.empty_cache()
                # Reset gradients and continue
                self.optimizer.zero_grad()
                return True
            else:
                logger.error(f"Training step failed: {e}")
                return False

    def save_checkpoint(self) -> bool:
        """Save training checkpoint"""
        try:
            # Prepare checkpoint state
            checkpoint_state = {
                'epoch': self.epoch_idx,
                'update_idx': self.update_idx,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'grad_scaler_state_dict': self.grad_scaler.state_dict(),
                'best_eval_loss': self.best_eval_loss,
                'patience_left': self.patience_left,
                'params': self.params,
                'training_stage': self.params.training_stage,
                'random_state': torch.get_rng_state(),
            }

            # Add CUDA random state if available
            if torch.cuda.is_available():
                checkpoint_state['cuda_random_state'] = torch.cuda.get_rng_state()

            success = self.checkpoint_manager.save_checkpoint(checkpoint_state, self.update_idx)
            if success:
                self.last_checkpoint_step = self.update_idx
            return success

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """FIXED: Load training checkpoint with stage reactivation"""
        try:
            checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            if checkpoint is None:
                return False

            # Restore training state
            self.epoch_idx = checkpoint['epoch']
            self.update_idx = checkpoint['update_idx']
            self.best_eval_loss = checkpoint.get('best_eval_loss')
            self.patience_left = checkpoint.get('patience_left', self.params.patience)

            # Restore model state first
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # CRITICAL FIX: Reactivate LoRA for current stage BEFORE updating optimizer
            if self.has_lora and self.params.training_stage:
                logger.info(f"Reactivating LoRA modules for stage: {self.params.training_stage}")

                # Import here to avoid circular imports
                from .finetune import activate_lora_for_stage
                activated = activate_lora_for_stage(self.model, self.params.training_stage)
                logger.info(f"Reactivated {activated} LoRA modules for stage: {self.params.training_stage}")

            # THEN restore optimizer and scheduler state
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to restore optimizer/scheduler state: {e}")
                logger.warning("Continuing with fresh optimizer state")

            # Restore grad scaler state
            if 'grad_scaler_state_dict' in checkpoint:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

            # Restore random states
            if 'random_state' in checkpoint:
                torch.set_rng_state(checkpoint['random_state'])
            if 'cuda_random_state' in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint['cuda_random_state'])

            logger.info(f"111 Resumed training from epoch {self.epoch_idx}, step {self.update_idx}")

            # CRITICAL: Verify optimizer has correct parameters
            self._verify_optimizer_parameters()

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _save_lora_adapters(self) -> None:
        """Save LoRA adapter weights with stage information"""
        lora_state_dict = {}
        lora_config = {}
        active_modules = []

        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Save LoRA parameters
                lora_state_dict[f"{name}.lora_A"] = module.lora_A.cpu()
                lora_state_dict[f"{name}.lora_B"] = module.lora_B.cpu()

                # Track active modules
                if hasattr(module, 'is_active') and module.is_active:
                    module_path = name.replace('.original_module', '')
                    active_modules.append(module_path)

                # Save configuration (just once)
                if not lora_config:
                    lora_config = {
                        "r": module.r,
                        "lora_alpha": module.lora_alpha,
                        "scaling": module.scaling,
                        "training_stage": self.params.training_stage,
                        "active_modules": active_modules
                    }

        # Update active modules in config
        lora_config["active_modules"] = active_modules

        # Create stage-specific save path
        save_path = self.params.save_model_path
        if self.params.training_stage:
            # Check if stage suffix already exists to avoid duplication
            stage_suffix = f'_{self.params.training_stage}_lora.pt'
            if not str(save_path).endswith(stage_suffix):
                save_path = str(save_path).replace('.pt', stage_suffix)
        else:
            save_path = str(save_path).replace('.pt', '_lora_adapters.pt')

        torch.save({
            "model_name": self.params.model_name,
            "lora_adapters": lora_state_dict,
            "lora_config": lora_config,
            "training_stage": self.params.training_stage,
            "best_eval_loss": self.best_eval_loss,
            "epoch": self.epoch_idx,
            "completed": not self.training_interrupted
        }, save_path)

        logger.info(f"LoRA adapters saved to {save_path}")
        logger.info(f"Active LoRA modules: {len(active_modules)}")

    def _save_model(self) -> None:
        logger.info("Saving model")
        if dist_utils.is_main_process():
            if self.has_lora:
                # Save only LoRA adapters for manual LoRA
                self._save_lora_adapters()
            else:
                # Full model saving for non-LoRA training
                torch.save({
                    "model_name": self.params.model_name,
                    "model": {
                        key.replace("module.model.model.", ""): value
                        for key, value in self.model.state_dict().items()
                    },
                    "training_stage": self.params.training_stage,
                    "best_eval_loss": self.best_eval_loss,
                    "epoch": self.epoch_idx,
                    "completed": not self.training_interrupted
                }, self.params.save_model_path)
        if dist_utils.is_dist_initialized():
            dist.barrier()

    def run(self) -> bool:
        """Enhanced training loop with error recovery. Returns True if completed successfully."""
        stage_info = f" (Stage: {self.params.training_stage})" if self.params.training_stage else ""
        logger.info(f"Start Finetuning{stage_info}")

        # Attempt to resume from checkpoint
        if self.params.resume_from_checkpoint:
            if not self.load_checkpoint(self.params.resume_from_checkpoint):
                logger.warning("Failed to resume from checkpoint, starting fresh")
                self._reset_stats()
        else:
            # Check for automatic resume
            if not self.load_checkpoint():
                logger.info("No checkpoint found, starting fresh training")
                self._reset_stats()

        # Initial evaluation
        try:
            self._eval_model(n_batches=100)
        except Exception as e:
            logger.warning(f"Initial evaluation failed: {e}")

        train_dataloader = self.train_data_loader.get_dataloader()
        training_successful = False

        try:
            while self.epoch_idx < self.params.max_epochs and self.patience_left and not self.training_interrupted:
                logger.info(f"Starting epoch {self.epoch_idx + 1}/{self.params.max_epochs}{stage_info}")

                for train_batch in tqdm(train_dataloader, desc=f"Training Steps{stage_info}"):
                    if self.training_interrupted:
                        logger.info("Training interrupted, breaking from training loop")
                        break

                    # Execute training step with error recovery
                    if not self._train_step(train_batch):
                        logger.error("Training step failed, stopping training")
                        break

                    # Periodic checkpoint saving
                    if self.update_idx % self.params.checkpoint_steps == 0:
                        self.save_checkpoint()

                    if not self.update_idx or self.update_idx % self.params.eval_steps != 0:
                        continue

                    # Enhanced memory management for progressive training
                    torch.cuda.empty_cache()

                    try:
                        self._eval_model(n_batches=100)
                    except Exception as e:
                        logger.warning(f"Evaluation failed: {e}")
                        continue

                    if self.is_best_state:
                        self._save_model()
                    elif not self.patience_left:
                        no_improve_steps = self.params.eval_steps * self.params.patience
                        logger.info(
                            f"Early termination{stage_info}, as eval loss did not improve "
                            f"over last {no_improve_steps} updates"
                        )
                        training_successful = True
                        break

                if self.training_interrupted:
                    break

                # Handle leftover gradients at end of epoch
                accum_steps = max(1, getattr(self.params, "grad_accum_steps", 1))
                if (self.update_idx % accum_steps) != 0:
                    try:
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                        self.lr_scheduler.step()
                    finally:
                        self.optimizer.zero_grad()

                self.epoch_idx += 1

                # Memory cleanup between epochs
                torch.cuda.empty_cache()

            # Check if training completed successfully
            if not self.training_interrupted and (self.epoch_idx >= self.params.max_epochs or not self.patience_left):
                training_successful = True

        except Exception as e:
            logger.error(f"Training failed with exception: {e}")
            training_successful = False

        finally:
            # Always save final checkpoint and model
            try:
                if not self.training_interrupted:
                    self.save_checkpoint()
                    self._save_model()
                    logger.info(f"Training completed{stage_info}")
                else:
                    logger.info(f"Training interrupted{stage_info}, but state saved")

                # Final verification for progressive training
                if self.has_lora and self.params.training_stage:
                    self.verifier.verify_lora_setup_for_stage()

                # Enhanced cleanup at end of training
                torch.cuda.empty_cache()

                # Clear optimizer state to free memory
                if hasattr(self, 'optimizer'):
                    del self.optimizer
                    del self.lr_scheduler

                # Force garbage collection
                import gc
                gc.collect()

            except Exception as e:
                logger.error(f"Error during training cleanup: {e}")

        return training_successful and not self.training_interrupted
