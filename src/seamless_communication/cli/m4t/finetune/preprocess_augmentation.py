#!/usr/bin/env python3
"""
Create comprehensive audio augmentation pool with multiple variants per sample.
"""
import argparse
import json
import logging
import random
import os
from pathlib import Path
from typing import Dict, List, Optional
import multiprocessing as mp
from functools import partial
import copy

import torch
import torchaudio
import torchaudio.functional as F

from tqdm import tqdm

from seamless_communication.datasets.datatypes import LangPairSample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiverseAudioAugmentation:
    """Diverse audio augmentation with multiple random techniques per sample"""

    def __init__(self,
                 augmentation_prob: float = 0.33,
                 techniques_per_sample: tuple = (1, 3),
                 preserve_length: bool = True):
        """
        Args:
            augmentation_prob: Probability that a sample gets augmented (0.33 = 33%)
            techniques_per_sample: Range of techniques to apply per augmented sample
            preserve_length: Whether to preserve original audio length
        """
        self.augmentation_prob = augmentation_prob
        self.techniques_per_sample = techniques_per_sample
        self.preserve_length = preserve_length

        logger.info(f"Diverse audio augmentation: {augmentation_prob*100:.0f}% samples augmented, "
                   f"{techniques_per_sample[0]}-{techniques_per_sample[1]} techniques per sample")

    def should_augment(self) -> bool:
        """Decide if this sample should be augmented"""
        return random.random() < self.augmentation_prob

    def select_random_techniques(self) -> List[str]:
        """Select random augmentation techniques for this sample"""
        available_techniques = [
            'gaussian_noise', 'pink_noise', 'speed_change', 'pitch_shift',
            'volume_change', 'lowpass_filter', 'highpass_filter',
            'frequency_mask', 'time_mask', 'reverb_simple'
        ]

        num_techniques = random.randint(*self.techniques_per_sample)
        return random.sample(available_techniques, min(num_techniques, len(available_techniques)))

    def augment_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply diverse augmentations to waveform"""
        # if not self.should_augment():
        #     return waveform

        selected_techniques = self.select_random_techniques()
        augmented = waveform.clone()

        for technique in selected_techniques:
            try:
                augmented = self._apply_technique(augmented, technique, sample_rate)
            except Exception as e:
                logger.warning(f"Augmentation technique '{technique}' failed: {e}")
                continue

        return augmented

    def _apply_technique(self, waveform: torch.Tensor, technique: str, sample_rate: int) -> torch.Tensor:
        """Apply specific augmentation technique"""

        if technique == 'gaussian_noise':
            return self._add_gaussian_noise(waveform)
        elif technique == 'pink_noise':
            return self._add_pink_noise(waveform)
        elif technique == 'speed_change':
            return self._change_speed(waveform, sample_rate)
        elif technique == 'pitch_shift':
            return self._change_pitch(waveform, sample_rate)
        elif technique == 'volume_change':
            return self._change_volume(waveform)
        elif technique == 'lowpass_filter':
            return self._apply_lowpass_filter(waveform, sample_rate)
        elif technique == 'highpass_filter':
            return self._apply_highpass_filter(waveform, sample_rate)
        elif technique == 'frequency_mask':
            return self._apply_frequency_mask(waveform, sample_rate)
        elif technique == 'time_mask':
            return self._apply_time_mask(waveform)
        elif technique == 'reverb_simple':
            return self._add_simple_reverb(waveform)
        else:
            logger.warning(f"Unknown augmentation technique: {technique}")
            return waveform

    def _add_gaussian_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with random SNR"""
        snr_db = random.uniform(15, 30)  # Higher quality noise
        signal_power = torch.mean(waveform ** 2)
        if signal_power == 0:
            return waveform

        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def _add_pink_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add pink noise (1/f noise)"""
        snr_db = random.uniform(18, 25)
        signal_power = torch.mean(waveform ** 2)
        if signal_power == 0:
            return waveform

        # Generate pink noise (simplified approximation)
        white_noise = torch.randn_like(waveform)
        # Apply simple 1/f filter approximation
        if len(waveform) > 1:
            pink_noise = torch.zeros_like(white_noise)
            pink_noise[0] = white_noise[0]
            for i in range(1, len(white_noise)):
                pink_noise[i] = 0.7 * pink_noise[i-1] + 0.3 * white_noise[i]
        else:
            pink_noise = white_noise

        noise_power = signal_power / (10 ** (snr_db / 10))
        pink_noise = pink_noise * torch.sqrt(noise_power) / (torch.std(pink_noise) + 1e-8)

        return waveform + pink_noise

    def _change_speed(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Change playback speed"""
        speed_factor = random.uniform(0.85, 1.15)

        try:
            if len(waveform.shape) == 1:
                waveform_2d = waveform.unsqueeze(0)
            else:
                waveform_2d = waveform

            augmented, _ = F.speed(waveform_2d, sample_rate, speed_factor)

            if len(waveform.shape) == 1:
                return augmented.squeeze(0)
            else:
                return augmented
        except:
            return waveform

    def _change_pitch(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Change pitch"""
        n_steps = random.uniform(-2, 2)

        try:
            if len(waveform.shape) == 1:
                waveform_2d = waveform.unsqueeze(0)
            else:
                waveform_2d = waveform

            augmented = F.pitch_shift(waveform_2d, sample_rate, n_steps)

            if len(waveform.shape) == 1:
                return augmented.squeeze(0)
            else:
                return augmented
        except:
            return waveform

    def _change_volume(self, waveform: torch.Tensor) -> torch.Tensor:
        """Change volume (gain)"""
        gain_db = random.uniform(-6, 6)  # ±6dB range
        gain_linear = 10 ** (gain_db / 20)
        return waveform * gain_linear

    def _apply_lowpass_filter(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply lowpass filter"""
        cutoff_freq = random.uniform(3000, 6000)  # Hz

        try:
            if len(waveform.shape) == 1:
                waveform_2d = waveform.unsqueeze(0)
            else:
                waveform_2d = waveform

            filtered = F.lowpass_biquad(waveform_2d, sample_rate, cutoff_freq)

            if len(waveform.shape) == 1:
                return filtered.squeeze(0)
            else:
                return filtered
        except:
            return waveform

    def _apply_highpass_filter(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply highpass filter"""
        cutoff_freq = random.uniform(100, 300)  # Hz

        try:
            if len(waveform.shape) == 1:
                waveform_2d = waveform.unsqueeze(0)
            else:
                waveform_2d = waveform

            filtered = F.highpass_biquad(waveform_2d, sample_rate, cutoff_freq)

            if len(waveform.shape) == 1:
                return filtered.squeeze(0)
            else:
                return filtered
        except:
            return waveform

    def _apply_frequency_mask(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply frequency masking (simulate frequency-specific interference)"""
        # This is a simplified version - in practice you'd use proper spectral masking
        try:
            # Apply a notch filter at random frequency
            center_freq = random.uniform(800, 4000)  # Hz
            quality_factor = random.uniform(2, 10)

            if len(waveform.shape) == 1:
                waveform_2d = waveform.unsqueeze(0)
            else:
                waveform_2d = waveform

            # Use bandpass as approximation for notch effect
            low_freq = center_freq * 0.9
            high_freq = center_freq * 1.1

            filtered = F.bandpass_biquad(waveform_2d, sample_rate, low_freq, high_freq)
            # Subtract filtered content to create notch effect
            result = waveform_2d - 0.3 * filtered

            if len(waveform.shape) == 1:
                return result.squeeze(0)
            else:
                return result
        except:
            return waveform

    def _apply_time_mask(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply time masking (zero out random time segments)"""
        mask_length = min(int(len(waveform) * 0.1), len(waveform) // 10)  # Max 10% or length/10
        if mask_length < 1:
            return waveform

        mask_start = random.randint(0, max(0, len(waveform) - mask_length))

        augmented = waveform.clone()
        augmented[mask_start:mask_start + mask_length] *= 0.1  # Reduce to 10% instead of complete zero
        return augmented

    def _add_simple_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add simple reverb effect"""
        try:
            # Simple reverb using delayed copies
            delay_samples = random.randint(800, 2400)  # 50-150ms at 16kHz
            decay_factor = random.uniform(0.2, 0.4)

            if len(waveform) <= delay_samples:
                return waveform

            reverb = torch.zeros_like(waveform)
            reverb[delay_samples:] = waveform[:-delay_samples] * decay_factor

            return waveform + reverb
        except:
            return waveform


class ComprehensiveAudioAugmenter:
    """Create comprehensive augmentation pool with multiple variants"""

    def __init__(self,
                 augmentation_ratio: float = 0.25,
                 num_variants: int = 3,
                 seed: int = 42):
        self.augmentation_ratio = augmentation_ratio
        self.num_variants = num_variants
        self.seed = seed
        random.seed(seed)
        self.main_random = random.Random(seed)

        # Initialize augmenter
        self.audio_augmenter = DiverseAudioAugmentation(
            augmentation_prob=1.0,
            techniques_per_sample=(1, 2),
            preserve_length=True
        )

    def should_augment_sample(self, sample_idx: int) -> bool:
        """Determine if sample should be augmented (3:1 ratio)"""
        local_random = random.Random(self.seed + sample_idx)
        return local_random.random() < self.augmentation_ratio

    def create_augmented_variants(self, audio_path: str, output_dir: Path,
                                  sample_id: str) -> List[str]:
        """Create multiple augmented variants of the same audio"""
        variants = []

        try:
            # Load original audio
            wav, sample_rate = torchaudio.load(audio_path)

            # Create multiple variants
            for variant_idx in range(self.num_variants):
                # Use different random seed for each variant
                variant_seed = self.seed + hash(f"{sample_id}_{variant_idx}") % 10000
                torch.manual_seed(variant_seed)

                # Apply augmentation
                if len(wav.shape) == 2 and wav.shape[0] > 1:
                    # Multi-channel
                    augmented_channels = []
                    for ch in range(wav.shape[0]):
                        aug_ch = self.audio_augmenter.augment_waveform(wav[ch], sample_rate)
                        augmented_channels.append(aug_ch)
                    augmented_wav = torch.stack(augmented_channels)
                else:
                    # Mono
                    if len(wav.shape) == 2:
                        wav = wav.squeeze(0)
                    augmented_wav = self.audio_augmenter.augment_waveform(wav, sample_rate)
                    if len(augmented_wav.shape) == 1:
                        augmented_wav = augmented_wav.unsqueeze(0)

                # Save variant
                variant_filename = f"{sample_id}_variant_{variant_idx}.wav"
                variant_path = output_dir / variant_filename
                variant_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(variant_path), augmented_wav, sample_rate)
                variants.append(str(variant_path))

        except Exception as e:
            logger.warning(f"Failed to create variants for {audio_path}: {e}")
            # Return original path for all variants on failure
            variants = [audio_path] * self.num_variants

        return variants

    def process_sample(self, sample_data: Dict, output_audio_dir: Path,
                       sample_idx: int) -> List[Dict]:
        """Process sample and return original + variants"""
        results = []

        try:
            # Always include original
            original_sample = copy.deepcopy(sample_data)
            original_sample['augmentation_info'] = {
                'is_augmented': False,
                'variant_type': 'original',
                'sample_idx': sample_idx
            }
            results.append(original_sample)

            # Add augmented variants if selected
            if self.should_augment_sample(sample_idx):
                original_audio_path = sample_data.get('source', {}).get('audio_local_path')

                if original_audio_path and os.path.exists(original_audio_path):
                    # Create variants
                    variant_paths = self.create_augmented_variants(
                        original_audio_path, output_audio_dir, f"sample_{sample_idx}"
                    )

                    # Create sample for each variant
                    for variant_idx, variant_path in enumerate(variant_paths):
                        variant_sample = copy.deepcopy(sample_data)
                        variant_sample['source']['audio_local_path'] = variant_path
                        variant_sample['augmentation_info'] = {
                            'is_augmented': True,
                            'variant_type': f'audio_variant_{variant_idx}',
                            'sample_idx': sample_idx,
                            'original_audio_path': original_audio_path
                        }
                        results.append(variant_sample)

        except Exception as e:
            logger.warning(f"Failed to process sample {sample_idx}: {e}")
            # Return at least the original
            if not results:
                results = [sample_data]

        return results


def create_comprehensive_pool(input_manifest_path: str,
                              output_manifest_path: str,
                              output_audio_dir: str,
                              augmentation_ratio: float = 0.25,
                              num_variants: int = 3,
                              num_workers: int = 4,
                              seed: int = 42):
    """Create comprehensive augmentation pool"""

    input_path = Path(input_manifest_path)
    output_path = Path(output_manifest_path)
    audio_dir = Path(output_audio_dir)

    # Create directories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Initialize augmenter
    augmenter = ComprehensiveAudioAugmenter(
        augmentation_ratio=augmentation_ratio,
        num_variants=num_variants,
        seed=seed
    )

    # Load samples
    logger.info(f"Loading manifest: {input_path}")
    with open(input_path, 'r') as f:
        samples = [json.loads(line) for line in f]

    total_samples = len(samples)
    expected_augmented = int(total_samples * augmentation_ratio)
    expected_total = total_samples + (expected_augmented * num_variants)

    logger.info(f"Processing {total_samples} samples")
    logger.info(f"Target ratio: {augmentation_ratio:.1%} augmented")
    logger.info(f"Expected augmented samples: {expected_augmented}")
    logger.info(f"Expected total output: {expected_total} (with {num_variants} variants each)")

    # Process samples
    all_results = []

    if num_workers > 1:
        # TODO: Implement multiprocessing if needed
        pass

    # Single process for now (multiprocessing can be complex with audio files)
    for i, sample in enumerate(tqdm(samples, desc="Creating augmentation pool")):
        sample_results = augmenter.process_sample(sample, audio_dir, i)
        all_results.extend(sample_results)

    # Save comprehensive manifest
    logger.info(f"Saving {len(all_results)} samples to: {output_path}")
    with open(output_path, 'w') as f:
        for sample in all_results:
            f.write(json.dumps(sample) + '\n')

    # Count variants
    original_count = sum(1 for s in all_results if not s.get('augmentation_info', {}).get('is_augmented', False))
    augmented_count = len(all_results) - original_count

    logger.info(f"Pool created:")
    logger.info(f"  Original samples: {original_count}")
    logger.info(f"  Augmented samples: {augmented_count}")
    logger.info(f"  Total samples: {len(all_results)}")
    logger.info(f"  Effective ratio: {original_count / (augmented_count or 1):.1f}:1")


def main():
    parser = argparse.ArgumentParser(description="Create comprehensive audio augmentation pool")
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--output_audio_dir", required=True)
    parser.add_argument("--stage", default="audio_comprehensive")
    parser.add_argument("--augmentation_ratio", type=float, default=0.25, help="3:1 ratio")
    parser.add_argument("--create_variants", type=int, default=3, help="Number of variants per sample")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    create_comprehensive_pool(
        input_manifest_path=args.input_manifest,
        output_manifest_path=args.output_manifest,
        output_audio_dir=args.output_audio_dir,
        augmentation_ratio=args.augmentation_ratio,
        num_variants=args.create_variants,
        num_workers=args.num_workers,
        seed=args.seed
    )


if __name__ == "__main__":
    main()