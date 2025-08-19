import json
import logging
import random
import tempfile
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from fairseq2.data.text import TextTokenEncoder
from fairseq2.models.nllb import NllbTokenizer
from fairseq2.data.audio import WaveformToFbankConverter
from torch import Tensor
from torch.nn.functional import pad as pad_tensor
from torch.utils.data import DataLoader

from seamless_communication.datasets.datatypes import LangPairSample
from seamless_communication.models.unity.unit_tokenizer import (
    UnitTokenEncoder,
    UnitTokenizer,
)

# Import original classes to maintain compatibility
from seamless_communication.cli.m4t.finetune.dataloader import (
    SeqsBatch,
    MultimodalSeqsBatch,
    BatchingConfig,
    worker_init_fn,
    UnitYDataLoader as OriginalUnitYDataLoader
)

logger = logging.getLogger(__name__)

# CRITICAL FIX: Add constants for validation
EXPECTED_SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH_SAMPLES = 320000  # 20 seconds at 16kHz
MIN_AUDIO_LENGTH_SAMPLES = 160    # 0.1 seconds at 16kHz


class DiverseAudioAugmentation:
    """FIXED: Diverse audio augmentation with length preservation and efficiency improvements"""

    def __init__(self,
                 augmentation_prob: float = 0.33,
                 techniques_per_sample: tuple = (1, 3),
                 preserve_length: bool = True):
        """
        Args:
            augmentation_prob: Probability that a sample gets augmented (0.33 = 33%)
            techniques_per_sample: Range of techniques to apply per augmented sample
            preserve_length: Whether to preserve original audio length (critical for batching)
        """
        self.augmentation_prob = augmentation_prob
        self.techniques_per_sample = techniques_per_sample
        self.preserve_length = preserve_length

        # CRITICAL FIX: Separate length-preserving and length-changing techniques
        # self.length_preserving_techniques = [
        #     'gaussian_noise', 'pink_noise', 'volume_change',
        #     'lowpass_filter', 'highpass_filter', 'time_mask', 'reverb_simple'
        # ]
        #
        # self.length_changing_techniques = [
        #     'speed_change', 'pitch_shift', 'frequency_mask'
        # ]

        self.length_preserving_techniques = [
            'gaussian_noise', 'volume_change',
            'lowpass_filter', 'highpass_filter', 'time_mask'
        ]

        self.length_changing_techniques = [
            'speed_change', 'pitch_shift'
        ]

        logger.info(f"Diverse audio augmentation: {augmentation_prob*100:.0f}% samples augmented, "
                   f"{techniques_per_sample[0]}-{techniques_per_sample[1]} techniques per sample, "
                   f"preserve_length={preserve_length}")

    def should_augment(self) -> bool:
        """Decide if this sample should be augmented"""
        return random.random() < self.augmentation_prob

    def select_random_techniques(self) -> List[str]:
        """FIXED: Select techniques based on length preservation requirements"""
        if self.preserve_length:
            available_techniques = self.length_preserving_techniques
        else:
            available_techniques = self.length_preserving_techniques + self.length_changing_techniques

        num_techniques = random.randint(*self.techniques_per_sample)
        return random.sample(available_techniques, min(num_techniques, len(available_techniques)))

    def augment_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """FIXED: Apply diverse augmentations with validation and error recovery"""

        # Input validation
        if not self._validate_input(waveform, sample_rate):
            return waveform

        if not self.should_augment():
            return waveform

        selected_techniques = self.select_random_techniques()
        augmented = waveform.clone()
        original_length = len(waveform)

        for technique in selected_techniques:
            try:
                augmented = self._apply_technique(augmented, technique, sample_rate)

                # CRITICAL FIX: Validate length preservation if required
                if self.preserve_length and len(augmented) != original_length:
                    logger.warning(f"Technique '{technique}' changed audio length: {original_length} -> {len(augmented)}, reverting")
                    augmented = waveform.clone()  # Revert to original

            except Exception as e:
                logger.warning(f"Augmentation technique '{technique}' failed: {e}")
                augmented = waveform.clone()  # Revert to original
                continue

        return augmented

    def _validate_input(self, waveform: torch.Tensor, sample_rate: int) -> bool:
        """CRITICAL FIX: Validate input audio parameters"""

        # Check sample rate
        if sample_rate != EXPECTED_SAMPLE_RATE:
            logger.warning(f"Unexpected sample rate: {sample_rate}, expected {EXPECTED_SAMPLE_RATE}")
            return False

        # Check audio length
        if len(waveform) < MIN_AUDIO_LENGTH_SAMPLES:
            logger.warning(f"Audio too short: {len(waveform)} samples")
            return False

        if len(waveform) > MAX_AUDIO_LENGTH_SAMPLES:
            logger.warning(f"Audio too long: {len(waveform)} samples")
            return False

        # Check for NaN or infinite values
        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            logger.warning("Audio contains NaN or infinite values")
            return False

        return True

    def _apply_technique(self, waveform: torch.Tensor, technique: str, sample_rate: int) -> torch.Tensor:
        """Apply specific augmentation technique with improved error handling"""

        technique_map = {
            'gaussian_noise': self._add_gaussian_noise,
            'pink_noise': self._add_pink_noise_efficient,  # FIXED: Use efficient version
            'speed_change': self._change_speed_preserve_length,  # FIXED: Length preserving version
            'pitch_shift': self._change_pitch_preserve_length,   # FIXED: Length preserving version
            'volume_change': self._change_volume,
            'lowpass_filter': self._apply_lowpass_filter,
            'highpass_filter': self._apply_highpass_filter,
            'frequency_mask': self._apply_frequency_mask_spectral,  # FIXED: Proper spectral masking
            'time_mask': self._apply_time_mask,
            'reverb_simple': self._add_simple_reverb
        }

        if technique not in technique_map:
            logger.warning(f"Unknown augmentation technique: {technique}")
            return waveform

        return technique_map[technique](waveform, sample_rate)

    def _add_gaussian_noise(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Add Gaussian noise with random SNR"""
        snr_db = random.uniform(15, 30)
        signal_power = torch.mean(waveform ** 2)
        if signal_power < 1e-10:  # Handle near-silent audio
            return waveform

        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def _add_pink_noise_efficient(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """CRITICAL FIX: Efficient pink noise generation using FFT"""
        snr_db = random.uniform(18, 25)
        signal_power = torch.mean(waveform ** 2)
        if signal_power < 1e-10:
            return waveform

        try:
            # Generate pink noise efficiently using frequency domain
            n_samples = len(waveform)

            # Create frequency array
            freqs = torch.fft.fftfreq(n_samples, 1/sample_rate)
            freqs = torch.abs(freqs)
            freqs[0] = 1.0  # Avoid division by zero

            # Generate white noise in frequency domain
            white_noise_fft = torch.fft.fft(torch.randn(n_samples))

            # Apply 1/f filter (pink noise characteristic)
            pink_filter = 1.0 / torch.sqrt(freqs)
            pink_noise_fft = white_noise_fft * pink_filter

            # Convert back to time domain
            pink_noise = torch.fft.ifft(pink_noise_fft).real

            # Normalize and scale
            noise_power = signal_power / (10 ** (snr_db / 10))
            pink_noise = pink_noise * torch.sqrt(noise_power) / (torch.std(pink_noise) + 1e-8)

            return waveform + pink_noise

        except Exception as e:
            logger.warning(f"Efficient pink noise generation failed: {e}, using white noise")
            return self._add_gaussian_noise(waveform, sample_rate)

    def _change_speed_preserve_length(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """CRITICAL FIX: Speed change that preserves length through resampling"""
        if not self.preserve_length:
            return self._change_speed_original(waveform, sample_rate)

        try:
            speed_factor = random.uniform(0.9, 1.1)  # Smaller range to preserve quality
            original_length = len(waveform)

            # Apply speed change
            waveform_2d = waveform.unsqueeze(0) if len(waveform.shape) == 1 else waveform
            augmented, _ = F.speed(waveform_2d, sample_rate, speed_factor)

            # Resample to original length
            if augmented.shape[-1] != original_length:
                augmented = F.resample(augmented, augmented.shape[-1], original_length)

            return augmented.squeeze(0) if len(waveform.shape) == 1 else augmented

        except Exception as e:
            logger.warning(f"Length-preserving speed change failed: {e}")
            return waveform

    def _change_pitch_preserve_length(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """CRITICAL FIX: Pitch shift that preserves length"""
        try:
            n_steps = random.uniform(-1, 1)  # Smaller range
            waveform_2d = waveform.unsqueeze(0) if len(waveform.shape) == 1 else waveform

            augmented = F.pitch_shift(waveform_2d, sample_rate, n_steps)

            return augmented.squeeze(0) if len(waveform.shape) == 1 else augmented

        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return waveform

    def _change_speed_original(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Original speed change (may change length)"""
        try:
            speed_factor = random.uniform(0.85, 1.15)
            waveform_2d = waveform.unsqueeze(0) if len(waveform.shape) == 1 else waveform
            augmented, _ = F.speed(waveform_2d, sample_rate, speed_factor)
            return augmented.squeeze(0) if len(waveform.shape) == 1 else augmented
        except Exception as e:
            logger.warning(f"Speed change failed: {e}")
            return waveform

    def _change_volume(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Change volume with clipping protection"""
        gain_db = random.uniform(-3, 3)  # Reduced range to prevent clipping
        gain_linear = 10 ** (gain_db / 20)
        augmented = waveform * gain_linear

        # Protect against clipping
        max_val = torch.max(torch.abs(augmented))
        if max_val > 1.0:
            augmented = augmented / max_val * 0.95

        return augmented

    def _apply_lowpass_filter(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply lowpass filter with validation"""
        try:
            cutoff_freq = random.uniform(4000, 7000)  # Higher range to preserve speech
            waveform_2d = waveform.unsqueeze(0) if len(waveform.shape) == 1 else waveform
            filtered = F.lowpass_biquad(waveform_2d, sample_rate, cutoff_freq)
            return filtered.squeeze(0) if len(waveform.shape) == 1 else filtered
        except Exception as e:
            logger.warning(f"Lowpass filter failed: {e}")
            return waveform

    def _apply_highpass_filter(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply highpass filter with validation"""
        try:
            cutoff_freq = random.uniform(80, 200)  # Conservative range
            waveform_2d = waveform.unsqueeze(0) if len(waveform.shape) == 1 else waveform
            filtered = F.highpass_biquad(waveform_2d, sample_rate, cutoff_freq)
            return filtered.squeeze(0) if len(waveform.shape) == 1 else filtered
        except Exception as e:
            logger.warning(f"Highpass filter failed: {e}")
            return waveform

    def _apply_frequency_mask_spectral(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """CRITICAL FIX: Proper spectral frequency masking"""
        try:
            # Convert to spectrogram
            n_fft = 1024
            hop_length = 256

            stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                             return_complex=True, normalized=False)

            # Apply frequency mask
            freq_bins = stft.shape[0]
            mask_width = random.randint(freq_bins // 20, freq_bins // 10)  # 5-10% of frequencies
            mask_start = random.randint(0, freq_bins - mask_width)

            # Create mask
            stft_masked = stft.clone()
            stft_masked[mask_start:mask_start + mask_width, :] *= 0.1  # Attenuate rather than zero

            # Convert back to waveform
            augmented = torch.istft(stft_masked, n_fft=n_fft, hop_length=hop_length,
                                   length=len(waveform))

            return augmented

        except Exception as e:
            logger.warning(f"Spectral frequency masking failed: {e}")
            return waveform

    def _apply_time_mask(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply time masking with conservative parameters"""
        mask_length = min(int(len(waveform) * 0.05), len(waveform) // 20)  # Max 5%
        if mask_length < 100:  # Minimum mask length
            return waveform

        mask_start = random.randint(0, max(0, len(waveform) - mask_length))
        augmented = waveform.clone()
        augmented[mask_start:mask_start + mask_length] *= 0.1  # Attenuate rather than zero
        return augmented

    def _add_simple_reverb(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Add simple reverb with multiple delays"""
        try:
            # Multiple delay lines for more realistic reverb
            delays = [
                random.randint(800, 1200),   # Early reflection
                random.randint(1600, 2400), # Late reflection
            ]
            decay_factors = [0.3, 0.2]

            reverb = torch.zeros_like(waveform)

            for delay, decay in zip(delays, decay_factors):
                if len(waveform) > delay:
                    reverb[delay:] += waveform[:-delay] * decay

            return waveform + reverb * 0.3  # Mix with original

        except Exception as e:
            logger.warning(f"Reverb failed: {e}")
            return waveform


class DiverseTextAugmentation:
    """FIXED: Text augmentation with linguistic preservation"""

    def __init__(self,
                 augmentation_prob: float = 0.33,
                 techniques_per_sample: tuple = (1, 2),
                 preserve_special_tokens: bool = True):
        """
        Args:
            augmentation_prob: Probability that a sample gets augmented
            techniques_per_sample: Range of techniques to apply per sample
            preserve_special_tokens: Whether to preserve language codes and special tokens
        """
        self.augmentation_prob = augmentation_prob
        self.techniques_per_sample = techniques_per_sample
        self.preserve_special_tokens = preserve_special_tokens

        # CRITICAL FIX: Define special tokens to preserve
        self.special_tokens = {
            '<', '>', '__', '©', '®', '™',
            'http', 'www', '.com', '.org', '.net',
            '@', '#', '$', '%', '&', '*'
        }

        logger.info(f"Diverse text augmentation: {augmentation_prob*100:.0f}% samples augmented, "
                   f"{techniques_per_sample[0]}-{techniques_per_sample[1]} techniques per sample, "
                   f"preserve_special_tokens={preserve_special_tokens}")

    def should_augment(self) -> bool:
        """Decide if this sample should be augmented"""
        return random.random() < self.augmentation_prob

    def select_random_techniques(self) -> List[str]:
        """Select random augmentation techniques"""
        available_techniques = [
            'word_dropout', 'word_shuffle', 'character_noise',
            'punctuation_noise', 'case_change'  # Removed word_repetition as it can be confusing
        ]

        num_techniques = random.randint(*self.techniques_per_sample)
        return random.sample(available_techniques, min(num_techniques, len(available_techniques)))

    def augment_text(self, text: str) -> str:
        """FIXED: Apply augmentation with validation and preservation"""

        # Input validation
        if not self._validate_text_input(text):
            return text

        if not self.should_augment():
            return text

        selected_techniques = self.select_random_techniques()
        augmented = text

        for technique in selected_techniques:
            try:
                augmented = self._apply_technique(augmented, technique)

                # Validate result
                if not self._validate_text_output(augmented, text):
                    augmented = text  # Revert on failure

            except Exception as e:
                logger.warning(f"Text augmentation technique '{technique}' failed: {e}")
                augmented = text  # Revert to original
                continue

        return augmented if augmented.strip() else text

    def _validate_text_input(self, text: str) -> bool:
        """Validate input text"""
        if not text or len(text.strip()) < 3:
            return False
        if len(text) > 1000:  # Very long text
            return False
        return True

    def _validate_text_output(self, augmented: str, original: str) -> bool:
        """Validate augmented text quality"""
        if not augmented or len(augmented.strip()) < max(1, len(original.split()) // 2):
            return False  # Too much content lost
        return True

    def _apply_technique(self, text: str, technique: str) -> str:
        """Apply specific text augmentation technique"""

        technique_map = {
            'word_dropout': self._word_dropout_safe,
            'word_shuffle': self._word_shuffle_conservative,
            'character_noise': self._add_character_noise_safe,
            'punctuation_noise': self._punctuation_noise_conservative,
            'case_change': self._random_case_change_safe
        }

        if technique not in technique_map:
            logger.warning(f"Unknown text augmentation technique: {technique}")
            return text

        return technique_map[technique](text)

    def _word_dropout_safe(self, text: str) -> str:
        """FIXED: Safe word dropout that preserves meaning"""
        words = text.split()
        if len(words) <= 3:  # Don't dropout from very short texts
            return text

        # Identify words that should never be dropped
        protected_words = []
        for i, word in enumerate(words):
            if (i == 0 or i == len(words) - 1 or  # First/last word
                len(word) <= 2 or  # Very short words
                any(special in word.lower() for special in self.special_tokens)):
                protected_words.append(i)

        dropout_rate = random.uniform(0.05, 0.1)  # Reduced rate
        kept_words = []

        for i, word in enumerate(words):
            if i in protected_words or random.random() > dropout_rate:
                kept_words.append(word)

        return ' '.join(kept_words) if len(kept_words) >= max(2, len(words) // 2) else text

    def _word_shuffle_conservative(self, text: str) -> str:
        """FIXED: Conservative word shuffling that preserves structure"""
        words = text.split()
        if len(words) <= 4:
            return text

        augmented_words = words.copy()

        # Only shuffle middle words, preserve first and last
        middle_start = 1
        middle_end = len(words) - 1

        if middle_end - middle_start >= 2:
            # Shuffle in small windows
            window_size = 2
            for i in range(middle_start, middle_end - window_size + 1, window_size):
                if random.random() < 0.3:  # 30% chance
                    window = augmented_words[i:i + window_size]
                    random.shuffle(window)
                    augmented_words[i:i + window_size] = window

        return ' '.join(augmented_words)

    def _add_character_noise_safe(self, text: str) -> str:
        """FIXED: Safe character noise that preserves readability"""
        if len(text) < 10:
            return text

        chars = list(text)
        noise_rate = random.uniform(0.005, 0.015)  # Very low rate

        for i in range(1, len(chars) - 1):  # Don't modify first/last character
            if (chars[i].isalpha() and
                random.random() < noise_rate and
                not any(special in text[max(0, i-3):i+4].lower() for special in self.special_tokens)):

                # Only substitute with similar characters
                similar_chars = {
                    'a': 'aeio', 'e': 'aeio', 'i': 'aeio', 'o': 'aeio', 'u': 'aeio',
                    'b': 'bp', 'p': 'bp', 'd': 'bd', 'g': 'gq',
                    'm': 'mn', 'n': 'mn', 'f': 'fv', 'v': 'fv'
                }

                original = chars[i].lower()
                if original in similar_chars:
                    replacement = random.choice(similar_chars[original])
                    chars[i] = replacement if chars[i].islower() else replacement.upper()

        return ''.join(chars)

    def _punctuation_noise_conservative(self, text: str) -> str:
        """FIXED: Conservative punctuation changes"""
        if random.random() < 0.2:  # Lower probability
            if random.choice([True, False]):
                # Add punctuation only if none exists
                if not text.rstrip().endswith(('.', '!', '?', ';', ':')):
                    text += random.choice(['.', '!', '?'])
            else:
                # Remove punctuation only if it exists
                if text.rstrip().endswith(('.', '!', '?', ';', ':')):
                    text = text.rstrip()[:-1]
        return text

    def _random_case_change_safe(self, text: str) -> str:
        """FIXED: Safe case changes that preserve proper nouns"""
        words = text.split()

        for i in range(len(words)):
            # Don't change case of first word or words that look like proper nouns
            if (i > 0 and
                not words[i][0].isupper() and  # Not already capitalized
                len(words[i]) > 2 and
                random.random() < 0.05):  # Very low probability

                if words[i].islower():
                    words[i] = words[i].capitalize()

        return ' '.join(words)


class AugmentedUnitYDataLoader(OriginalUnitYDataLoader):
    """FIXED: Enhanced UnitYDataLoader with robust augmentation"""

    def __init__(
        self,
        text_tokenizer: NllbTokenizer,
        unit_tokenizer: UnitTokenizer,
        dataset_manifest_path: str,
        batching_config: BatchingConfig,
        max_src_tokens_per_batch: int = 100000,
        # Augmentation parameters
        enable_audio_aug: bool = False,
        enable_text_aug: bool = False,
        augmentation_prob: float = 0.33,
        is_training: bool = True,
        audio_aug_config: Optional[Dict[str, Any]] = None,
        text_aug_config: Optional[Dict[str, Any]] = None,
        num_workers: int = 0
    ):
        batching_config.num_workers = num_workers

        # Initialize parent class
        super().__init__(
            text_tokenizer=text_tokenizer,
            unit_tokenizer=unit_tokenizer,
            dataset_manifest_path=dataset_manifest_path,
            batching_config=batching_config,
            max_src_tokens_per_batch=max_src_tokens_per_batch
        )

        self.invalid_audio_count = 0
        self.total_audio_count = 0

        # CRITICAL FIX: Validate augmentation parameters
        self._validate_augmentation_config(audio_aug_config, text_aug_config)

        # Initialize augmentation (only for training)
        self.enable_audio_aug = enable_audio_aug and is_training
        self.enable_text_aug = enable_text_aug and is_training
        self.is_training = is_training

        if self.enable_audio_aug:
            audio_config = audio_aug_config or {}
            audio_config['augmentation_prob'] = augmentation_prob
            audio_config['preserve_length'] = True  # CRITICAL: Always preserve length for batching
            self.audio_augmenter = DiverseAudioAugmentation(**audio_config)
        else:
            self.audio_augmenter = None

        if self.enable_text_aug:
            text_config = text_aug_config or {}
            text_config['augmentation_prob'] = augmentation_prob
            text_config['preserve_special_tokens'] = True  # CRITICAL: Preserve special tokens
            self.text_augmenter = DiverseTextAugmentation(**text_config)
        else:
            self.text_augmenter = None

        # Log configuration
        if self.is_training and (self.enable_audio_aug or self.enable_text_aug):
            logger.info(f"=== Augmentation enabled for training: "
                       f"audio={self.enable_audio_aug}, text={self.enable_text_aug}, "
                       f"prob={augmentation_prob*100:.0f}%")
        else:
            logger.info("??? No augmentation (validation mode or disabled)")

    def _validate_augmentation_config(self, audio_config: Optional[Dict], text_config: Optional[Dict]):
        """CRITICAL FIX: Validate augmentation configuration"""
        if audio_config:
            techniques_per_sample = audio_config.get('techniques_per_sample', (1, 3))
            if not isinstance(techniques_per_sample, tuple) or len(techniques_per_sample) != 2:
                raise ValueError("audio_config.techniques_per_sample must be a tuple of (min, max)")
            if techniques_per_sample[0] > techniques_per_sample[1]:
                raise ValueError("audio_config.techniques_per_sample min > max")

        if text_config:
            techniques_per_sample = text_config.get('techniques_per_sample', (1, 2))
            if not isinstance(techniques_per_sample, tuple) or len(techniques_per_sample) != 2:
                raise ValueError("text_config.techniques_per_sample must be a tuple of (min, max)")
            if techniques_per_sample[0] > techniques_per_sample[1]:
                raise ValueError("text_config.techniques_per_sample min > max")

    def _get_source_fbank(self, sample: LangPairSample) -> Tensor:
        """FIXED: Robust audio processing with comprehensive error handling"""
        self.total_audio_count += 1
        try:
            # Load audio exactly like parent class
            wav, sample_rate = torchaudio.load(sample.source.audio_local_path)

            # CRITICAL FIX: Standardize shape FIRST, then validate
            wav = self._standardize_audio_shape(wav)

            # CRITICAL FIX: Validate loaded audio
            if not self._validate_loaded_audio(wav, sample_rate, sample.source.audio_local_path):
                self.invalid_audio_count += 1
                # Log summary every 1000 files instead of individual warnings
                if self.invalid_audio_count % 1000 == 0:
                    logger.info(f"Skipped {self.invalid_audio_count}/{self.total_audio_count} invalid audio files so far")
                raise ValueError(f"Invalid audio")


            # Apply audio augmentation if enabled
            if self.enable_audio_aug and self.audio_augmenter is not None:
                try:
                    # Process mono audio (most common case)
                    if wav.shape[1] == 1:
                        augmented_wav = self.audio_augmenter.augment_waveform(
                            wav.squeeze(-1), sample_rate
                        ).unsqueeze(-1)
                    else:
                        # Process multi-channel audio
                        augmented_channels = []
                        for ch in range(wav.shape[1]):
                            aug_ch = self.audio_augmenter.augment_waveform(
                                wav[:, ch], sample_rate
                            )
                            augmented_channels.append(aug_ch)
                        augmented_wav = torch.stack(augmented_channels, dim=1)

                    # Validate augmented audio
                    if self._validate_augmented_audio(augmented_wav, wav):
                        wav = augmented_wav
                    else:
                        logger.warning(f"Augmented audio validation failed for {sample.source.audio_local_path}")

                except Exception as e:
                    logger.warning(f"Audio augmentation failed for {sample.source.audio_local_path}: {e}")

            # Extract fbank features exactly like parent class
            fbank_result = WaveformToFbankConverter(**self._fbank_extract_params)(
                {
                    "waveform": wav,
                    "sample_rate": self.SAMPLE_RATE,
                }
            )["fbank"]
            return fbank_result

        except Exception as e:
            # Better error recovery - create a proper minimum-length fbank
            if self.invalid_audio_count % 100 == 0:  # Log every 100 failures
                logger.warning(f"Failed to process audio {sample.source.audio_local_path}: {e}")

            # Return a proper dummy fbank (minimum valid length)
            # Use 10 frames (about 0.1 seconds) instead of just 1
            return torch.zeros((10, 80), dtype=self.batching_config.float_dtype)

    def _validate_loaded_audio(self, wav: torch.Tensor, sample_rate: int, audio_path: str) -> bool:
        """CRITICAL FIX: Comprehensive audio validation"""

        return (wav.numel() > MIN_AUDIO_LENGTH_SAMPLES and
            sample_rate == EXPECTED_SAMPLE_RATE and
            not torch.isnan(wav).any())

    def _standardize_audio_shape(self, wav: torch.Tensor) -> torch.Tensor:
        """CRITICAL FIX: Robust audio shape standardization with validation"""

        # Handle 1D audio - convert to [samples, 1]
        if len(wav.shape) == 1:
            return wav.unsqueeze(-1)

        # Handle 2D audio
        elif len(wav.shape) == 2:
            # If first dimension looks like channels (small number), transpose
            if wav.shape[0] <= 8 and wav.shape[0] < wav.shape[1]:  # Up to 8 channels max
                return wav.transpose(0, 1)  # [channels, samples] → [samples, channels]
            else:
                return wav  # Already [samples, channels]

        else:
            # Unexpected dimensions - flatten and treat as mono
            logger.warning(f"Unexpected audio shape {wav.shape}, flattening to mono")
            return wav.flatten().unsqueeze(-1)

    def _validate_augmented_audio(self, augmented: torch.Tensor, original: torch.Tensor) -> bool:
        """Validate that augmented audio is reasonable"""

        # Check shape preservation
        if augmented.shape != original.shape:
            return False

        # Check for NaN/inf
        if torch.isnan(augmented).any() or torch.isinf(augmented).any():
            return False

        # Check amplitude range (should be reasonable)
        max_amplitude = torch.max(torch.abs(augmented))
        if max_amplitude > 10.0:  # Unreasonably loud
            return False

        return True

    def _get_tokenized_target_text(self, sample: LangPairSample) -> Tensor:
        """FIXED: Robust text processing with validation"""
        target_lang = sample.target.lang
        target_text = sample.target.text

        # Apply text augmentation if enabled
        if self.enable_text_aug and self.text_augmenter is not None:
            try:
                augmented_text = self.text_augmenter.augment_text(target_text)

                # Use augmented text only if it's valid
                if self._validate_augmented_text(augmented_text, target_text):
                    target_text = augmented_text
                else:
                    logger.warning(f"Text augmentation validation failed, using original")

            except Exception as e:
                logger.warning(f"Text augmentation failed: {e}")

        # Tokenize exactly like parent class
        if target_lang not in self.text_encoders_per_lang:
            self.text_encoders_per_lang[target_lang] = (
                self.text_tokenizer.create_encoder(lang=target_lang, mode="target")
            )

        try:
            tokens = self.text_encoders_per_lang[target_lang](target_text)
            eos_idx = self.text_tokenizer.vocab_info.eos_idx
            tokens = torch.concat([tokens, torch.LongTensor([eos_idx])])
            return tokens
        except Exception as e:
            logger.error(f"Tokenization failed for text: '{target_text}': {e}")
            # Return a minimal valid token sequence
            eos_idx = self.text_tokenizer.vocab_info.eos_idx
            return torch.LongTensor([eos_idx])

    def _validate_augmented_text(self, augmented: str, original: str) -> bool:
        """Validate augmented text quality"""

        # Check minimum length
        if len(augmented.strip()) < max(3, len(original) // 3):
            return False

        # Check maximum length (shouldn't grow too much)
        if len(augmented) > len(original) * 2:
            return False

        # Check that we still have some words
        if len(augmented.split()) < max(1, len(original.split()) // 3):
            return False

        return True


def create_compatible_dataloader(
    text_tokenizer: NllbTokenizer,
    unit_tokenizer: UnitTokenizer,
    dataset_manifest_path: str,
    batching_config: BatchingConfig,
    max_src_tokens_per_batch: int = 100000,
    augmentation_config: Optional[Dict[str, Any]] = None,
    is_training: bool = True,
    num_workers: int = 0  # NEW: Add num_workers parameter
) -> OriginalUnitYDataLoader:
    """Factory function with num_workers parameter"""

    # Validate inputs
    if not Path(dataset_manifest_path).exists():
        raise FileNotFoundError(f"Dataset manifest not found: {dataset_manifest_path}")

    if (augmentation_config is None or
        augmentation_config.get("stage") == "none" or
        not is_training):
        # Return original dataloader with custom num_workers
        batching_config.num_workers = num_workers  # Override here too
        return OriginalUnitYDataLoader(
            text_tokenizer=text_tokenizer,
            unit_tokenizer=unit_tokenizer,
            dataset_manifest_path=dataset_manifest_path,
            batching_config=batching_config,
            max_src_tokens_per_batch=max_src_tokens_per_batch
        )

    # Validate augmentation config
    stage = augmentation_config.get("stage", "none")
    if stage not in ["audio_only", "text_only", "both"]:
        raise ValueError(f"Invalid augmentation stage: {stage}")

    augmentation_prob = augmentation_config.get("augmentation_prob", 0.33)
    if not 0.0 <= augmentation_prob <= 1.0:
        raise ValueError(f"Invalid augmentation_prob: {augmentation_prob}")

    audio_config = augmentation_config.get("audio_config", {})
    text_config = augmentation_config.get("text_config", {})

    try:
        if stage == "audio_only":
            return AugmentedUnitYDataLoader(
                text_tokenizer=text_tokenizer,
                unit_tokenizer=unit_tokenizer,
                dataset_manifest_path=dataset_manifest_path,
                batching_config=batching_config,
                max_src_tokens_per_batch=max_src_tokens_per_batch,
                enable_audio_aug=True,
                enable_text_aug=False,
                augmentation_prob=augmentation_prob,
                is_training=is_training,
                audio_aug_config=audio_config or {'techniques_per_sample': (1, 3)},
                num_workers=num_workers
            )

        elif stage == "text_only":
            return AugmentedUnitYDataLoader(
                text_tokenizer=text_tokenizer,
                unit_tokenizer=unit_tokenizer,
                dataset_manifest_path=dataset_manifest_path,
                batching_config=batching_config,
                max_src_tokens_per_batch=max_src_tokens_per_batch,
                enable_audio_aug=False,
                enable_text_aug=True,
                augmentation_prob=augmentation_prob,
                is_training=is_training,
                text_aug_config=text_config or {'techniques_per_sample': (1, 2)},
                num_workers=num_workers
            )

        elif stage == "both":
            return AugmentedUnitYDataLoader(
                text_tokenizer=text_tokenizer,
                unit_tokenizer=unit_tokenizer,
                dataset_manifest_path=dataset_manifest_path,
                batching_config=batching_config,
                max_src_tokens_per_batch=max_src_tokens_per_batch,
                enable_audio_aug=True,
                enable_text_aug=True,
                augmentation_prob=augmentation_prob * 0.7,  # Lighter augmentation
                is_training=is_training,
                audio_aug_config=audio_config or {'techniques_per_sample': (1, 2)},
                text_aug_config=text_config or {'techniques_per_sample': (1, 1)},
                num_workers=num_workers
            )

    except Exception as e:
        logger.error(f"Failed to create augmented dataloader: {e}")
        logger.info("Falling back to original dataloader")
        return OriginalUnitYDataLoader(
            text_tokenizer=text_tokenizer,
            unit_tokenizer=unit_tokenizer,
            dataset_manifest_path=dataset_manifest_path,
            batching_config=batching_config,
            max_src_tokens_per_batch=max_src_tokens_per_batch
        )
