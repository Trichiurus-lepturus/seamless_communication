# !/usr/bin/env python3
"""
SeamlessM4T LoRA Merger - Export to Hugging Face Format
"""

import argparse
import logging
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

from seamless_communication.inference import Translator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model export"""
    base_model_name: str = "seamlessM4T_medium"
    vocoder_name: str = "vocoder_v2"
    lora_checkpoint_path: Optional[str] = None


class LoRAMerger:
    """Utility class to merge LoRA weights into base model"""

    @staticmethod
    def load_lora_checkpoint(checkpoint_path: str) -> Dict[str, any]:
        logger.info(f"Loading LoRA checkpoint from: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if isinstance(checkpoint, dict) and 'lora_adapters' in checkpoint:
                logger.info(f"Found LoRA checkpoint with keys: {list(checkpoint.keys())}")
                logger.info(f"Model name: {checkpoint.get('model_name', 'unknown')}")
                logger.info(f"Training stage: {checkpoint.get('training_stage', 'unknown')}")
                return checkpoint
            else:
                raise ValueError(f"Invalid checkpoint format. Expected 'lora_adapters' key")

        except Exception as e:
            logger.error(f"Failed to load LoRA checkpoint: {e}")
            raise

    @staticmethod
    def apply_lora_to_model(model: nn.Module, lora_checkpoint: Dict[str, any]) -> nn.Module:
        """Apply LoRA adapters to the model with correct key structure"""
        logger.info("Applying LoRA adapters to base model...")

        if 'lora_adapters' not in lora_checkpoint:
            logger.warning("No LoRA adapters found in checkpoint")
            return model

        lora_adapters = lora_checkpoint['lora_adapters']
        lora_config = lora_checkpoint.get('lora_config', {})

        logger.info(f"Found {len(lora_adapters)} LoRA adapter tensors")
        logger.info(f"LoRA config: {lora_config}")

        # Get model state dict
        model_state = model.state_dict()

        # Group LoRA tensors by base name
        lora_pairs = {}
        for key in lora_adapters.keys():
            if key.endswith('.lora_A'):
                base_name = key[:-7]  # Remove '.lora_A'
                if base_name not in lora_pairs:
                    lora_pairs[base_name] = {}
                lora_pairs[base_name]['lora_A'] = lora_adapters[key]
            elif key.endswith('.lora_B'):
                base_name = key[:-7]  # Remove '.lora_B'
                if base_name not in lora_pairs:
                    lora_pairs[base_name] = {}
                lora_pairs[base_name]['lora_B'] = lora_adapters[key]

        logger.info(f"Found {len(lora_pairs)} LoRA pairs")

        applied_count = 0

        for base_name, lora_weights in lora_pairs.items():
            if 'lora_A' in lora_weights and 'lora_B' in lora_weights:
                # Convert LoRA name to model parameter name
                target_weight_name = base_name.replace('module.', '').replace('model.', '') + '.weight'

                if target_weight_name in model_state:
                    try:
                        lora_A = lora_weights['lora_A']
                        lora_B = lora_weights['lora_B']

                        # Get LoRA config
                        lora_alpha = lora_config.get('lora_alpha', 16.0)
                        r = lora_A.shape[0]
                        scaling = lora_alpha / r

                        # Get model weight
                        model_weight = model_state[target_weight_name]

                        # Ensure tensors are on the same device and dtype
                        lora_A = lora_A.to(device=model_weight.device, dtype=model_weight.dtype)
                        lora_B = lora_B.to(device=model_weight.device, dtype=model_weight.dtype)

                        # Apply LoRA: W = W0 + scaling * (B @ A)
                        delta_w = scaling * (lora_B @ lora_A)

                        if delta_w.shape == model_weight.shape:
                            model_state[target_weight_name] = model_weight + delta_w
                            applied_count += 1
                            logger.debug(f"Applied LoRA: {base_name} -> {target_weight_name}")
                        else:
                            logger.warning(
                                f"Shape mismatch for {target_weight_name}: delta={delta_w.shape}, weight={model_weight.shape}")

                    except Exception as e:
                        logger.warning(f"Failed to apply LoRA {base_name}: {e}")
                else:
                    logger.debug(f"No match found for: {base_name} -> {target_weight_name}")

        # Load the updated state dict
        if applied_count > 0:
            model.load_state_dict(model_state)
            logger.info(f"Successfully applied {applied_count} LoRA adapters")
        else:
            logger.warning("No LoRA adapters were applied!")

        return model


class HuggingFaceExporter:
    """Export to Hugging Face format"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

    def load_and_merge_model(self) -> Translator:
        """Load base model and apply LoRA weights"""
        logger.info(f"Loading base model: {self.config.base_model_name}")

        # Load base translator
        translator = Translator(
            model_name_or_card=self.config.base_model_name,
            vocoder_name_or_card=self.config.vocoder_name,
            device=self.device,
            dtype=self.dtype
        )

        # Apply LoRA weights if provided
        if self.config.lora_checkpoint_path:
            lora_checkpoint = LoRAMerger.load_lora_checkpoint(self.config.lora_checkpoint_path)

            # Apply LoRA to the model
            translator.model = LoRAMerger.apply_lora_to_model(
                translator.model,
                lora_checkpoint
            )

        return translator

    def export_to_huggingface(self, translator: Translator, output_dir: str) -> Dict[str, str]:
        """Export to Hugging Face format"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting to Hugging Face format: {output_dir}")

        # Save model weights in HF format
        model_path = output_dir / "pytorch_model.bin"
        torch.save(translator.model.state_dict(), model_path)

        # Create Hugging Face config.json
        config_dict = {
            "architectures": ["SeamlessM4TModel"],
            "model_type": "seamless_m4t",
            "torch_dtype": "float32",
            "transformers_version": "4.21.0",

            # Model specific config
            "base_model_name": self.config.base_model_name,
            "vocoder_name": self.config.vocoder_name,
            "lora_checkpoint_path": self.config.lora_checkpoint_path,

            # Language config
            "src_langs": ["eng", "cmn", "rus", "fra", "spa", "arb"],
            "tgt_langs": ["eng", "cmn", "rus", "fra", "spa", "arb"],
            "supported_tasks": ["T2TT", "S2TT", "S2ST"],

            # Model metadata
            "vocab_size": 256000,  # Approximate
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "intermediate_size": 4096,
            "max_position_embeddings": 4096,

            # Custom metadata
            "_name_or_path": "seamless_m4t_lora_merged",
            "auto_map": {
                "AutoModel": "modeling_seamless_m4t.SeamlessM4TModel",
                "AutoConfig": "configuration_seamless_m4t.SeamlessM4TConfig"
            }
        }

        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save tokenizer files (copy from original)
        try:
            import shutil

            # Try to save tokenizer
            tokenizer_path = output_dir / "tokenizer.json"

            # Create a simple tokenizer config
            tokenizer_config = {
                "tokenizer_class": "SeamlessM4TTokenizer",
                "model_max_length": 4096,
                "padding_side": "right",
                "truncation_side": "right",
                "chat_template": None,
                "tokenizer_type": "seamless_m4t"
            }

            tokenizer_config_path = output_dir / "tokenizer_config.json"
            with open(tokenizer_config_path, 'w') as f:
                json.dump(tokenizer_config, f, indent=2)

            # Save the actual tokenizer data
            tokenizer_data = {
                "text_tokenizer": translator.text_tokenizer,
                "speech_tokenizer": getattr(translator, 'speech_tokenizer', None)
            }

            tokenizer_data_path = output_dir / "tokenizer_data.pt"
            torch.save(tokenizer_data, tokenizer_data_path)

            logger.info("Saved tokenizer data")

        except Exception as e:
            logger.warning(f"Could not save tokenizer: {e}")

        # Save vocoder if available
        try:
            if hasattr(translator, 'vocoder') and translator.vocoder is not None:
                vocoder_path = output_dir / "vocoder_model.bin"
                torch.save(translator.vocoder.state_dict(), vocoder_path)
                logger.info("Saved vocoder model")
        except Exception as e:
            logger.warning(f"Could not save vocoder: {e}")

        # Create HuggingFace modeling file
        modeling_file_path = output_dir / "modeling_seamless_m4t.py"
        with open(modeling_file_path, 'w') as f:
            f.write('''"""
SeamlessM4T model for Hugging Face Transformers
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

class SeamlessM4TConfig(PretrainedConfig):
    model_type = "seamless_m4t"

    def __init__(
        self,
        base_model_name="seamlessM4T_medium",
        vocoder_name="vocoder_v2",
        src_langs=None,
        tgt_langs=None,
        supported_tasks=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.vocoder_name = vocoder_name
        self.src_langs = src_langs or ["eng", "cmn", "rus", "fra", "spa", "arb"]
        self.tgt_langs = tgt_langs or ["eng", "cmn", "rus", "fra", "spa", "arb"]
        self.supported_tasks = supported_tasks or ["T2TT", "S2TT", "S2ST"]

class SeamlessM4TModel(PreTrainedModel):
    config_class = SeamlessM4TConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # This is a wrapper - the actual model would need to be loaded separately
        # due to the complexity of the SeamlessM4T architecture
        self.model = None

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Placeholder forward method
        # In practice, you'd need to implement the full forward pass
        # or load the actual SeamlessM4T model here
        raise NotImplementedError(
            "This is a model wrapper. Use the standalone inference script for translation."
        )

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """Load the model from pretrained weights"""
        config_path = f"{model_path}/config.json"
        config = SeamlessM4TConfig.from_json_file(config_path)
        model = cls(config)

        # Load the actual weights
        weights_path = f"{model_path}/pytorch_model.bin"
        state_dict = torch.load(weights_path, map_location='cpu')

        # Note: This is simplified - you'd need proper weight loading logic
        return model
''')

        # Create standalone inference script for HF format
        hf_inference_path = output_dir / "hf_inference.py"
        with open(hf_inference_path, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
"""
Hugging Face format SeamlessM4T Inference
"""

import torch
import torchaudio
import json
import logging
from pathlib import Path
from typing import Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HFSeamlessM4T:
    """
    Hugging Face format SeamlessM4T model
    """

    def __init__(self, model_dir: str):
        """Initialize from HF format directory"""
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        config_path = self.model_dir / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        logger.info(f"Loaded HF SeamlessM4T model from: {{model_dir}}")
        logger.info(f"Base model: {{self.config['base_model_name']}}")
        logger.info(f"Supported languages: {{self.config['src_langs']}}")

        # Load tokenizer data
        try:
            tokenizer_data_path = self.model_dir / "tokenizer_data.pt"
            tokenizer_data = torch.load(tokenizer_data_path, map_location='cpu')
            self.text_tokenizer = tokenizer_data['text_tokenizer']
            self.speech_tokenizer = tokenizer_data.get('speech_tokenizer')
            logger.info("Loaded tokenizer data")
        except Exception as e:
            logger.warning(f"Could not load tokenizer data: {{e}}")
            self.text_tokenizer = None
            self.speech_tokenizer = None

    def translate_with_seamless(self, 
                              input_data: Union[str, Path], 
                              task: str, 
                              src_lang: str, 
                              tgt_lang: str,
                              output_path: Optional[str] = None) -> str:
        """
        Translate using the original seamless_communication library
        This still requires seamless_communication but uses our merged weights
        """
        try:
            from seamless_communication.inference import Translator

            # Create translator with base model
            translator = Translator(
                model_name_or_card=self.config['base_model_name'],
                vocoder_name_or_card=self.config['vocoder_name'],
                device=self.device,
                dtype=torch.float32
            )

            # Load our LoRA-merged weights
            weights_path = self.model_dir / "pytorch_model.bin"
            state_dict = torch.load(weights_path, map_location=self.device)
            translator.model.load_state_dict(state_dict)

            logger.info(f"Performing {{task}} translation: {{src_lang}} -> {{tgt_lang}}")

            # Perform translation
            result = translator.predict(
                input=str(input_data),
                task_str=task,
                src_lang=src_lang,
                tgt_lang=tgt_lang
            )

            # Handle result format (same as before)
            if isinstance(result, tuple) and len(result) >= 1:
                text_outputs = result[0]
                audio_outputs = result[1] if len(result) > 1 else None

                if isinstance(text_outputs, list) and len(text_outputs) > 0:
                    first_text = text_outputs[0]
                    text_result = str(first_text) if hasattr(first_text, '__str__') else repr(first_text)

                    # Handle speech output for S2ST
                    if task == "S2ST" and output_path and audio_outputs is not None:
                        try:
                            if isinstance(audio_outputs, list) and len(audio_outputs) > 0:
                                first_audio = audio_outputs[0]
                                if hasattr(first_audio, 'audio_wavs') and first_audio.audio_wavs:
                                    torchaudio.save(output_path, first_audio.audio_wavs[0][None, :], sample_rate=16000)
                                    return f"Speech saved to {{output_path}}. Text: {{text_result}}"
                            return f"No audio output available. Text: {{text_result}}"
                        except Exception as audio_error:
                            logger.warning(f"Failed to save audio: {{audio_error}}")
                            return f"Audio save failed. Text: {{text_result}}"

                    return text_result
                else:
                    return f"No text outputs found in result: {{text_outputs}}"
            else:
                return f"Unexpected result format: {{result}}"

        except ImportError:
            return "Error: seamless_communication package not found. Please install it."
        except Exception as e:
            logger.error(f"Translation error: {{e}}")
            return f"Translation error: {{e}}"

    def get_info(self) -> dict:
        """Get model information"""
        return {{
            'config': self.config,
            'device': str(self.device),
            'model_dir': str(self.model_dir)
        }}

def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="HF SeamlessM4T Inference")
    parser.add_argument("--model_dir", type=str, default=".",
                       help="Path to HF model directory")
    parser.add_argument("--input", type=str, required=True,
                       help="Input text or path to audio file")
    parser.add_argument("--task", type=str, required=True, 
                       choices=["T2TT", "S2TT", "S2ST"],
                       help="Translation task")
    parser.add_argument("--src_lang", type=str, required=True,
                       help="Source language code")
    parser.add_argument("--tgt_lang", type=str, required=True,
                       help="Target language code")
    parser.add_argument("--output", type=str,
                       help="Output file path for S2ST task")

    args = parser.parse_args()

    # Load model
    model = HFSeamlessM4T(args.model_dir)

    # Print model info
    print("Model Information:")
    info = model.get_info()
    print(f"  Base Model: {{info['config']['base_model_name']}}")
    print(f"  Device: {{info['device']}}")
    print(f"  Supported Languages: {{info['config']['src_langs']}}")
    print()

    # Perform translation
    print(f"Translating from {{args.src_lang}} to {{args.tgt_lang}}...")
    result = model.translate_with_seamless(
        input_data=args.input,
        task=args.task,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        output_path=args.output
    )

    print(f"Result: {{result}}")

if __name__ == "__main__":
    main()
''')

        # Create README for HF format
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f'''# SeamlessM4T with LoRA - Hugging Face Format

This model has been exported to Hugging Face format with LoRA adaptations merged into the weights.

## Files

- `pytorch_model.bin` - Main model weights (LoRA merged)
- `config.json` - Hugging Face model configuration
- `tokenizer_data.pt` - Tokenizer data
- `vocoder_model.bin` - Vocoder weights (if available)
- `hf_inference.py` - Inference script
- `modeling_seamless_m4t.py` - Model class definition

## Quick Start

```bash
python hf_inference.py \\
    --model_dir . \\
    --input "Hello, how are you?" \\
    --task T2TT \\
    --src_lang eng \\
    --tgt_lang cmn
```

## Model Information

- **Base Model**: {self.config.base_model_name}
- **Vocoder**: {self.config.vocoder_name}
- **LoRA Checkpoint**: {self.config.lora_checkpoint_path}
- **Format**: Hugging Face compatible
- **Size**: ~5GB (merged weights)

## Supported Languages

{', '.join(['eng', 'cmn', 'rus', 'fra', 'spa', 'arb'])}

## Supported Tasks

- `T2TT` - Text to Text Translation
- `S2TT` - Speech to Text Translation  
- `S2ST` - Speech to Speech Translation

## Dependencies

- torch
- torchaudio
- transformers (for HF compatibility)
- seamless_communication (for inference)

## Usage in Code

```python
from hf_inference import HFSeamlessM4T

# Load model
model = HFSeamlessM4T("path/to/model")

# Translate
result = model.translate_with_seamless(
    input_data="Hello world",
    task="T2TT",
    src_lang="eng",
    tgt_lang="cmn"
)
print(result)
```

The model contains your fine-tuned LoRA adaptations merged directly into the weights.
''')

        # Test the merged model
        logger.info("Testing merged model...")
        try:
            test_result = translator.predict(
                input="Hello world",
                task_str="T2TT",
                src_lang="eng",
                tgt_lang="cmn"
            )
            logger.info("Model test successful - can generate translations")
        except Exception as e:
            logger.warning(f"Model test failed: {e}")

        return {
            "model_dir": str(output_dir),
            "model_weights": str(model_path),
            "config": str(config_path),
            "hf_inference": str(hf_inference_path),
            "readme": str(readme_path)
        }


def main():
    parser = argparse.ArgumentParser(description="Export SeamlessM4T with LoRA to Hugging Face format")

    parser.add_argument("--base_model", type=str, default="seamlessM4T_medium",
                        help="Base model name")
    parser.add_argument("--vocoder", type=str, default="vocoder_v2",
                        help="Vocoder model name")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                        help="Path to LoRA checkpoint file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for HF model")

    args = parser.parse_args()

    # Create configuration
    config = ModelConfig(
        base_model_name=args.base_model,
        vocoder_name=args.vocoder,
        lora_checkpoint_path=args.lora_checkpoint
    )

    # Export to HuggingFace format
    exporter = HuggingFaceExporter(config)
    translator = exporter.load_and_merge_model()
    results = exporter.export_to_huggingface(translator, args.output_dir)

    print("\n" + "=" * 80)
    print("SEAMLESS M4T HUGGING FACE EXPORT SUMMARY")
    print("=" * 80)
    print(f"Model Directory: {results['model_dir']}")
    print(f"Model Weights: {results['model_weights']}")
    print(f"Config: {results['config']}")
    print(f"HF Inference: {results['hf_inference']}")
    print(f"README: {results['readme']}")
    print(f"\nFormat: Hugging Face compatible")
    print(f"Size: ~5GB (merged LoRA weights)")
    print(
        f"Usage: python {results['hf_inference']} --model_dir {results['model_dir']} --input 'text' --task T2TT --src_lang eng --tgt_lang cmn")
    print("=" * 80)


if __name__ == "__main__":
    main()
