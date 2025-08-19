#!/usr/bin/env bash
set -euo pipefail

# Proxy settings
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HF_ENDPOINT=https://hf-mirror.com

# CRITICAL FIX: Validate environment basics
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment .venv not found!"
    exit 1
fi

# Activate environment and install
source .venv/bin/activate
pip uninstall -y seamless-communication
pip install .

# Dataset configuration
export DATASET_SAVE_DIR=/root/lanyun-tmp/data/fleurs
mkdir -p "$DATASET_SAVE_DIR"

# Enhanced memory and error handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
# CRITICAL FIX: Add tokenizer parallelism control
export TOKENIZERS_PARALLELISM=false

# Training configuration
TRAIN_DATASET="$DATASET_SAVE_DIR/train_all_pairs_manifest.json"
EVAL_DATASET="$DATASET_SAVE_DIR/validation_all_pairs_manifest.json"
OUTPUT_DIR="$DATASET_SAVE_DIR/progressive_training"
mkdir -p "$OUTPUT_DIR"

# CRITICAL FIX: Validate datasets exist
if [ ! -f "$TRAIN_DATASET" ]; then
    echo "❌ Training dataset not found: $TRAIN_DATASET"
    exit 1
fi

if [ ! -f "$EVAL_DATASET" ]; then
    echo "❌ Evaluation dataset not found: $EVAL_DATASET"
    exit 1
fi

# Check if augmented dataloader is available
DATALOADER_CHECK=$(python -c "
try:
    from dataloader_augmented import create_compatible_dataloader
    print('augmented')
except ImportError:
    print('standard')
" 2>/dev/null || echo 'standard')

echo "Dataloader type: $DATALOADER_CHECK"

# Function to run single stage training
run_single_stage() {
    local stage=$1
    local epochs=$2
    local log_suffix=$3

    echo "========================================="
    echo "Starting training stage: $stage"
    echo "Epochs: $epochs"
    echo "========================================="

    torchrun \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        --nnodes=1 \
        --nproc-per-node=1 \
        --no-python \
        m4t_finetune \
            --mode SPEECH_TO_TEXT \
            --train_dataset "$TRAIN_DATASET" \
            --eval_dataset "$EVAL_DATASET" \
            --learning_rate 5e-5 \
            --warmup_steps 200 \
            --batch_size 24 \
            --grad_accum_steps 1 \
            --max_src_tokens 2000 \
            --eval_steps 200 \
            --max_epochs $epochs \
            --patience 72 \
            --log_steps 200 \
            --model_name seamlessM4T_medium \
            --save_model_to "$OUTPUT_DIR/checkpoint_${stage}.pt" \
            --lora_r 8 \
            --lora_alpha 16 \
            --lora_dropout 0.05 \
            --training_stage $stage \
            --checkpoint_steps 200 \
        2>&1 | tee -a "$OUTPUT_DIR/train_${stage}_${log_suffix}.log"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "✅ Stage $stage completed successfully"
        return 0
    elif [ $exit_code -eq 130 ]; then
        echo "⚠️  Stage $stage interrupted by user"
        return 130
    else
        echo "❌ Stage $stage failed with exit code $exit_code"
        return $exit_code
    fi
}

# Function to run progressive training
run_progressive_training() {
    echo "========================================="
    echo "Starting PROGRESSIVE TRAINING"
    echo "========================================="

    torchrun \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        --nnodes=1 \
        --nproc-per-node=1 \
        --no-python \
        m4t_finetune \
            --mode SPEECH_TO_TEXT \
            --train_dataset "$TRAIN_DATASET" \
            --eval_dataset "$EVAL_DATASET" \
            --learning_rate 5e-5 \
            --warmup_steps 200 \
            --batch_size 24 \
            --grad_accum_steps 1 \
            --max_src_tokens 2000 \
            --eval_steps 200 \
            --patience 72 \
            --log_steps 200 \
            --model_name seamlessM4T_medium \
            --save_model_to "$OUTPUT_DIR/checkpoint_progressive.pt" \
            --lora_r 8 \
            --lora_alpha 16 \
            --lora_dropout 0.05 \
            --progressive \
            --stage_epochs 3 3 4 \
            --checkpoint_steps 200 \
        2>&1 | tee -a "$OUTPUT_DIR/train_progressive_full.log"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "✅ Progressive training completed successfully"
        return 0
    elif [ $exit_code -eq 130 ]; then
        echo "⚠️  Progressive training interrupted by user"
        return 130
    else
        echo "❌ Progressive training failed with exit code $exit_code"
        return $exit_code
    fi
}

# CRITICAL FIX: Function to resume training if interrupted
resume_training() {
    local stage=$1
    local checkpoint_dir="$OUTPUT_DIR/training_checkpoints"

    if [ -d "$checkpoint_dir" ]; then
        echo "Found checkpoint directory, attempting to resume..."

        # CRITICAL FIX: Find the latest checkpoint for the stage
        local latest_checkpoint=""
        if [ -f "$checkpoint_dir/latest_checkpoint.json" ]; then
            latest_checkpoint=$(python -c "
import json, sys
try:
    with open('$checkpoint_dir/latest_checkpoint.json', 'r') as f:
        data = json.load(f)
    if data.get('training_stage') == '$stage':
        print(data.get('latest_checkpoint', ''))
except:
    pass
")
        fi

        # Build resume command
        local resume_args=""
        if [ -n "$latest_checkpoint" ] && [ -f "$latest_checkpoint" ]; then
            resume_args="--resume_from_checkpoint $latest_checkpoint"
            echo "Resuming from checkpoint: $latest_checkpoint"
        else
            echo "⚠️  No valid checkpoint found for stage $stage, starting fresh"
        fi

        torchrun \
            --rdzv-backend=c10d \
            --rdzv-endpoint=localhost:0 \
            --nnodes=1 \
            --nproc-per-node=1 \
            --no-python \
            m4t_finetune \
                --mode SPEECH_TO_TEXT \
                --train_dataset "$TRAIN_DATASET" \
                --eval_dataset "$EVAL_DATASET" \
                --learning_rate 5e-5 \
                --warmup_steps 200 \
                --batch_size 24 \
                --grad_accum_steps 1 \
                --max_src_tokens 2000 \
                --eval_steps 200 \
                --max_epochs 10 \
                --patience 72 \
                --log_steps 200 \
                --model_name seamlessM4T_medium \
                --save_model_to "$OUTPUT_DIR/checkpoint_${stage}_resumed.pt" \
                --lora_r 8 \
                --lora_alpha 16 \
                --lora_dropout 0.05 \
                --training_stage $stage \
                --checkpoint_steps 200 \
                $resume_args \
            2>&1 | tee -a "$OUTPUT_DIR/train_${stage}_resumed.log"
    else
        echo "No checkpoint directory found"
        return 1
    fi
}

# Main training logic
main() {
    echo "Starting enhanced M4T fine-tuning with progressive training support"
    echo "Output directory: $OUTPUT_DIR"
    echo "Augmented dataloader: $DATALOADER_CHECK"

    # CRITICAL FIX: Basic CUDA validation
    if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
        echo "⚠️  CUDA not available, training may be very slow"
    else
        echo "✅ CUDA available: $(python -c "import torch; print(torch.cuda.get_device_name())")"
    fi

    # Parse command line arguments
    TRAINING_MODE="${1:-progressive}"  # Default to progressive
    SPECIFIC_STAGE="${2:-}"

    case $TRAINING_MODE in
        "progressive")
            echo "Running full progressive training pipeline..."
            run_progressive_training
            ;;
        "single")
            if [ -z "$SPECIFIC_STAGE" ]; then
                echo "Error: Specific stage required for single mode"
                echo "Usage: $0 single [speech_encoder|text_decoder|full|conservative]"
                exit 1
            fi
            echo "Running single stage training: $SPECIFIC_STAGE"
            run_single_stage "$SPECIFIC_STAGE" 10 "single"
            ;;
        "stages")
            echo "Running individual stages sequentially..."
            # Speech encoder stage
            if ! run_single_stage "speech_encoder" 3 "individual"; then
                echo "Speech encoder stage failed, stopping"
                exit 1
            fi

            # Text decoder stage
            if ! run_single_stage "text_decoder" 3 "individual"; then
                echo "Text decoder stage failed, stopping"
                exit 1
            fi

            # Full model stage
            if ! run_single_stage "full" 4 "individual"; then
                echo "Full model stage failed, stopping"
                exit 1
            fi

            echo "✅ All individual stages completed successfully"
            ;;
        "resume")
            if [ -z "$SPECIFIC_STAGE" ]; then
                echo "Error: Specific stage required for resume mode"
                echo "Usage: $0 resume [speech_encoder|text_decoder|full]"
                exit 1
            fi
            echo "Attempting to resume training for stage: $SPECIFIC_STAGE"
            resume_training "$SPECIFIC_STAGE"
            ;;
        "test")
            echo "Running conservative mode test (original behavior)..."
            run_single_stage "conservative" 2 "test"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes:"
            echo "  progressive  - Run full progressive pipeline (default)"
            echo "  single <stage> - Run single stage"
            echo "  stages       - Run individual stages sequentially"
            echo "  resume <stage> - Resume interrupted training"
            echo "  test         - Test conservative mode"
            exit 1
            ;;
    esac

    local exit_code=$?

    # Generate summary
    echo "========================================="
    echo "TRAINING SUMMARY"
    echo "========================================="
    echo "Training mode: $TRAINING_MODE"
    echo "Output directory: $OUTPUT_DIR"
    echo "Exit code: $exit_code"

    if [ $exit_code -eq 0 ]; then
        echo "Status: ✅ SUCCESS"
        echo ""
        echo "Generated files:"
        ls -la "$OUTPUT_DIR"/*.pt 2>/dev/null || echo "No model files found"
        echo ""
        echo "Log files:"
        ls -la "$OUTPUT_DIR"/*.log 2>/dev/null || echo "No log files found"
    elif [ $exit_code -eq 130 ]; then
        echo "Status: ⚠️  INTERRUPTED"
        echo "You can resume with: $0 resume <stage>"
    else
        echo "Status: ❌ FAILED"
        echo "Check logs in: $OUTPUT_DIR"
    fi

    exit $exit_code
}

# Handle interruption gracefully
trap 'echo "Training interrupted. Checkpoints saved."; exit 130' INT TERM

# Run main function with all arguments
main "$@"
