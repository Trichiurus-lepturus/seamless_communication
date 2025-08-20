#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# Enhanced M4T Fine-tuning Script with Progressive Training
# A complete 1-click setup for dataset generation and model training
#==============================================================================

# ASCII Art Header
cat << 'EOF'
================================================================================
                      Progressive Training Pipeline
================================================================================
EOF

# Environment and Proxy Configuration
echo "[INFO] Setting up environment..."
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HF_ENDPOINT=https://hf-mirror.com

# Enhanced memory and error handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

# Configuration Variables
DATASET_SAVE_DIR="/root/lanyun-tmp/data/fleurs"
VENV_PATH=".venv"
TRAIN_DATASET="$DATASET_SAVE_DIR/train_all_pairs_manifest.json"
EVAL_DATASET="$DATASET_SAVE_DIR/validation_all_pairs_manifest.json"
OUTPUT_DIR="$DATASET_SAVE_DIR/progressive_training"
LOG_DIR="$OUTPUT_DIR/logs"

# Training Hyperparameters
LEARNING_RATE="5e-5"
WARMUP_STEPS="200"
BATCH_SIZE="24"
GRAD_ACCUM_STEPS="1"
MAX_SRC_TOKENS="2000"
EVAL_STEPS="200"
PATIENCE="72"
LOG_STEPS="200"
MODEL_NAME="seamlessM4T_medium"
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"
CHECKPOINT_STEPS="200"

#==============================================================================
# Utility Functions
#==============================================================================

print_banner() {
    local message="$1"
    local length=${#message}
    local padding=$((80 - length))
    local left_pad=$((padding / 2))
    local right_pad=$((padding - left_pad))

    echo ""
    echo "================================================================================"
    printf "%*s%s%*s\n" $left_pad "" "$message" $right_pad ""
    echo "================================================================================"
}

print_status() {
    local status="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $status in
        "INFO")   echo "[$timestamp] [INFO]    $message" ;;
        "WARN")   echo "[$timestamp] [WARN]    $message" ;;
        "ERROR")  echo "[$timestamp] [ERROR]   $message" ;;
        "SUCCESS") echo "[$timestamp] [SUCCESS] $message" ;;
        *)        echo "[$timestamp] [$status] $message" ;;
    esac
}

validate_environment() {
    print_status "INFO" "Validating environment setup..."

    # Check virtual environment
    if [ ! -d "$VENV_PATH" ]; then
        print_status "ERROR" "Virtual environment $VENV_PATH not found!"
        return 1
    fi

    # Check CUDA availability
    source "$VENV_PATH/bin/activate"
    if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        local gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name())" 2>/dev/null)
        print_status "SUCCESS" "CUDA available: $gpu_name"
    else
        print_status "WARN" "CUDA not available, training may be very slow"
    fi

    return 0
}

check_dataset_exists() {
    if [ -f "$TRAIN_DATASET" ] && [ -f "$EVAL_DATASET" ]; then
        print_status "SUCCESS" "Datasets found"
        return 0
    else
        print_status "INFO" "Datasets not found, will generate them"
        return 1
    fi
}

#==============================================================================
# Dataset Generation Functions
#==============================================================================

# Update the generate_original_datasets function:
generate_original_datasets() {
    print_banner "DATASET GENERATION"
    print_status "INFO" "Generating CMN-ENG bilateral datasets..."

    mkdir -p "$DATASET_SAVE_DIR"
    source "$VENV_PATH/bin/activate"

    # Install package
    pip uninstall -y seamless-communication >/dev/null 2>&1 || true
    pip install . >/dev/null 2>&1

    local splits=("train" "validation")

    for split in "${splits[@]}"; do
        print_status "INFO" "Processing $split dataset (CMN-ENG only)..."

        local max_retries=3
        local retry_count=0
        local success=false

        while [ $retry_count -lt $max_retries ] && [ "$success" = false ]; do
            retry_count=$((retry_count + 1))
            print_status "INFO" "Attempt $retry_count/$max_retries for $split"

            # FIXED: Only generate CMN-ENG pairs
            if python src/seamless_communication/cli/m4t/finetune/dataset_enhanced.py \
                --langs eng cmn \
                --split "$split" \
                --save_dir "$DATASET_SAVE_DIR" \
                --merge \
                --overwrite \
                --use_cache \
                >/dev/null 2>&1; then

                print_status "SUCCESS" "$split dataset generated successfully"
                success=true
            else
                print_status "WARN" "$split dataset generation failed (attempt $retry_count)"
                sleep 2
            fi
        done

        if [ "$success" = false ]; then
            print_status "ERROR" "Failed to generate $split dataset after $max_retries attempts"
            return 1
        fi
    done

    print_status "SUCCESS" "CMN-ENG bilateral datasets generated"
    return 0
}

# COMPLETELY NEW: Generate augmented audio pool
generate_augmented_audio_pool() {
    print_banner "AUDIO AUGMENTATION POOL"
    print_status "INFO" "Creating comprehensive audio augmentation pool (3:1 ratio)..."

    source "$VENV_PATH/bin/activate"

    if [ ! -f "src/seamless_communication/cli/m4t/finetune/preprocess_augmentation.py" ]; then
        print_status "WARN" "preprocess_augmentation.py not found, skipping augmentation"
        return 1
    fi

    # Create ONE comprehensive augmented dataset with multiple variants
    local output_manifest="$DATASET_SAVE_DIR/train_audio_augmented_pool.json"
    local output_audio_dir="$DATASET_SAVE_DIR/augmented_audio_pool"

    print_status "INFO" "Generating comprehensive audio augmentation pool..."

    if python src/seamless_communication/cli/m4t/finetune/preprocess_augmentation.py \
        --input_manifest "$TRAIN_DATASET" \
        --output_manifest "$output_manifest" \
        --output_audio_dir "$output_audio_dir" \
        --stage "audio_comprehensive" \
        --augmentation_ratio 0.25 \
        --num_workers 4 \
        --seed 42 \
        --create_variants 3 \
        >/dev/null 2>&1; then

        print_status "SUCCESS" "Audio augmentation pool created (3:1 ratio with variants)"
    else
        print_status "WARN" "Failed to create audio augmentation pool"
        return 1
    fi

    return 0
}

# NEW: Create stage-specific datasets from the augmented pool
create_stage_datasets() {
    print_banner "STAGE-SPECIFIC DATASETS"
    print_status "INFO" "Creating stage-specific datasets from augmentation pool..."

    source "$VENV_PATH/bin/activate"

    local base_manifest="$DATASET_SAVE_DIR/train_audio_augmented_pool.json"

    if [ ! -f "$base_manifest" ]; then
        print_status "WARN" "Augmented pool not found, using original dataset"
        return 0
    fi

    # Create stage-specific datasets with 3:1 ratio
    local stages=("speech_encoder" "text_decoder" "full")
    local variant_seeds=(100 200 300)  # Different seeds for different variants

    for i in "${!stages[@]}"; do
        local stage="${stages[$i]}"
        local seed="${variant_seeds[$i]}"
        local output_manifest="$DATASET_SAVE_DIR/train_${stage}_dataset.json"

        print_status "INFO" "Creating dataset for $stage stage (3:1 ratio, seed: $seed)..."

        if python src/seamless_communication/cli/m4t/finetune/create_stage_dataset.py \
            --input_manifest "$base_manifest" \
            --output_manifest "$output_manifest" \
            --stage "$stage" \
            --ratio 3.0 \
            --seed "$seed" \
            >/dev/null 2>&1; then

            print_status "SUCCESS" "$stage dataset created"
        else
            print_status "WARN" "Failed to create $stage dataset"
        fi
    done

    return 0
}

# Update the generate_augmented_datasets function:
generate_augmented_datasets() {
    # Generate the comprehensive audio pool
    generate_augmented_audio_pool

    # Create stage-specific datasets from the pool
    create_stage_datasets

    return 0
}

# Update select_training_dataset function:
select_training_dataset() {
    local stage="$1"
    local base_dataset="$TRAIN_DATASET"
    local dataset_dir=$(dirname "$base_dataset")
    local selected_dataset="$base_dataset"

    case $stage in
        "speech_encoder")
            # Look for stage-specific dataset (from your new augmentation strategy)
            local stage_dataset="$dataset_dir/train_speech_encoder_dataset.json"
            if [ -f "$stage_dataset" ]; then
                selected_dataset="$stage_dataset"
                print_status "INFO" "Using speech_encoder dataset (3:1 ratio, variant A)"
            else
                print_status "WARN" "Stage-specific dataset not found, using original"
            fi
            ;;
        "text_decoder")
            local stage_dataset="$dataset_dir/train_text_decoder_dataset.json"
            if [ -f "$stage_dataset" ]; then
                selected_dataset="$stage_dataset"
                print_status "INFO" "Using text_decoder dataset (3:1 ratio, variant B)"
            else
                print_status "WARN" "Stage-specific dataset not found, using original"
            fi
            ;;
        "full")
            local stage_dataset="$dataset_dir/train_full_dataset.json"
            if [ -f "$stage_dataset" ]; then
                selected_dataset="$stage_dataset"
                print_status "INFO" "Using full stage dataset (3:1 ratio, variant C)"
            else
                print_status "WARN" "Stage-specific dataset not found, using original"
            fi
            ;;
        *)
            print_status "INFO" "Using original dataset for $stage stage"
            ;;
    esac

    echo "$selected_dataset"
}

run_training_stage() {
    local stage="$1"
    local epochs="$2"
    local log_suffix="$3"

    print_banner "TRAINING STAGE: $(echo $stage | tr '[:lower:]' '[:upper:]')"
    print_status "INFO" "Stage: $stage | Epochs: $epochs | Log: $log_suffix"

    # Select appropriate dataset
    local selected_dataset=$(select_training_dataset "$stage")
    local log_file="$LOG_DIR/train_${stage}_${log_suffix}.log"

    print_status "INFO" "Training dataset: $(basename "$selected_dataset")"
    print_status "INFO" "Log file: $log_file"

    # Ensure directories exist
    mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

    # Validate datasets
    if [ ! -f "$selected_dataset" ]; then
        print_status "ERROR" "Training dataset not found: $selected_dataset"
        return 1
    fi

    if [ ! -f "$EVAL_DATASET" ]; then
        print_status "ERROR" "Evaluation dataset not found: $EVAL_DATASET"
        return 1
    fi

    # Source environment
    source "$VENV_PATH/bin/activate"

    # Build torchrun command
    local save_model_path="$OUTPUT_DIR/checkpoint_${stage}.pt"

    print_status "INFO" "Starting torchrun for stage: $stage"

    torchrun \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        --nnodes=1 \
        --nproc-per-node=1 \
        --no-python \
        m4t_finetune \
            --mode SPEECH_TO_TEXT \
            --train_dataset "$selected_dataset" \
            --eval_dataset "$EVAL_DATASET" \
            --learning_rate "$LEARNING_RATE" \
            --warmup_steps "$WARMUP_STEPS" \
            --batch_size "$BATCH_SIZE" \
            --grad_accum_steps "$GRAD_ACCUM_STEPS" \
            --max_src_tokens "$MAX_SRC_TOKENS" \
            --eval_steps "$EVAL_STEPS" \
            --max_epochs "$epochs" \
            --patience "$PATIENCE" \
            --log_steps "$LOG_STEPS" \
            --model_name "$MODEL_NAME" \
            --save_model_to "$save_model_path" \
            --lora_r "$LORA_R" \
            --lora_alpha "$LORA_ALPHA" \
            --lora_dropout "$LORA_DROPOUT" \
            --training_stage "$stage" \
            --checkpoint_steps "$CHECKPOINT_STEPS" \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}

    case $exit_code in
        0)
            print_status "SUCCESS" "Stage $stage completed successfully"
            return 0
            ;;
        130)
            print_status "WARN" "Stage $stage interrupted by user"
            return 130
            ;;
        *)
            print_status "ERROR" "Stage $stage failed with exit code $exit_code"
            return $exit_code
            ;;
    esac
}

run_progressive_training() {
    print_banner "PROGRESSIVE TRAINING PIPELINE"
    print_status "INFO" "Starting full progressive training pipeline"

    local save_model_path="$OUTPUT_DIR/checkpoint_progressive.pt"
    local log_file="$LOG_DIR/train_progressive_full.log"

    mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
    source "$VENV_PATH/bin/activate"

    print_status "INFO" "Log file: $log_file"

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
            --learning_rate "$LEARNING_RATE" \
            --warmup_steps "$WARMUP_STEPS" \
            --batch_size "$BATCH_SIZE" \
            --grad_accum_steps "$GRAD_ACCUM_STEPS" \
            --max_src_tokens "$MAX_SRC_TOKENS" \
            --eval_steps "$EVAL_STEPS" \
            --patience "$PATIENCE" \
            --log_steps "$LOG_STEPS" \
            --model_name "$MODEL_NAME" \
            --save_model_to "$save_model_path" \
            --lora_r "$LORA_R" \
            --lora_alpha "$LORA_ALPHA" \
            --lora_dropout "$LORA_DROPOUT" \
            --progressive \
            --stage_epochs 3 3 4 \
            --checkpoint_steps "$CHECKPOINT_STEPS" \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}

    case $exit_code in
        0)
            print_status "SUCCESS" "Progressive training completed successfully"
            return 0
            ;;
        130)
            print_status "WARN" "Progressive training interrupted by user"
            return 130
            ;;
        *)
            print_status "ERROR" "Progressive training failed with exit code $exit_code"
            return $exit_code
            ;;
    esac
}

resume_training() {
    local stage="$1"
    print_banner "RESUMING TRAINING: $(echo $stage | tr '[:lower:]' '[:upper:]')"

    local checkpoint_dir="$OUTPUT_DIR/training_checkpoints"

    if [ ! -d "$checkpoint_dir" ]; then
        print_status "ERROR" "No checkpoint directory found: $checkpoint_dir"
        return 1
    fi

    source "$VENV_PATH/bin/activate"

    # Find latest checkpoint for stage
    local latest_checkpoint=""
    if [ -f "$checkpoint_dir/latest_checkpoint.json" ]; then
        latest_checkpoint=$(python -c "
import json
try:
    with open('$checkpoint_dir/latest_checkpoint.json', 'r') as f:
        data = json.load(f)
    if data.get('training_stage') == '$stage':
        print(data.get('latest_checkpoint', ''))
except:
    pass
" 2>/dev/null || echo "")
    fi

    local resume_args=""
    if [ -n "$latest_checkpoint" ] && [ -f "$latest_checkpoint" ]; then
        resume_args="--resume_from_checkpoint $latest_checkpoint"
        print_status "INFO" "Resuming from checkpoint: $(basename "$latest_checkpoint")"
    else
        print_status "WARN" "No valid checkpoint found for stage $stage, starting fresh"
    fi

    local selected_dataset=$(select_training_dataset "$stage")
    local save_model_path="$OUTPUT_DIR/checkpoint_${stage}_resumed.pt"
    local log_file="$LOG_DIR/train_${stage}_resumed.log"

    mkdir -p "$LOG_DIR"

    torchrun \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        --nnodes=1 \
        --nproc-per-node=1 \
        --no-python \
        m4t_finetune \
            --mode SPEECH_TO_TEXT \
            --train_dataset "$selected_dataset" \
            --eval_dataset "$EVAL_DATASET" \
            --learning_rate "$LEARNING_RATE" \
            --warmup_steps "$WARMUP_STEPS" \
            --batch_size "$BATCH_SIZE" \
            --grad_accum_steps "$GRAD_ACCUM_STEPS" \
            --max_src_tokens "$MAX_SRC_TOKENS" \
            --eval_steps "$EVAL_STEPS" \
            --max_epochs 10 \
            --patience "$PATIENCE" \
            --log_steps "$LOG_STEPS" \
            --model_name "$MODEL_NAME" \
            --save_model_to "$save_model_path" \
            --lora_r "$LORA_R" \
            --lora_alpha "$LORA_ALPHA" \
            --lora_dropout "$LORA_DROPOUT" \
            --training_stage "$stage" \
            --checkpoint_steps "$CHECKPOINT_STEPS" \
            $resume_args \
        2>&1 | tee "$log_file"

    return ${PIPESTATUS[0]}
}

#==============================================================================
# Reporting Functions
#==============================================================================

generate_training_summary() {
    local exit_code="$1"
    local training_mode="$2"

    print_banner "TRAINING SUMMARY"

    echo "Training Mode:      $training_mode"
    echo "Exit Code:          $exit_code"
    echo "Output Directory:   $OUTPUT_DIR"
    echo "Log Directory:      $LOG_DIR"
    echo "Timestamp:          $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    case $exit_code in
        0)
            print_status "SUCCESS" "Training completed successfully"
            echo ""
            echo "Generated Model Files:"
            if ls "$OUTPUT_DIR"/*.pt >/dev/null 2>&1; then
                ls -la "$OUTPUT_DIR"/*.pt
            else
                echo "  No model files found"
            fi
            ;;
        130)
            print_status "WARN" "Training was interrupted"
            echo ""
            echo "Recovery Options:"
            echo "  - Resume with: $0 resume <stage>"
            echo "  - Check logs in: $LOG_DIR"
            ;;
        *)
            print_status "ERROR" "Training failed"
            echo ""
            echo "Troubleshooting:"
            echo "  - Check logs in: $LOG_DIR"
            echo "  - Verify dataset integrity"
            echo "  - Check GPU memory usage"
            ;;
    esac

    echo ""
    echo "Log Files:"
    if ls "$LOG_DIR"/*.log >/dev/null 2>&1; then
        ls -la "$LOG_DIR"/*.log
    else
        echo "  No log files found"
    fi

    echo ""
    if [ -d "$DATASET_SAVE_DIR/cache" ]; then
        echo "Cache Statistics:"
        echo "  Cache directory size: $(du -sh "$DATASET_SAVE_DIR/cache" 2>/dev/null | cut -f1 || echo 'N/A')"
        echo "  Audio cache files: $(find "$DATASET_SAVE_DIR/cache/audio_cache" -name "*.pt" 2>/dev/null | wc -l || echo '0')"
        echo "  Text cache files: $(find "$DATASET_SAVE_DIR/cache/text_cache" -name "*.json" 2>/dev/null | wc -l || echo '0')"
    fi

    echo ""
    echo "================================================================================"
}

print_usage() {
    cat << 'EOF'
Usage: ./training.sh [MODE] [STAGE]

MODES:
    progressive     Run full progressive training pipeline (default)
    single         Run single stage training
    stages         Run individual stages sequentially
    resume         Resume interrupted training
    test           Test conservative mode

STAGES (for single/resume modes):
    speech_encoder  Train speech encoder with audio augmentation
    text_decoder    Train text decoder with text augmentation
    full           Train both components with light augmentation
    conservative   Original training behavior

EXAMPLES:
    ./training.sh                           # Progressive training (default)
    ./training.sh progressive               # Same as above
    ./training.sh single speech_encoder     # Train only speech encoder
    ./training.sh resume text_decoder       # Resume text decoder training
    ./training.sh test                      # Quick test run

EOF
}

#==============================================================================
# Main Execution
#==============================================================================

setup_environment() {
    print_banner "ENVIRONMENT SETUP"

    # Validate environment
    if ! validate_environment; then
        print_status "ERROR" "Environment validation failed"
        return 1
    fi

    # Setup datasets
    if ! check_dataset_exists; then
        if ! generate_original_datasets; then
            print_status "ERROR" "Dataset generation failed"
            return 1
        fi
    fi

    # Generate augmented datasets (if possible)
    generate_augmented_datasets

    return 0
}

main() {
    # Handle help/usage
    if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
        print_usage
        exit 0
    fi

    # Setup trap for interruption handling
    trap 'echo ""; print_status "WARN" "Training interrupted. Checkpoints saved."; exit 130' INT TERM

    # Parse arguments
    local training_mode="${1:-progressive}"
    local specific_stage="${2:-}"

    # Setup environment and datasets
    if ! setup_environment; then
        print_status "ERROR" "Setup failed"
        exit 1
    fi

    local exit_code=0

    # Execute training based on mode
    case $training_mode in
        "progressive")
            print_status "INFO" "Running full progressive training pipeline"
            run_progressive_training
            exit_code=$?
            ;;

        "single")
            if [ -z "$specific_stage" ]; then
                print_status "ERROR" "Specific stage required for single mode"
                print_usage
                exit 1
            fi
            print_status "INFO" "Running single stage training: $specific_stage"
            run_training_stage "$specific_stage" 10 "single"
            exit_code=$?
            ;;

        "stages")
            print_status "INFO" "Running individual stages sequentially"
            local stages=("speech_encoder" "text_decoder" "full")
            local epochs=(3 3 4)

            for i in "${!stages[@]}"; do
                local stage="${stages[$i]}"
                local epoch_count="${epochs[$i]}"

                if ! run_training_stage "$stage" "$epoch_count" "individual"; then
                    print_status "ERROR" "$stage stage failed, stopping pipeline"
                    exit_code=1
                    break
                fi
            done

            if [ $exit_code -eq 0 ]; then
                print_status "SUCCESS" "All individual stages completed successfully"
            fi
            ;;

        "resume")
            if [ -z "$specific_stage" ]; then
                print_status "ERROR" "Specific stage required for resume mode"
                print_usage
                exit 1
            fi
            print_status "INFO" "Attempting to resume training for stage: $specific_stage"
            resume_training "$specific_stage"
            exit_code=$?
            ;;

        "test")
            print_status "INFO" "Running conservative mode test"
            run_training_stage "conservative" 2 "test"
            exit_code=$?
            ;;

        *)
            print_status "ERROR" "Unknown training mode: $training_mode"
            print_usage
            exit 1
            ;;
    esac

    # Generate summary report
    generate_training_summary "$exit_code" "$training_mode"

    exit $exit_code
}

# Execute main function with all arguments
main "$@"
