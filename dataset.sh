#!/usr/bin/env bash
set -euo pipefail

export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HF_ENDPOINT=https://hf-mirror.com

source .venv/bin/activate

export DATASET_SAVE_DIR=/root/lanyun-tmp/data/fleurs
mkdir -p "$DATASET_SAVE_DIR"

echo "=== Enhanced Dataset Generation with Caching ==="

# Function to run with retries
run_with_retry() {
    local split=$1
    local max_retries=5
    local retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        echo "Processing $split dataset (attempt $((retry_count + 1))/$max_retries)"

        # Use the enhanced dataset builder
        python src/seamless_communication/cli/m4t/finetune/dataset_enhanced.py \
            --split "$split" \
            --save_dir "$DATASET_SAVE_DIR" \
            --merge \
            --overwrite \
            --use_cache

        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "000 Successfully processed $split dataset"
            break
        fi

        retry_count=$((retry_count + 1))
        echo "111 Failed! Exit code: $exit_code"
        echo "Retry: $retry_count / $max_retries"
        sleep 2
    done

    if [ $retry_count -eq $max_retries ]; then
        echo "222 Failed to process $split after $max_retries attempts"
        exit 1
    fi
}

# Process train set
echo "Processing training data..."
run_with_retry "train"

# Process validation set
echo "Processing validation data..."
run_with_retry "validation"

echo "111 Dataset generation completed!"
echo "Cache statistics:"
if [ -d "$DATASET_SAVE_DIR/cache" ]; then
    echo "  Cache directory size: $(du -sh $DATASET_SAVE_DIR/cache | cut -f1)"
    echo "  Audio cache files: $(find $DATASET_SAVE_DIR/cache/audio_cache -name "*.pt" | wc -l)"
    echo "  Text cache files: $(find $DATASET_SAVE_DIR/cache/text_cache -name "*.json" | wc -l)"
fi

echo "Generated manifests:"
ls -la "$DATASET_SAVE_DIR"/*.json
