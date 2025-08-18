#!/usr/bin/env bash.
set -euo pipefail
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HF_ENDPOINT=https://hf-mirror.com

source .venv/bin/activate
pip uninstall -y seamless-communication
pip install .

export DATASET_SAVE_DIR=/root/lanyun-tmp/data/fleurs
mkdir -p "$DATASET_SAVE_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

torchrun \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:0 \
  --nnodes=1 \
  --nproc-per-node=1  \
  --no-python \
  m4t_finetune \
    --mode SPEECH_TO_TEXT \
    --train_dataset "$DATASET_SAVE_DIR/train_all_pairs_manifest.json" \
    --eval_dataset  "$DATASET_SAVE_DIR/validation_all_pairs_manifest.json" \
    --learning_rate 3e-5 \
    --warmup_steps 500 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --max_src_tokens 2000 \
    --eval_steps 200 \
    --max_epochs 5 \
    --patience 5 \
    --model_name seamlessM4T_medium \
    --save_model_to "$DATASET_SAVE_DIR/checkpoint_manual_lora.pt" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
  2>&1 | tee -a "$DATASET_SAVE_DIR/train_manual_lora_fixed.log"
