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

max_retries=5
retry_count=0
while [ $retry_count -lt $max_retries ]; do
   python src/seamless_communication/cli/m4t/finetune/dataset_multi.py --split train --save_dir "$DATASET_SAVE_DIR" --merge --overwrite
   exit_code=$?
   if [ $exit_code -eq 0 ]; then
       break
   fi
   retry_count=$((retry_count + 1))
   echo "########## FAILED! Exit code: $exit_code ##########"
   echo "Retry: $retry_count / $max_retries"
   sleep 2
done

max_retries=5
retry_count=0
while [ $retry_count -lt $max_retries ]; do
   python src/seamless_communication/cli/m4t/finetune/dataset_multi.py --split validation --save_dir "$DATASET_SAVE_DIR" --merge --overwrite
   exit_code=$?
   if [ $exit_code -eq 0 ]; then
       break
   fi
   retry_count=$((retry_count + 1))
   echo "########## FAILED! Exit code: $exit_code ##########"
   echo "Retry: $retry_count / $max_retries"
   sleep 2
done
