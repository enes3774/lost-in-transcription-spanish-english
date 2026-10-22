#!/bin/bash
# Run the full ten-member ensemble on one data directory and write submission.csv.
#
#   bash run_inference.sh <data_dir> <out_csv> [work_dir] [gpu]
#
# <data_dir> must hold clips/ and test_metadata.csv (prepare_data.py builds it from a folder of audio).
# Needs one GPU with >= 48 GB (the competition runner was an A100 80 GB), the environment of
# environment/README.md, and the weights in <repo root>/models/ (download_weights.py); OMNI_MODELS_DIR overrides that folder.
# main.py and stage_omni.py read models/ next to themselves, so this script links src/inference/models -> the weights folder.
# Runtime for the full test set on an A100: about 35 min. The run is strict: if any member fails,
# no output file is written and the exit code is 1; the reason is the last "ABORT:" line in the log.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$(cd "$1" && pwd); OUT=$2; WORK=${3:-$HERE/work}; GPU=${4:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-TRITON_ATTN} VLLM_USE_FLASHINFER_SAMPLER=0
export OMNI_DATA_DIR="$DATA" OMNI_OUT_PATH="$OUT" OMNI_WORK_DIR="$WORK" OMNI_TIME_BUDGET_S=${OMNI_TIME_BUDGET_S:-7200}
MODELS=${OMNI_MODELS_DIR:-$(cd "$HERE/../.." && pwd)/models}
if [ ! -e "$HERE/models" ]; then ln -s "$MODELS" "$HERE/models"; fi
mkdir -p "$WORK"
cd "$HERE"
python -u main.py
