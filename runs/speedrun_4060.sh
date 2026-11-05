#!/bin/bash

# Single-GPU speedrun for an 8GB card (RTX 4060): pretrain a small 6 layer model, then SFT.
# Unlike speedrun.sh this does not sync the venv, and only downloads missing shards.

# Run from anywhere with bash (not fish):
# 1) Check that the whole pipeline works, ~20 steps of each stage, tag gets a -smoke suffix:
# SMOKE_TEST=1 bash runs/speedrun_4060.sh
# 2) The real run:
# bash runs/speedrun_4060.sh
# 3) In a tmux session, with a custom tag and a smaller batch if you OOM (8, 4, 2 or 1):
# tmux new -s speedrun 'MODEL_TAG=d6-try2 DEVICE_BATCH_SIZE=1 bash runs/speedrun_4060.sh'
# 4) With wandb logging, run `wandb login` first:
# WANDB_RUN=d6-4060 bash runs/speedrun_4060.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export PYTHONUNBUFFERED=1
WANDB_RUN="${WANDB_RUN:-dummy}"
MODEL_TAG="${MODEL_TAG:-d6-4060-v2}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
TOTAL_BATCH_SIZE=16384
# chat_sft.py truncates SFT conversations longer than a row: about 44% of them at 512, under 1% at 2048.
MAX_SEQ_LEN=2048

# DEVICE_BATCH_SIZE x MAX_SEQ_LEN must divide TOTAL_BATCH_SIZE
case "$DEVICE_BATCH_SIZE" in
    1|2|4|8) ;;
    *)
        echo "ERROR: DEVICE_BATCH_SIZE must be 8, 4, 2 or 1, got: $DEVICE_BATCH_SIZE" >&2
        exit 1
        ;;
esac

if [ "${SMOKE_TEST:-0}" = "1" ]; then
    MODEL_TAG="$MODEL_TAG-smoke"
    BASE_ITERATIONS=20
    SFT_ITERATIONS=20
    EVAL_EVERY=-1
    EVAL_TOKENS=16384
    CHAT_EVAL_PROBLEMS=8
else
    BASE_ITERATIONS=5000
    SFT_ITERATIONS=1500
    EVAL_EVERY=250
    EVAL_TOKENS=524288
    CHAT_EVAL_PROBLEMS=200
fi

# -----------------------------------------------------------------------------
# Preflight checks

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: .venv not found, create it first with: uv sync --extra gpu" >&2
    exit 1
fi
source .venv/bin/activate

if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: PyTorch cannot see a CUDA GPU, try: uv sync --extra gpu" >&2
    exit 1
fi

# Checkpoints are named by step only, so reusing a tag would overwrite an earlier run.
# Smoke runs are disposable and may overwrite their own previous smoke run.
if [ "${SMOKE_TEST:-0}" != "1" ]; then
    for dir in "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG"; do
        if [ -e "$dir" ]; then
            echo "ERROR: $dir already exists, choose another tag: MODEL_TAG=... bash $0" >&2
            exit 1
        fi
    done
fi

# Keep the log outside of the repo, runs/*.log is not ignored by git
LOG_DIR="$NANOCHAT_BASE_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/speedrun_$MODEL_TAG.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Tag: $MODEL_TAG | device batch size: $DEVICE_BATCH_SIZE | log: $LOG_FILE"

# -----------------------------------------------------------------------------
# Data and tokenizer, skipped if already present

# 5000 steps x 16384 tokens only needs about 3 shards, but the tokenizer trains on ~2B characters = 8 shards
NUM_SHARDS=$(find "$NANOCHAT_BASE_DIR/base_data_climbmix" -name 'shard_*.parquet' 2>/dev/null | wc -l || true)
if [ "$NUM_SHARDS" -lt 9 ]; then
    python -m nanochat.dataset -n 8
fi

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    python -m scripts.tok_train
    python -m scripts.tok_eval
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Single GPU, so plain python instead of torchrun. Checkpoints are only saved at the end, ~540MB each.
python -m scripts.base_train \
    --device-type=cuda \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="$BASE_ITERATIONS" \
    --eval-every="$EVAL_EVERY" \
    --eval-tokens="$EVAL_TOKENS" \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN"

# Leaves out the "core" eval because it downloads eval_bundle, add it to --eval if you want the CORE metric.
python -m scripts.base_eval \
    --device-type=cuda \
    --eval=bpb,sample \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --split-tokens="$EVAL_TOKENS"

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# The low --init-lr-frac and the warmup: over 300 SFT steps at seq len 2048 (from a d6 base pretrained at 512),
# these reached val bpb 0.83, while the defaults (0.8, no warmup) first got worse and reached 0.90.
# --chatcore-every=-1 because ChatCORE needs the ARC and HumanEval datasets, which would be downloaded.
python -m scripts.chat_sft \
    --device-type=cuda \
    --model-tag="$MODEL_TAG" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="$SFT_ITERATIONS" \
    --init-lr-frac=0.1 \
    --warmup-ratio=0.05 \
    --eval-every="$EVAL_EVERY" \
    --eval-tokens="$EVAL_TOKENS" \
    --chatcore-every=-1 \
    --run="$WANDB_RUN"

# Only the tasks whose test data is cached. Remove -a to evaluate all tasks (downloads ARC and HumanEval).
python -m scripts.chat_eval \
    --device-type=cuda \
    -i sft \
    -g "$MODEL_TAG" \
    -a "MMLU|GSM8K" \
    -x "$CHAT_EVAL_PROBLEMS"

echo "Done. Chat with the model: python -m scripts.chat_cli -i sft -g $MODEL_TAG"
