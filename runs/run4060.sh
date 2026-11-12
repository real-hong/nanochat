#!/bin/bash

# Full nanochat pipeline (tokenizer -> pretrain -> eval -> SFT -> chat eval) tuned to
# saturate a single 8GB GPU (RTX 4060). The GPU sits at 100% utilization throughout.
#
# Run as:
#   bash runs/run4060.sh                 # d8:  ~2h pretraining + ~2h SFT
#   DEPTH=10 bash runs/run4060.sh        # d10: ~5.5h pretraining + ~3h SFT
#   DEPTH=12 bash runs/run4060.sh        # d12: ~14h pretraining + ~5h SFT
# Long runs are best started inside tmux:  tmux new -s nanochat 'bash runs/run4060.sh'
#
# The script is resumable: re-run the same command after a crash / Ctrl+C / reboot and it
# continues pretraining from the last checkpoint, and skips stages that already finished.
# (SFT has no mid-run checkpoints, an interrupted SFT stage starts over.)
#
# Measured on RTX 4060 (seq len 2048, bf16, Flash Attention 3, ~1.5GB VRAM used by the desktop):
#   depth | max device batch | tok/sec | peak VRAM
#     8   |        8         |  ~70K   |  4.8GB
#    10   |        4         |  ~42K   |  4.5GB
#    12   |        2         |  ~27K   |  4.6GB
# VRAM is dominated by the logits (batch x seq x vocab), so closing the browser or driving the
# monitor from the integrated GPU frees room; throughput however is compute-bound, not VRAM-bound.
#
# Knobs (environment variables):
#   DEPTH              model depth (default 8)
#   TAG                checkpoint/log name (default d$DEPTH, the repo convention)
#   DEVICE_BATCH_SIZE  override the auto-picked batch size (power of 2)
#   NUM_ITERATIONS     override pretraining steps (default: compute-optimal for the depth)
#   SFT_ITERATIONS     debug only: cap SFT at this many micro-batches (default: one full epoch, ~480M tokens)
#   CORE_MAX_PER_TASK  examples per task for the CORE eval (default 500)
#   WANDB_RUN          wandb run name (default "dummy" = no wandb)
#   KEEP_CKPTS=1       keep intermediate pretraining checkpoints (default: only the final one)
#   NO_TUNED=1         do not touch the CPU power profile

set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

DEPTH=${DEPTH:-8}
TAG=${TAG:-d${DEPTH}}
WANDB_RUN=${WANDB_RUN:-dummy}
CORE_MAX_PER_TASK=${CORE_MAX_PER_TASK:-500}
EVAL_TOKENS=$((4 * 524288)) # ~2M tokens per val eval: a few seconds instead of minutes

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1 # otherwise progress reaches the terminal/log in 8KB chunks through the pipe
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
BASE_CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$TAG"
SFT_CKPT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG"
LOG="$NANOCHAT_BASE_DIR/logs/run4060_$TAG.log"
mkdir -p "$NANOCHAT_BASE_DIR/logs"

# One run per tag at a time: two trainers writing the same checkpoint dir would corrupt it.
# The lock is inherited by all children, so it also holds while a stray training process is alive.
exec 9> "$LOG.lock"
flock -n 9 || {
    echo "A run for TAG=$TAG is already in progress (lock: $LOG.lock)."
    echo "If you killed the script itself, its training process may still be alive: check nvidia-smi."
    exit 1
}

# Keep the machine awake while this script is alive. Best effort: without permission
# (e.g. over ssh) the inhibitor just fails and training proceeds anyway.
if command -v systemd-inhibit &> /dev/null; then
    systemd-inhibit --no-ask-password --what=sleep:idle --who=nanochat --why="nanochat training" \
        tail --pid=$$ -f /dev/null &> /dev/null 9>&- &
fi

# -----------------------------------------------------------------------------
# helpers

stage() { echo; echo "=== [$(date '+%F %T')] $* ==="; }

# CPU power profile: the data loader is a single Python thread, give it a full-speed core.
restore_profile() { [ -n "${OLD_PROFILE:-}" ] && tuned-adm profile "$OLD_PROFILE" &> /dev/null; }
set_performance_profile() {
    [ -n "${NO_TUNED:-}" ] && return
    command -v tuned-adm &> /dev/null || return
    OLD_PROFILE=$(tuned-adm active 2> /dev/null | sed -n 's/^Current active profile: //p')
    [ -z "$OLD_PROFILE" ] || [ "$OLD_PROFILE" = "throughput-performance" ] && { OLD_PROFILE=; return; }
    if timeout 20 tuned-adm profile throughput-performance &> /dev/null; then
        echo "CPU power profile: $OLD_PROFILE -> throughput-performance (restored on exit)"
        trap restore_profile EXIT
    else
        OLD_PROFILE=
    fi
}

# Largest power-of-2 batch size that fits in the VRAM that is free right now.
# Fit to the peak allocations measured above: a fixed cost (weights + optimizer state, ~d^2)
# plus a per-row cost (activations + logits, ~d), plus ~500MiB of CUDA context / headroom.
pick_batch_size() {
    local b=32 need free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [[ "$free" =~ ^[0-9]+$ ]] || { echo "could not read free VRAM from nvidia-smi, set DEVICE_BATCH_SIZE" >&2; return 1; }
    while [ "$b" -gt 1 ]; do
        need=$(( 227 * DEPTH * DEPTH / 10 + 350 + (82 * DEPTH - 224) * b ))
        [ "$free" -ge "$need" ] && break
        b=$((b / 2))
    done
    echo "$b"
}

# Latest pretraining step that has model + optimizer + meta files, or nothing
last_resumable_step() {
    local files i step
    files=("$BASE_CKPT_DIR"/model_*.pt) # glob order is ascending (zero-padded step numbers)
    for (( i = ${#files[@]} - 1; i >= 0; i-- )); do
        [ -f "${files[i]}" ] || continue # unmatched glob stays literal
        step=$(basename "${files[i]}" .pt); step=${step#model_}
        if [ -f "$BASE_CKPT_DIR/meta_$step.json" ] && [ -f "$BASE_CKPT_DIR/optim_${step}_rank0.pt" ]; then
            echo "$((10#$step))"; return
        fi
    done
}

main() {
    echo "Tag: $TAG | depth: $DEPTH | log: $LOG"

    # -------------------------------------------------------------------------
    stage "Environment"
    command -v uv &> /dev/null || { echo "uv not found: https://docs.astral.sh/uv/"; return 1; }
    uv sync --extra gpu --frozen || return 1 # --frozen: install from uv.lock without rewriting it
    source .venv/bin/activate
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available (driver mismatch? try a reboot)'; print('GPU:', torch.cuda.get_device_name(0))" || return 1
    set_performance_profile
    B=${DEVICE_BATCH_SIZE:-$(pick_batch_size)} || return 1
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/VRAM in use before training: /'
    EVAL_B=$(( B > 1 ? B / 2 : 1 )) # standalone evals run uncompiled and need more VRAM per row
    echo "Device batch size: $B"

    # -------------------------------------------------------------------------
    stage "Data + tokenizer"
    # ~40M usable training tokens per shard; compute-optimal horizons are ~0.5B (d8), ~0.85B (d10), ~1.3B (d12)
    if   [ "$DEPTH" -le 8 ];  then shards=20
    elif [ "$DEPTH" -le 10 ]; then shards=32
    elif [ "$DEPTH" -le 12 ]; then shards=48
    else shards=170; fi
    python -m nanochat.dataset -n ${N_SHARDS:-$shards} | grep -v "already exists" || return 1
    if [ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
        echo "Tokenizer already trained, skipping"
    else
        python -m scripts.tok_train || return 1
        python -m scripts.tok_eval
    fi

    # -------------------------------------------------------------------------
    stage "Pretraining"
    # Only ever reuse checkpoints that this script created with the same settings: resuming someone
    # else's run (or with a different depth / horizon) silently trains or evaluates the wrong thing.
    config="depth=$DEPTH num_iterations=${NUM_ITERATIONS:-auto}"
    if ls "$BASE_CKPT_DIR"/model_*.pt &> /dev/null; then
        if [ ! -f "$BASE_CKPT_DIR/.run4060" ]; then
            echo "ERROR: $BASE_CKPT_DIR already has checkpoints that were not created by this script."
            echo "Pick another name, e.g. TAG=${TAG}-v2, or move that directory away."
            return 1
        elif [ "$(cat "$BASE_CKPT_DIR/.run4060")" != "$config" ]; then
            echo "ERROR: $BASE_CKPT_DIR was created with '$(cat "$BASE_CKPT_DIR/.run4060")' but this run is '$config'."
            echo "Use the same settings, a different TAG, or delete that directory to start over."
            return 1
        fi
    fi
    # same for the SFT dir, checked now rather than after hours of pretraining
    if ls "$SFT_CKPT_DIR"/model_*.pt &> /dev/null && [ ! -f "$SFT_CKPT_DIR/.run4060" ]; then
        echo "ERROR: $SFT_CKPT_DIR already has checkpoints that were not created by this script."
        echo "Move that directory away, or use another TAG."
        return 1
    fi
    if [ -f "$BASE_CKPT_DIR/.done" ]; then
        echo "Already finished, skipping (delete $BASE_CKPT_DIR to retrain; SFT is then redone too)"
    else
        mkdir -p "$BASE_CKPT_DIR" && echo "$config" > "$BASE_CKPT_DIR/.run4060"
        rm -f "$SFT_CKPT_DIR/.done" # a new base model invalidates any earlier SFT of this tag

        extra=()
        [ -n "${NUM_ITERATIONS:-}" ] && extra+=(--num-iterations="$NUM_ITERATIONS")
        resume=$(last_resumable_step)
        [ -n "$resume" ] && { echo "Resuming from step $resume"; extra+=(--resume-from-step="$resume"); }
        # checkpoint roughly every 30 minutes of work (steps get slower and bigger with depth)
        if   [ "$DEPTH" -le 8 ];  then save_every=500
        elif [ "$DEPTH" -le 10 ]; then save_every=150
        else save_every=100; fi
        python -m scripts.base_train \
            --depth="$DEPTH" \
            --device-batch-size="$B" \
            --eval-every=250 \
            --eval-tokens="$EVAL_TOKENS" \
            --core-metric-every=-1 \
            --sample-every=-1 \
            --save-every="$save_every" \
            --model-tag="$TAG" \
            --run="$WANDB_RUN" \
            "${extra[@]}" || return 1
        touch "$BASE_CKPT_DIR/.done"
        if [ -z "${KEEP_CKPTS:-}" ]; then # intermediate checkpoints were only needed for resuming
            final=$(last_resumable_step)
            [ -n "$final" ] && find "$BASE_CKPT_DIR" -maxdepth 1 -type f \( -name 'model_*' -o -name 'optim_*' -o -name 'meta_*' \) ! -name "*_$(printf "%06d" "$final")*" -delete
        fi
    fi

    stage "Base model eval (CORE, bpb, samples)"
    python -m scripts.base_eval --model-tag="$TAG" --device-batch-size="$EVAL_B" --split-tokens="$EVAL_TOKENS" --max-per-task="$CORE_MAX_PER_TASK" \
        || echo "WARNING: base eval failed, continuing"

    # -------------------------------------------------------------------------
    stage "SFT (chat finetuning)"
    if [ -f "$SFT_CKPT_DIR/.done" ]; then
        echo "Already finished, skipping (delete $SFT_CKPT_DIR to redo)"
    else
        if ls "$SFT_CKPT_DIR"/model_*.pt &> /dev/null; then
            # leftovers of our own earlier SFT would win if their step number is higher (loaders pick the max step)
            find "$SFT_CKPT_DIR" -maxdepth 1 -type f \( -name 'model_*' -o -name 'optim_*' -o -name 'meta_*' \) -delete
        fi
        mkdir -p "$SFT_CKPT_DIR" && touch "$SFT_CKPT_DIR/.run4060"
        extra=()
        [ -n "${SFT_ITERATIONS:-}" ] && extra+=(--num-iterations="$SFT_ITERATIONS")
        # ChatCORE during training is disabled, the full chat eval runs right after instead
        python -m scripts.chat_sft \
            --model-tag="$TAG" \
            --device-batch-size="$B" \
            --eval-every=200 \
            --eval-tokens="$EVAL_TOKENS" \
            --chatcore-every=-1 \
            --run="$WANDB_RUN" \
            "${extra[@]}" || return 1
        touch "$SFT_CKPT_DIR/.done"
    fi

    stage "Chat eval"
    python -m scripts.chat_eval -i sft -g "$TAG" ${CHAT_EVAL_MAX_PROBLEMS:+-x "$CHAT_EVAL_MAX_PROBLEMS"} \
        || echo "WARNING: chat eval failed"

    stage "Done"
    echo "Talk to your model:"
    echo "  source .venv/bin/activate && python -m scripts.chat_cli -g $TAG"
}

main 2>&1 | tee -ai "$LOG"
exit "${PIPESTATUS[0]}"
