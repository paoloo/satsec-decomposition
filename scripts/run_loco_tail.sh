#!/usr/bin/env bash
# Parallel/resume helper: generate LOCO folds START_FOLD..23 only.
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "usage: $0 MODEL MODEL_REVISION SIZE_TAG OUTDIR START_FOLD" >&2
  exit 2
fi
MODEL="$1"; REV="$2"; TAG="$3"; OUTDIR="$4"; START_FOLD="$5"
export PYTHONPATH="${PYTHONPATH:-}:src"
FOLDS="$OUTDIR/folds"
python3 tools/loco_split.py --data data/tuning_set.v2.jsonl --out-dir "$FOLDS"
mkdir -p "$OUTDIR/adapters" "$OUTDIR/predictions"

for FOLD_JSONL in "$FOLDS"/fold_*/tuning_set.jsonl; do
  FOLD_DIR="$(dirname "$FOLD_JSONL")"
  FOLD="$(basename "$FOLD_DIR")"
  FOLD_NUMBER="${FOLD#fold_}"
  FOLD_NUMBER="${FOLD_NUMBER%%_*}"
  if (( 10#$FOLD_NUMBER < START_FOLD )); then
    continue
  fi
  ADAPTER="$OUTDIR/adapters/$FOLD"
  if [ ! -s "$ADAPTER/adapter_model.safetensors" ]; then
    python3 -m satsec.training.train_lora \
      --data "$FOLD_JSONL" --base-model "$MODEL" --model-revision "$REV" \
      --out "$ADAPTER" --save-strategy no
  fi
  if [ ! -s "$OUTDIR/predictions/${FOLD}_adapter.jsonl" ]; then
    python3 benchmark/decomp_generate.py \
      --base "$MODEL" --model-revision "$REV" --adapter "$ADAPTER" \
      --config "+adapter LOCO greedy ($TAG)" --data "$FOLD_JSONL" \
      --temperature 0 --seed 0 --out "$OUTDIR/predictions/${FOLD}_adapter.jsonl"
  fi
  if [ ! -s "$OUTDIR/predictions/${FOLD}_schema.jsonl" ]; then
    python3 benchmark/decomp_baseline_generate.py \
      --base "$MODEL" --model-revision "$REV" --strategy schema \
      --config "base+schema LOCO greedy ($TAG)" --data "$FOLD_JSONL" \
      --temperature 0 --seed 0 --out "$OUTDIR/predictions/${FOLD}_schema.jsonl"
  fi
  if [ ! -s "$OUTDIR/predictions/${FOLD}_fewshot.jsonl" ]; then
    python3 benchmark/decomp_baseline_generate.py \
      --base "$MODEL" --model-revision "$REV" --strategy fewshot --shots 2 \
      --exemplar-seed 0 --config "base+2shot LOCO greedy ($TAG)" --data "$FOLD_JSONL" \
      --temperature 0 --seed 0 --out "$OUTDIR/predictions/${FOLD}_fewshot.jsonl"
  fi
done
