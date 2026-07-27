#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "usage: $0 MODEL MODEL_REVISION SIZE_TAG PRIMARY_ADAPTER OUTDIR" >&2
  exit 2
fi
MODEL="$1"; REV="$2"; TAG="$3"; PRIMARY_ADAPTER="$4"; OUTDIR="$5"
export PYTHONPATH="${PYTHONPATH:-}:src"
mkdir -p "$OUTDIR/adapters" "$OUTDIR/predictions"

for TRAIN_SEED in 43 44; do
  python3 -m satsec.training.train_lora \
    --data data/tuning_set.v2.jsonl --base-model "$MODEL" --model-revision "$REV" \
    --seed "$TRAIN_SEED" --save-strategy no \
    --out "$OUTDIR/adapters/seed_${TRAIN_SEED}"
done

for TRAIN_SEED in 42 43 44; do
  if [ "$TRAIN_SEED" -eq 42 ]; then
    ADAPTER="$PRIMARY_ADAPTER"
  else
    ADAPTER="$OUTDIR/adapters/seed_${TRAIN_SEED}"
  fi
  python3 benchmark/decomp_generate.py \
    --base "$MODEL" --model-revision "$REV" --adapter "$ADAPTER" \
    --config "+adapter train-seed=${TRAIN_SEED} ($TAG)" \
    --data data/tuning_set.v2.jsonl --temperature 0 --seed 0 \
    --out "$OUTDIR/predictions/train_seed_${TRAIN_SEED}_greedy.jsonl"
done

python3 tools/merge_predictions.py --glob "$OUTDIR/predictions/*.jsonl" \
  --out "$OUTDIR/training_seed_all.jsonl"
python3 -m satsec.training.decomp_score \
  --pred "$OUTDIR/training_seed_all.jsonl" --data data/tuning_set.v2.jsonl \
  --json-out "$OUTDIR/training_seed_scores.json"
