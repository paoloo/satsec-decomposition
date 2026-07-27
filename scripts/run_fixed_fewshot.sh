#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 MODEL MODEL_REVISION OUTDIR" >&2
  exit 2
fi
MODEL="$1"; REV="$2"; OUTDIR="$3"
mkdir -p "$OUTDIR"
export PYTHONPATH="${PYTHONPATH:-}:src"
SIZE="$(basename "$MODEL")"
for SEED in 0 1 2 3 4; do
  python3 benchmark/decomp_baseline_generate.py \
    --base "$MODEL" --model-revision "$REV" --data data/tuning_set.v2.jsonl \
    --strategy fewshot --shots 2 --exemplar-seed 0 \
    --temperature 0.7 --seed "$SEED" \
    --config "base+2shot-fixed ($SIZE)" \
    --out "$OUTDIR/fewshot_fixed_${SEED}.jsonl"
done
