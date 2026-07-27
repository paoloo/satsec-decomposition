#!/usr/bin/env bash
set -euo pipefail

# Collect the detached 7B evaluation left on atadev, then merge and score every
# completed fixed-split run. Run from any directory; local Python always uses mcp.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="/home/paolo/t/decomp-adapt-v2/code"

if ssh atadev "docker ps --filter name=^satsec-eval-v2-7b$ --format '{{.Names}}'" | grep -q .; then
  echo "satsec-eval-v2-7b is still running. Progress:" >&2
  ssh atadev "tail -12 $REMOTE_ROOT/artifacts/results/eval_qwen2.5-7b-v2.log"
  echo "Run this script again after the container finishes." >&2
  exit 3
fi

if ! ssh atadev "tail -1 $REMOTE_ROOT/artifacts/results/eval_qwen2.5-7b-v2.log" \
    | grep -q "positional-copy predictions"; then
  echo "The 7B container stopped without the expected completion marker." >&2
  echo "Inspect: ssh atadev 'tail -80 $REMOTE_ROOT/artifacts/results/eval_qwen2.5-7b-v2.log'" >&2
  exit 4
fi

mkdir -p "$ROOT/artifacts/raw_predictions" "$ROOT/artifacts/results"
for size in 0.5b 1.5b 7b; do
  mkdir -p "$ROOT/artifacts/raw_predictions/fixed_$size"
  rsync -az "atadev:$REMOTE_ROOT/artifacts/raw_predictions/fixed_$size/" \
    "$ROOT/artifacts/raw_predictions/fixed_$size/"
done
rsync -az "atadev:$REMOTE_ROOT/artifacts/results/eval_qwen2.5-"'*-v2.log' \
  "$ROOT/artifacts/results/"

cd "$ROOT"
for size in 0.5b 1.5b 7b; do
  conda run -n mcp python tools/merge_predictions.py \
    --glob "artifacts/raw_predictions/fixed_$size/*.jsonl" \
    --out "artifacts/raw_predictions/fixed_${size}_all.jsonl"
  PYTHONPATH=.:src conda run -n mcp python -m satsec.training.decomp_score \
    --data data/tuning_set.v2.jsonl \
    --pred "artifacts/raw_predictions/fixed_${size}_all.jsonl" \
    --json-out "artifacts/results/fixed_${size}_scores.json"
done

echo "Collection and scoring complete."
echo "Results: $ROOT/artifacts/results/fixed_{0.5b,1.5b,7b}_scores.json"
