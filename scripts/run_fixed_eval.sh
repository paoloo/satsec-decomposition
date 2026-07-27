#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "usage: $0 MODEL ADAPTER MODEL_REVISION OUTDIR" >&2
  exit 2
fi
MODEL="$1"; ADAPTER="$2"; REV="$3"; OUTDIR="$4"
mkdir -p "$OUTDIR"
export PYTHONPATH="${PYTHONPATH:-}:src"
SIZE="$(basename "$MODEL")"
for SEED in 0 1 2 3 4; do
  COMMON=(--base "$MODEL" --model-revision "$REV" --data data/tuning_set.v2.jsonl \
    --temperature 0.7 --seed "$SEED")
  python3 benchmark/decomp_generate.py "${COMMON[@]}" --config "candidate-only ($SIZE)" \
    --out "$OUTDIR/base_${SEED}.jsonl"
  python3 benchmark/decomp_baseline_generate.py "${COMMON[@]}" --strategy schema \
    --config "base+schema ($SIZE)" --out "$OUTDIR/schema_${SEED}.jsonl"
  python3 benchmark/decomp_baseline_generate.py "${COMMON[@]}" --strategy fewshot --shots 2 \
    --exemplar-seed 0 \
    --config "base+2shot ($SIZE)" --out "$OUTDIR/fewshot_${SEED}.jsonl"
  if [ -n "$ADAPTER" ]; then
    python3 benchmark/decomp_generate.py "${COMMON[@]}" --adapter "$ADAPTER" \
      --config "+adapter ($SIZE)" --out "$OUTDIR/adapter_${SEED}.jsonl"
    python3 benchmark/decomp_generate.py "${COMMON[@]}" --adapter "$ADAPTER" \
      --type next_step --config "+adapter teacher-forced ($SIZE)" \
      --out "$OUTDIR/teacher_adapter_${SEED}.jsonl"
    python3 benchmark/next_step_rollout.py "${COMMON[@]}" --adapter "$ADAPTER" \
      --config "+adapter rollout ($SIZE)" --max-new-tokens 192 \
      --out "$OUTDIR/rollout_adapter_${SEED}.jsonl"
  fi
  python3 benchmark/decomp_generate.py "${COMMON[@]}" --type next_step \
    --config "base teacher-forced ($SIZE)" \
    --out "$OUTDIR/teacher_base_${SEED}.jsonl"
  python3 benchmark/next_step_rollout.py "${COMMON[@]}" \
    --config "base rollout ($SIZE)" --max-new-tokens 192 \
    --out "$OUTDIR/rollout_base_${SEED}.jsonl"
done
python3 benchmark/decomp_poscopy_baseline.py --data data/tuning_set.v2.jsonl \
  --out "$OUTDIR/poscopy.jsonl"
