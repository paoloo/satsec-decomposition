#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python3 -m satsec.training.build_tuning_set \
  --enrich-nextstep --distractors 8 --out data/tuning_set.v2.jsonl
